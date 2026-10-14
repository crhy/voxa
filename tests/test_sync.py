"""Synthetic tests for voxa/sync.py drift tracking and resampling."""

from __future__ import annotations

import numpy as np

from voxa.anc import erle_db
from voxa.sync import DriftTracker, align_pair, band_coherence, resample_to

SR = 16000


def make_reference(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def build_mic(ref: np.ndarray, drift_ppm: float, delay: int, noise: float, seed: int = 7) -> np.ndarray:
    """Mic = reference resampled at +drift_ppm, shifted right by delay, plus noise."""
    ratio = 1.0 + drift_ppm * 1e-6
    drifted = resample_to(ref, len(ref), ratio=ratio, frac=0.0)
    mic = np.zeros(len(ref), dtype=np.float64)
    mic[delay:] = drifted[: len(ref) - delay]
    rng = np.random.default_rng(seed)
    mic += noise * rng.standard_normal(len(ref))
    return mic


def test_resample_identity():
    ref = make_reference(4000)
    out = resample_to(ref, 4000, ratio=1.0, frac=0.0)
    assert np.allclose(out, ref, atol=1e-9)


def test_resample_stretch():
    ref = np.arange(100, dtype=np.float64)
    out = resample_to(ref, 50, ratio=2.0, frac=0.0)
    assert np.allclose(out, np.arange(0, 100, 2, dtype=np.float64))


def test_align_recovers_coherence():
    n = 16000
    ref = make_reference(n, seed=3)
    mic = build_mic(ref, drift_ppm=300.0, delay=1200, noise=0.05, seed=11)
    aligned = align_pair(mic, ref)
    coh = band_coherence(mic, aligned, fmin=300.0, fmax=3000.0)
    assert coh > 0.9, f"coherence {coh:.3f} too low"


def test_filter_erle_after_align():
    n = 16000
    ref = make_reference(n, seed=5)
    mic = build_mic(ref, drift_ppm=300.0, delay=1200, noise=0.02, seed=13)
    aligned = align_pair(mic, ref)
    taps = 512
    x = aligned
    y = mic
    A = np.lib.stride_tricks.sliding_window_view(x, taps)
    b = y[taps - 1: taps - 1 + A.shape[0]]
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = A @ coef
    err = erle_db(b, b - pred)
    assert err >= 25.0, f"ERLE {err:.1f} dB too low"


def test_drift_tracker_slope():
    n = 32000
    ref = make_reference(n, seed=9)
    mic = build_mic(ref, drift_ppm=300.0, delay=1500, noise=0.02, seed=17)
    tr = DriftTracker(probe_every=4000, window=4096)
    step = tr._probe_every
    for i in range(0, n, step):
        tr.observe(mic[i:i + step], ref[i:i + step], i)
    # slope should be near -300 ppm (mic clock faster than ref)
    assert abs(tr._slope + 300e-6) < 50e-6, f"slope {tr._slope} not near -300ppm"
