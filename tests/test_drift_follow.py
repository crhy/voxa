"""Tests for sub-sample clock-drift following in voxa/sync.py (issue #88)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voxa.sync import band_coherence_segments, fit_drift_fine, follow_drift, fractional_delay, predictable_share

SR = 16000


def decaying_response(ref: np.ndarray, taps: int) -> np.ndarray:
    """Pass ``ref`` through a ``taps``-tap decaying (room-reverberation) response."""
    k = np.array([0.7 ** i for i in range(taps)], dtype=np.float64)
    return np.convolve(ref, k)[: len(ref)]


def build_mic(ref: np.ndarray, delay0: float, drift: float, taps: int, noise_db: float,
              seed: int = 7) -> np.ndarray:
    """Mic = ref through a decaying response, delayed by ``delay0 + drift * t`` samples, plus noise."""
    echo = decaying_response(ref, taps)
    n = len(ref)
    mic = np.zeros(n, dtype=np.float64)
    block = SR
    for i in range(0, n, block):
        seg = echo[i:i + block]
        t = (i + len(seg) / 2.0) / SR
        mic[i:i + len(seg)] = fractional_delay(seg, delay0 + drift * t)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    power = float(np.mean(mic ** 2))
    scale = np.sqrt(power / (10.0 ** (noise_db / 10.0))) / (float(np.std(noise)) + 1e-12)
    return mic + noise * scale


def test_fit_drift_fine_synthetic():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.4, 200, 20.0)
    _lag0, drift, _resid = fit_drift_fine(mic, ref)
    assert abs(drift - 0.4) < 0.05


def test_follow_drift_beats_fixed_delay():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.4, 200, 20.0)
    lag0, drift, _resid = fit_drift_fine(mic, ref)
    mean_lag = lag0 + drift * ((len(mic) / 2.0) / SR)
    k = int(round(mean_lag))
    nn = len(mic) - 800
    fixed = band_coherence_segments([(mic[k:k + nn], ref[:nn])], 2000.0, 3000.0)
    mic_a, ref_a, _info = follow_drift(mic, ref)
    followed = band_coherence_segments([(mic_a, ref_a)], 2000.0, 3000.0)
    assert followed > 0.85
    assert fixed < 0.5


def test_real_recording():
    d = Path(".voxa-spec/out/anclab/drift")
    mic_path = d / "pink15_100_mic.npy"
    ref_path = d / "pink15_100_ref.npy"
    if not (mic_path.is_file() and ref_path.is_file()):
        return
    mic = np.asarray(np.load(mic_path), dtype=np.float64)
    ref = np.asarray(np.load(ref_path), dtype=np.float64)
    nn = min(len(mic), len(ref))
    mic, ref = mic[:nn], ref[:nn]
    thr = 10.0 ** (-50.0 / 20.0)
    nz = np.flatnonzero(np.abs(ref) > thr)
    on = (int(nz[0]) + SR // 2) if nz.size else 0
    mic, ref = mic[on:], ref[on:]
    _lag0, drift, _resid = fit_drift_fine(mic, ref)
    assert abs(drift - 0.32) < 0.05
    mic_a, ref_a, _info = follow_drift(mic, ref)
    assert band_coherence_segments([(mic_a, ref_a)], 1000.0, 2000.0) > 0.8
    assert band_coherence_segments([(mic_a, ref_a)], 2000.0, 3000.0) > 0.8


def test_real_bands_recordings():
    for name in ("bands_100", "bands_70"):
        d = Path(".voxa-spec/out/anclab/drift")
        mic_path = d / f"{name}_mic.npy"
        ref_path = d / f"{name}_ref.npy"
        if not (mic_path.is_file() and ref_path.is_file()):
            continue
        mic = np.asarray(np.load(mic_path), dtype=np.float64)
        ref = np.asarray(np.load(ref_path), dtype=np.float64)
        nn = min(len(mic), len(ref))
        mic, ref = mic[:nn], ref[:nn]
        thr = 10.0 ** (-50.0 / 20.0)
        nz = np.flatnonzero(np.abs(ref) > thr)
        on = (int(nz[0]) + SR // 2) if nz.size else 0
        mic, ref = mic[on:], ref[on:]
        lag0, drift, _resid = fit_drift_fine(mic, ref)
        assert abs(drift - 0.32) < 0.10
        mic_a, ref_a, _info = follow_drift(mic, ref)
        mean_lag = lag0 + drift * ((len(mic) / 2.0) / SR)
        k = int(round(mean_lag))
        nseg = len(mic) - 800
        fixed = [(mic[k:k + nseg], ref[:nseg])]
        followed = [(mic_a, ref_a)]
        for lo, hi in ((1000.0, 2000.0), (2000.0, 3000.0), (3000.0, 6000.0)):
            f_c = band_coherence_segments(fixed, lo, hi)
            d_c = band_coherence_segments(followed, lo, hi)
            assert d_c >= f_c - 0.02


def test_silence_in_middle():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.4, 200, 20.0)
    mid = 6 * SR
    gap = 3 * SR
    rng = np.random.default_rng(11)
    ins = rng.standard_normal(gap)
    echo_pow = float(np.mean(mic[:mid] ** 2))
    ins = ins * (np.sqrt(echo_pow / 100.0) / (float(np.std(ins)) + 1e-12))
    ref2 = np.concatenate([ref[:mid], np.zeros(gap), ref[mid:]])
    mic2 = np.concatenate([mic[:mid], ins, mic[mid:]])
    _lag0, drift, _resid = fit_drift_fine(mic2, ref2)
    assert abs(drift - 0.4) < 0.05


def test_no_drift_pair():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.0, 200, 20.0)
    lag0, drift, _resid = fit_drift_fine(mic, ref)
    assert abs(drift) < 0.03
    mean_lag = lag0 + drift * ((len(mic) / 2.0) / SR)
    k = int(round(mean_lag))
    nn = len(mic) - 800
    fixed = band_coherence_segments([(mic[k:k + nn], ref[:nn])], 2000.0, 3000.0)
    mic_a, ref_a, _info = follow_drift(mic, ref)
    followed = band_coherence_segments([(mic_a, ref_a)], 2000.0, 3000.0)
    assert followed >= fixed - 0.02


def test_predictable_share_echo_high():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.4, 200, 20.0)
    mic_a, ref_a, _info = follow_drift(mic, ref)
    share = predictable_share(mic_a, ref_a)
    assert share > 0.8


def test_predictable_share_noise_low():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(n)
    share = predictable_share(noise, ref)
    assert share < 0.1


def test_predictable_share_short_zero():
    ref = np.random.default_rng(1).standard_normal(4000)
    mic = np.random.default_rng(2).standard_normal(4000)
    assert predictable_share(mic, ref) == 0.0


def test_follow_drift_session_ppm():
    n = 12 * SR
    ref = np.random.default_rng(1).standard_normal(n)
    mic = build_mic(ref, 300.0, 0.4, 200, 20.0)
    true_ppm = 0.4 / SR * 1e6
    mic_a, ref_a, info = follow_drift(mic, ref, ppm=true_ppm)
    assert abs(info["ppm"] - true_ppm) < 0.5
    assert band_coherence_segments([(mic_a, ref_a)], 2000.0, 3000.0) > 0.85
