"""Synthetic checks for the Q12 shared-clock probe (issue #88): metrics and timeline placement.

Pure numpy, no audio devices.
"""

from __future__ import annotations

import numpy as np

from tools.anc_lab import SR, make_pink, oracle_erle_db, shift_forward
from tools.clock_probe import to_timeline
from voxa.sync import band_coherence


def test_band_coherence_on_linear_pair():
    ref = make_pink(3, 10.0)
    rng = np.random.default_rng(5)
    taps = 200
    h = np.zeros(taps)
    h[0] = 1.0
    h[1:] = 0.02 * rng.standard_normal(taps - 1) * np.exp(-np.arange(1, taps) / (0.02 * SR))
    delay = int(round(0.037 * SR))
    echo = shift_forward(np.convolve(ref, h, mode="full")[: len(ref)], delay)
    echo_rms = float(np.sqrt(np.mean(np.asarray(echo, dtype=np.float64) ** 2)))
    noise = rng.standard_normal(len(ref)) * (echo_rms / 10.0)
    mic = np.asarray(echo, dtype=np.float64) + noise
    coh = band_coherence(mic, shift_forward(ref, delay))
    assert coh > 0.9, f"band coherence {coh:.3f} too low for a linear pair"


def test_oracle_erle_on_linear_pair():
    ref = make_pink(3, 10.0)
    rng = np.random.default_rng(5)
    taps = 200
    h = np.zeros(taps)
    h[0] = 1.0
    h[1:] = 0.02 * rng.standard_normal(taps - 1) * np.exp(-np.arange(1, taps) / (0.02 * SR))
    delay = int(round(0.037 * SR))
    echo = shift_forward(np.convolve(ref, h, mode="full")[: len(ref)], delay)
    echo_rms = float(np.sqrt(np.mean(np.asarray(echo, dtype=np.float64) ** 2)))
    noise = rng.standard_normal(len(ref)) * (echo_rms / 10.0)
    mic = np.asarray(echo, dtype=np.float64) + noise
    erle = oracle_erle_db(mic, ref, delay)
    assert erle > 15.0, f"oracle ERLE {erle:.1f} dB too low for a linear pair"


def test_to_timeline_gap_placement():
    rate = SR
    a = np.ones(160, dtype=np.float64)
    b = np.full(160, 2.0)
    out, gaps, overlaps = to_timeline([(0, a), (20_000_000, b)], rate, 0)
    assert len(out) == 480
    assert np.all(out[0:160] == 1.0)
    assert np.all(out[160:320] == 0.0)
    assert np.all(out[320:480] == 2.0)
    assert gaps == 1
    assert overlaps == 0
