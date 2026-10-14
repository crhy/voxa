"""Pure tests for the ANC lab helpers: end-alignment, correlation, levels, mixing."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals

from tools import anc_lab
from voxa.anc import EchoCanceller


def test_align_ends_trims_front_of_longer():
    a = np.arange(10, dtype=np.float32)
    b = np.arange(4, dtype=np.float32)
    x, y = anc_lab.align_ends(a, b)
    assert len(x) == len(y) == 4
    assert np.array_equal(x, np.arange(6, 10, dtype=np.float32))
    assert np.array_equal(y, b)


def test_align_ends_equal_length_is_identity():
    a = np.arange(5, dtype=np.float32)
    x, y = anc_lab.align_ends(a, a)
    assert np.array_equal(x, a)
    assert np.array_equal(y, a)


def test_correlation_peak_finds_synthetic_echo():
    ref, echo, _speech, delay = anc_signals.make()
    peak, ratio = anc_lab.correlation_peak(echo, ref)
    assert delay <= peak <= delay + 400
    assert ratio > 8.0


def test_correlation_peak_noise_is_low():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(4000)
    b = rng.standard_normal(4000)
    _peak, ratio = anc_lab.correlation_peak(a, b)
    assert ratio < 8.0


def test_correlation_peak_silent_returns_zero():
    z = np.zeros(1000, dtype=np.float32)
    peak, ratio = anc_lab.correlation_peak(z, z)
    assert (peak, ratio) == (0, 0.0)


def test_wav_seconds_matches_written_wav(tmp_path):
    x = np.zeros(3 * 16000, dtype=np.float32)
    p = tmp_path / "t.wav"
    anc_lab.write_wav(p, x)
    assert abs(anc_lab.wav_seconds(p) - 3.0) < 1e-6


def test_mix_command_scales_to_relative_level():
    rng = np.random.default_rng(1)
    mic = rng.standard_normal(1000).astype(np.float32)
    cmd = np.ones(100, dtype=np.float32)
    mic2, off = anc_lab.mix_command(mic, cmd, 10, 0.1, -6.0)
    added = mic2[off:off + len(cmd)] - mic[off:off + len(cmd)]
    assert abs(anc_lab.rms(added) - 0.1 * 10.0 ** (-6.0 / 20.0)) < 1e-6


def test_word_accuracy_counts_in_order():
    assert anc_lab.word_accuracy("vox please stop dictation now", ["stop", "dictation"]) == 1.0
    assert anc_lab.word_accuracy("dictation stop", ["stop", "dictation"]) == 0.5


def test_removed_percent_halved_energy():
    mic = np.full(320, 0.5, dtype=np.float32)
    ref = np.full(320, 0.3, dtype=np.float32)
    cleaned = mic * 0.5
    assert abs(anc_lab.removed_percent(mic, ref, cleaned) - 75.0) < 1e-6


def test_feed_returns_full_length():
    ref, echo, _speech, delay = anc_signals.make()
    cleaned, _converged = anc_lab.feed(EchoCanceller(delay=delay), echo, ref)
    assert len(cleaned) == len(echo)
