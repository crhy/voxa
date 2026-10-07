"""Tests for the echo-delay measurement and energy-removal estimate."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals

from voxa.anc import erle_db, estimate_delay


def test_estimate_delay_echo_only():
    ref, echo, _speech, delay = anc_signals.make()
    got = estimate_delay(echo, ref)
    assert delay <= got <= delay + 400


def test_estimate_delay_with_speech():
    ref, echo, speech, delay = anc_signals.make()
    echo_only = estimate_delay(echo, ref)
    mixed = estimate_delay(echo + speech, ref)
    assert abs(mixed - echo_only) <= 32


def test_estimate_delay_silence():
    ref, echo, _speech, delay = anc_signals.make()
    assert estimate_delay(np.zeros_like(echo), ref) == 0
    assert estimate_delay(echo, np.zeros_like(ref)) == 0


def test_estimate_delay_unrelated_noise():
    ref, echo, _speech, delay = anc_signals.make()
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(len(echo)).astype(np.float32)
    assert estimate_delay(noise, ref) == 0


def test_estimate_delay_shifted_reference():
    ref, _echo, _speech, delay = anc_signals.make()
    shifted = np.concatenate([np.zeros(1000), ref])[: len(ref)]
    assert estimate_delay(shifted, ref) == 1000


def test_erle_db_ratio():
    x = np.linspace(-1.0, 1.0, 10001, dtype=np.float64)
    assert abs(erle_db(x, 0.1 * x) - 20.0) <= 0.01


def test_erle_db_silence():
    zeros = np.zeros(1000, dtype=np.float64)
    assert erle_db(zeros, zeros) == 0.0
