"""Synthetic tests for voxa/refmask.py reference masking."""

from __future__ import annotations

import numpy as np

from voxa.refmask import ReferenceMask, user_present

SR = 16000


def make_ref(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 0.1).astype(np.float32)


def echo_only(n: int, delay: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    ref = make_ref(n, seed)
    mic = np.zeros(n, dtype=np.float32)
    mic[delay:] = ref[: n - delay]
    return mic, ref


def with_user_tone(n: int, delay: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mic, ref = echo_only(n, delay, seed)
    t = np.arange(n) / SR
    tone = 0.5 * np.sin(2.0 * np.pi * 2500.0 * t)
    mic = mic + tone.astype(np.float32)
    return mic, ref


def energy(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def test_silent_reference_bit_identical():
    n = 4 * SR
    mic = (np.random.default_rng(1).standard_normal(n) * 0.05).astype(np.float32)
    ref = np.zeros(n, dtype=np.float32)
    mask = ReferenceMask()
    out = mask.process(mic, ref)
    assert np.array_equal(out, mic)


def test_echo_only_is_reduced():
    n = 4 * SR
    mic, ref = echo_only(n, 100)
    mask = ReferenceMask()
    out = mask.process(mic, ref)
    assert energy(out) < 0.5 * energy(mic)


def test_user_tone_is_preserved():
    n = 4 * SR
    mic, ref = with_user_tone(n, 100)
    mask = ReferenceMask()
    out = mask.process(mic, ref)
    assert energy(out) > 0.5 * energy(mic)


def test_user_present_with_tone():
    n = 4 * SR
    mic, ref = with_user_tone(n, 100)
    mask = ReferenceMask()
    mask.process(mic, ref)
    assert user_present(mask)


def test_user_absent_echo_only():
    n = 4 * SR
    mic, ref = echo_only(n, 100)
    mask = ReferenceMask()
    mask.process(mic, ref)
    assert not user_present(mask)
