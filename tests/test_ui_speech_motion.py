from __future__ import annotations

from voxa.ui.speech_motion import speech_level


def test_speech_level_stays_in_range() -> None:
    for i in range(2000):
        value = speech_level(i * 0.005)
        assert 0.0 <= value <= 1.0


def test_speech_level_is_deterministic() -> None:
    assert speech_level(1.234, 7) == speech_level(1.234, 7)
    assert speech_level(0.0) == speech_level(0.0)


def test_speech_level_negative_time_is_zero() -> None:
    assert speech_level(-1.0) == 0.0
    assert speech_level(-0.001) == 0.0


def test_speech_level_shape_over_ten_seconds() -> None:
    samples = [speech_level(i * 0.005) for i in range(2000)]
    mean = sum(samples) / len(samples)
    assert 0.2 <= mean <= 0.7

    quiet = sum(1 for value in samples if value < 0.05)
    assert quiet >= 0.05 * len(samples)

    loud = sum(1 for value in samples if value > 0.6)
    assert loud >= 0.10 * len(samples)


def test_speech_level_varies_with_seed() -> None:
    first = [speech_level(i * 0.01, seed=1) for i in range(100)]
    second = [speech_level(i * 0.01, seed=2) for i in range(100)]
    assert first != second
