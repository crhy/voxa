from __future__ import annotations

from voxa.ui.visemes import CONSONANTS, VOWELS, blend, viseme_weights


def test_viseme_weights_are_deterministic() -> None:
    first = viseme_weights(1.234, seed=7)
    second = viseme_weights(1.234, seed=7)
    assert first == second


def test_viseme_weights_are_within_unit_range() -> None:
    for step in range(0, 200):
        weights = viseme_weights(step / 20.0)
        for value in weights.values():
            assert 0.0 <= value <= 1.0


def test_viseme_weights_have_at_most_two_nonzero() -> None:
    for step in range(0, 200):
        weights = viseme_weights(step / 20.0, seed=step % 5)
        assert len(weights) <= 2


def test_negative_time_returns_empty() -> None:
    assert viseme_weights(-0.5) == {}


def test_pause_returns_only_sil() -> None:
    found_pause = False
    for step in range(0, 200):
        weights = viseme_weights(step / 20.0)
        if "viseme_sil" in weights:
            assert weights == {"viseme_sil": 1.0}
            found_pause = True
    assert found_pause


def test_every_vowel_and_most_consonants_appear_over_ten_seconds() -> None:
    seen: set[str] = set()
    step = 0.0
    while step <= 10.0:
        seen.update(viseme_weights(step))
        step += 0.01
    assert set(VOWELS) <= seen
    assert len(seen & set(CONSONANTS)) >= 6


def test_blend_converges_to_target_within_twenty_steps() -> None:
    target = {"viseme_aa": 0.8, "viseme_PP": 0.4}
    current: dict[str, float] = {}
    for _ in range(20):
        current = blend(current, target, 0.45)
    assert set(current) == set(target)
    for key, value in target.items():
        assert abs(current[key] - value) < 0.01


def test_blend_towards_empty_closes_the_mouth() -> None:
    current = {"viseme_aa": 0.9, "viseme_sil": 0.5}
    for _ in range(30):
        current = blend(current, {}, 0.45)
    assert current == {}
