"""Unit tests for the near-miss wake matcher in tools/wake_replay.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from wake_replay import NEAR_MISSES, near_miss_wake


def test_each_near_miss_matches_at_the_start():
    for phrase in NEAR_MISSES:
        assert near_miss_wake(phrase) is not None
        assert near_miss_wake(f"{phrase} what time is it") is not None


def test_near_miss_must_be_the_first_word():
    assert near_miss_wake("the boxer won the race") is None
    assert near_miss_wake("I heard a vox a minute ago") is None
    assert near_miss_wake("tell me about locks a drawer") is None
    assert near_miss_wake("the bucks a month plan") is None


def test_vox_does_not_match_inside_a_longer_word():
    assert near_miss_wake("voxel what time is it") is None
    assert near_miss_wake("voxels open the calculator") is None


def test_leading_punctuation_and_case_are_tolerated():
    assert near_miss_wake("“Boxer, what time is it?") is not None
    assert near_miss_wake("BOXA pause.") is not None
    assert near_miss_wake("Loxa, how much disk space is left?") is not None


def test_bare_near_miss_returns_empty_command():
    assert near_miss_wake("vox") == ""
    assert near_miss_wake("Boxer.") == ""


def test_command_text_survives():
    assert near_miss_wake("boxer what time is it") == "what time is it"
    assert near_miss_wake("bucks a play some jazz") == "play some jazz"
