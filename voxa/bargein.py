"""Decide whether a loud sound in the microphone is the user interrupting.

Pure and model-free: word overlap for the own-voice check, and a gate that
works from sound levels alone before any transcription exists.
"""

from __future__ import annotations

import re

OWN_VOICE_OVERLAP = 0.6
BASELINE_ALPHA = 0.05


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"[^\W\d_]+", text.lower()) if len(word) >= 2]


def overlap(heard: str, speaking: str) -> float:
    """Fraction of the words in `heard` (lower-cased, punctuation stripped,
    words of 2+ letters) that also occur in `speaking`; 0.0 when `heard` has
    no such words."""
    heard_words = _words(heard)
    if not heard_words:
        return 0.0
    speaking_words = set(_words(speaking))
    hits = sum(1 for word in heard_words if word in speaking_words)
    return hits / len(heard_words)


def is_own_voice(heard: str, speaking_now: str, recently_spoken: str = "") -> bool:
    """True when `heard` is empty/noise (fewer than 2 words) or mostly words
    Voxa was already saying, i.e. the loud sound was her own voice."""
    if len(_words(heard)) < 2:
        return True
    return overlap(heard, f"{speaking_now} {recently_spoken}") >= OWN_VOICE_OVERLAP


class BargeInGate:
    """Opens only after `needed` consecutive calls whose mic level towers
    over the level heard while Voxa speaks; one quieter frame resets."""

    def __init__(self, ratio: float = 2.5, min_level: float = 0.02, needed: int = 6):
        self._ratio = ratio
        self._min_level = min_level
        self._needed = needed
        self._count = 0
        self._baseline = 0.0

    def update(self, mic_level: float, baseline: float) -> bool:
        if mic_level > max(self._min_level, self._ratio * baseline):
            self._count += 1
        else:
            self._count = 0
        return self._count >= self._needed

    def note_speaking_level(self, mic_level: float) -> None:
        self._baseline += BASELINE_ALPHA * (mic_level - self._baseline)

    @property
    def baseline(self) -> float:
        return self._baseline

    def reset(self) -> None:
        self._count = 0
