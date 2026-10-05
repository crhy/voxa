"""Deterministic viseme weights that drive a MakeHuman character's mouth.

A MakeHuman-built character carries fifteen morph targets whose names match the
:class:`VOWELS` and :class:`CONSONANTS` tuples below.  While Voxa speaks we want
the mouth to move through real viseme shapes rather than a single jaw slide, so
this module turns a time-since-speaking-began into a small dictionary of morph
weights (at most two non-zero at a time).

The syllable timing, amplitude and pause rule are shared with
:func:`voxa.ui.speech_motion.speech_level` so the flat-path mouth and the
morph-path mouth agree on *when* the mouth is open, even though the morph path
also decides *which* shape it opens into.
"""

from __future__ import annotations

import math

VOWELS = ("viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U")
CONSONANTS = (
    "viseme_PP",
    "viseme_FF",
    "viseme_TH",
    "viseme_DD",
    "viseme_kk",
    "viseme_CH",
    "viseme_SS",
    "viseme_nn",
    "viseme_RR",
)

SYL_PER_SEC = 5.0
VOWEL_LEAD = 0.15
VOWEL_SPAN = 0.85
CONSONANT_TAIL = 0.25
DROP_EPS = 0.01


def viseme_weights(t: float, seed: int = 0) -> dict[str, float]:
    """Return morph weights (0..1) for time ``t`` seconds of speech.

    Deterministic for a given ``(t, seed)`` pair.  At most two weights are
    non-zero; a pause syllable yields ``{"viseme_sil": 1.0}`` and ``t < 0``
    yields an empty dict.
    """
    if t < 0:
        return {}

    phase = t * SYL_PER_SEC
    syllable = int(phase)
    frac = phase - syllable

    hash_value = (syllable * 2654435761 + seed * 40503) % 1000
    if hash_value % 11 == 0:
        return {"viseme_sil": 1.0}

    amplitude = 0.35 + 0.65 * (hash_value / 1000)

    vowel_choice = (syllable * 2246822519 + seed * 3266489917 + 17) % len(VOWELS)
    consonant_choice = (syllable * 3266489917 + seed * 2654435761 + 911) % len(CONSONANTS)

    weights: dict[str, float] = {}

    if frac < CONSONANT_TAIL:
        consonant_weight = amplitude * max(0.0, 1.0 - frac / CONSONANT_TAIL)
        if consonant_weight > 0.0:
            weights[CONSONANTS[consonant_choice]] = min(1.0, max(0.0, consonant_weight))

    if frac >= VOWEL_LEAD:
        vowel_weight = amplitude * math.sin(math.pi * (frac - VOWEL_LEAD) / VOWEL_SPAN)
        if vowel_weight > 0.0:
            weights[VOWELS[vowel_choice]] = min(1.0, max(0.0, vowel_weight))

    return weights


def blend(prev: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
    """Linearly interpolate ``prev`` towards ``target`` by ``alpha``.

    Missing keys count as ``0.0``; keys whose blended value falls below
    :data:`DROP_EPS` are dropped so the mouth can close cleanly.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    blended: dict[str, float] = {}
    for key in set(prev) | set(target):
        source = prev.get(key, 0.0)
        goal = target.get(key, 0.0)
        value = source + (goal - source) * alpha
        if value >= DROP_EPS:
            blended[key] = value
    return blended
