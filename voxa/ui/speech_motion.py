from __future__ import annotations

import math


def speech_level(t: float, seed: int = 0) -> float:
    """Deterministic mouth-open amount (0.0-1.0) for time t seconds of speech."""
    if t < 0:
        return 0.0

    phase = t * 5.0
    syllable = int(phase)
    frac = phase - syllable

    hash_value = (syllable * 2654435761 + seed * 40503) % 1000
    if hash_value % 11 == 0:
        return 0.0

    amplitude = 0.35 + 0.65 * (hash_value / 1000)
    value = amplitude * math.sin(math.pi * frac)
    return max(0.0, min(1.0, value))
