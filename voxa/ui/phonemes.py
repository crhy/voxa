"""Rule-based English grapheme-to-viseme mapping for word-accurate lip sync.

Pure module: no GTK, no network.  :func:`word_to_visemes` turns one word into
viseme shape names (the same names :mod:`voxa.ui.visemes` drives the mouth
morphs with) using a small spelling rule table: digraphs first, then single
letters, with repeats merged.  :func:`timeline` spreads those shapes across
the real word timings Edge TTS reports, and :func:`weights_at` turns a
playback time into at most two blended morph weights with a short
cross-fade and a little anticipation so a shape starts slightly before its
time.
"""

from __future__ import annotations

from .visemes import DROP_EPS, VOWELS  # noqa: E402

DIGRAPHS: dict[str, str] = {
    "th": "viseme_TH",
    "ch": "viseme_CH",
    "sh": "viseme_CH",
    "ph": "viseme_FF",
    "oo": "viseme_U",
    "ou": "viseme_U",
    "ew": "viseme_U",
    "ee": "viseme_I",
    "ea": "viseme_I",
    "ie": "viseme_I",
    "ai": "viseme_E",
    "ay": "viseme_E",
    "oa": "viseme_O",
    "ow": "viseme_O",
}

LETTERS: dict[str, str | None] = {
    "p": "viseme_PP",
    "b": "viseme_PP",
    "m": "viseme_PP",
    "f": "viseme_FF",
    "v": "viseme_FF",
    "t": "viseme_DD",
    "d": "viseme_DD",
    "k": "viseme_kk",
    "g": "viseme_kk",
    "c": "viseme_kk",
    "q": "viseme_kk",
    "x": "viseme_kk",
    "s": "viseme_SS",
    "z": "viseme_SS",
    "n": "viseme_nn",
    "l": "viseme_nn",
    "r": "viseme_RR",
    "a": "viseme_aa",
    "e": "viseme_E",
    "i": "viseme_I",
    "y": "viseme_I",
    "o": "viseme_O",
    "u": "viseme_U",
    "w": "viseme_U",
    "h": None,
}

VOWEL_SHARE = 1.6
SILENCE_GAP = 0.18
ANTICIPATION = 0.040
CROSSFADE = 0.060


def word_to_visemes(word: str) -> list[str]:
    """Return the viseme shapes for one word, repeats merged.

    Spelling rules, in order at each position: a known digraph wins
    (:data:`DIGRAPHS`); an ``o`` directly before ``v`` is the ``oo`` sound of
    *move* and maps to ``viseme_U`` without consuming the ``v``; a final
    ``e`` on a multi-letter word is silent; ``h`` is silent; every other
    letter maps through :data:`LETTERS`.
    """
    letters = [char for char in word.lower() if "a" <= char <= "z"]
    shapes: list[str] = []
    index = 0
    total = len(letters)
    while index < total:
        pair = "".join(letters[index : index + 2])
        if len(pair) == 2 and pair in DIGRAPHS:
            shapes.append(DIGRAPHS[pair])
            index += 2
            continue
        char = letters[index]
        if char == "o" and index + 1 < total and letters[index + 1] == "v":
            shapes.append("viseme_U")
            index += 1
            continue
        if char == "e" and index == total - 1 and total > 1:
            index += 1
            continue
        shape = LETTERS.get(char)
        if shape:
            shapes.append(shape)
        index += 1

    merged: list[str] = []
    for shape in shapes:
        if not merged or merged[-1] != shape:
            merged.append(shape)
    return merged


def timeline(words: list[tuple[str, float, float]]) -> list[tuple[float, float, str]]:
    """Spread each word's visemes across its audio duration.

    ``words`` is ``(word, start_s, duration_s)`` per word, times relative to
    the start of the utterance.  Vowels get :data:`VOWEL_SHARE` times the
    share of consonants.  Gaps longer than :data:`SILENCE_GAP` seconds become
    ``viseme_sil`` segments.  Returns ``(start, end, viseme)`` segments
    ordered by time.
    """
    segments: list[tuple[float, float, str]] = []
    previous_end: float | None = None
    for word, start, duration in sorted(words, key=lambda entry: entry[1]):
        if previous_end is not None and start - previous_end > SILENCE_GAP:
            segments.append((previous_end, start, "viseme_sil"))
        shapes = word_to_visemes(word)
        if shapes:
            weights = [VOWEL_SHARE if shape in VOWELS else 1.0 for shape in shapes]
            total_weight = sum(weights)
            cursor = start
            for shape, weight in zip(shapes, weights, strict=True):
                span = duration * weight / total_weight
                segments.append((cursor, cursor + span, shape))
                cursor += span
        if previous_end is None or start + duration > previous_end:
            previous_end = start + duration
    return segments


def weights_at(segments: list[tuple[float, float, str]], t: float) -> dict[str, float]:
    """Morph weights at playback time ``t`` for a :func:`timeline` result.

    A shape begins :data:`ANTICIPATION` seconds before its time and blends
    in from the previous shape over :data:`CROSSFADE` seconds, so at most
    two visemes are ever non-zero.  Times before the first segment or past
    the last return an empty dict.
    """
    current: tuple[int, float, float, str] | None = None
    for index, (start, end, shape) in enumerate(segments):
        if t >= start - ANTICIPATION:
            current = (index, start, end, shape)
        else:
            break
    if current is None:
        return {}
    index, start, end, shape = current
    if t >= end:
        return {}
    effective_start = start - ANTICIPATION
    fade = (t - effective_start) / CROSSFADE
    if index > 0 and fade < 1.0:
        alpha = min(1.0, max(0.0, fade))
        weights = {shape: alpha}
        previous = segments[index - 1][2]
        if previous != shape:
            weights[previous] = 1.0 - alpha
        return {key: value for key, value in weights.items() if value >= DROP_EPS}
    return {shape: 1.0}
