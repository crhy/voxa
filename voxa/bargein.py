"""Decide whether a loud sound in the microphone is the user interrupting.

Pure and model-free: word overlap for the own-voice check, and a gate that
works from sound levels alone before any transcription exists.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

OWN_VOICE_OVERLAP = 0.6
BASELINE_ALPHA = 0.05
SETTLED_SAMPLES = 20
WINDOW = 8


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


STOP_WORDS = frozenset({"stop", "wait", "voxa", "cancel", "pause", "quiet", "enough", "no", "hey", "silence", "shush"})


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def sounds_like(heard: str, spoken: str) -> float:
    """How closely `heard` matches some stretch of `spoken`, letter by letter (0..1).

    Whisper often mishears Voxa's own voice into different WORDS that keep most of the LETTERS
    ("BLC doesn't seem to be open" came back as "Bmc, done, themes"), so words alone cannot tell
    her voice from a person's.
    """
    h, s = _letters(heard), _letters(spoken)
    if not h or not s:
        return 0.0
    if len(s) <= len(h):
        return SequenceMatcher(None, h, s).ratio()
    best = 0.0
    step = max(1, len(h) // 4)
    for width in (len(h), int(len(h) * 1.3) + 1):
        for start in range(0, max(1, len(s) - width + 1), step):
            best = max(best, SequenceMatcher(None, h, s[start:start + width]).ratio())
    return best


OWN_VOICE_SOUND = 0.5


def is_own_voice(heard: str, speaking_now: str, recently_spoken: str = "", raw: str = "") -> bool:
    """True when what was heard is (a mishearing of) what Voxa is saying, or too little to be a person.

    `heard` is the transcript after her own words were stripped; `raw` is the transcript before that.
    """
    spoken_text = f"{speaking_now} {recently_spoken}"
    words = _words(heard)
    spoken = set(_words(spoken_text))
    stop = [w for w in words if w in STOP_WORDS and w not in spoken and not any(s.startswith(w) for s in spoken)]
    if stop and (len(words) <= 3 or sounds_like(raw or heard, spoken_text) < OWN_VOICE_SOUND):
        # The words people actually interrupt with count — unless she is saying them herself
        # ("Stopping dictation"), or they sit inside a long mishearing of her own sentence.
        return False
    if len(words) < 3:
        # One or two stray words are usually her own voice misheard ("a celibate"), not a person.
        return True
    if sounds_like(raw or heard, spoken_text) >= OWN_VOICE_SOUND:
        return True
    return overlap(heard, spoken_text) >= OWN_VOICE_OVERLAP


class BargeInGate:
    """Opens once `needed` of the last `WINDOW` calls tower over the level heard
    while Voxa speaks; a quieter frame only lowers the count, it does not reset."""

    def __init__(self, ratio: float = 2.0, min_level: float = 0.02, needed: int = 4):
        self._ratio = ratio
        self._min_level = min_level
        self._needed = needed
        self._count = 0
        self._recent: list[bool] = []
        self._baseline = 0.0
        self._samples = 0

    def update(self, mic_level: float, baseline: float) -> bool:
        # Speech dips between syllables, so "six loud ticks in a row" almost never happened for a
        # short "Voxa, stop" (traced: 374 ticks at full level, gate never opened). Count the loud
        # ticks among the last WINDOW instead.
        self._recent.append(mic_level > max(self._min_level, self._ratio * baseline))
        del self._recent[:-WINDOW]
        self._count = sum(self._recent)
        # Never open before the baseline has heard enough of her own voice to be trusted:
        # at session start the baseline is 0.0, so any loud sound would open the gate.
        return self.settled and self._count >= self._needed

    def note_speaking_level(self, mic_level: float) -> None:
        self._baseline += BASELINE_ALPHA * (mic_level - self._baseline)
        self._samples += 1

    @property
    def settled(self) -> bool:
        """True once the baseline has heard enough of her own voice to be trusted."""
        return self._samples >= SETTLED_SAMPLES

    @property
    def baseline(self) -> float:
        return self._baseline

    def reset(self) -> None:
        self._count = 0
        self._recent = []
