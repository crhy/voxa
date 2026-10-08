"""Echo filter for Voxa's own voice (issue #88).

Even with a perfect canceller some of her own voice leaks into the microphone.
She always knows the sentence she is saying, so a transcript that is (part of)
that sentence is echo, not the user. This is pure text work: no audio, no GTK.
"""

from __future__ import annotations

import time
from collections import deque
from difflib import SequenceMatcher

PROTECTED = ("stop", "pause", "cancel", "quiet", "wait")

FUZZY_RATIO = 0.8
MIN_RUN = 3
SHORT_HEARD_RUN = 2
SHORT_HEARD_WORDS = 5


def normalise(text: str) -> list[str]:
    """Lower-case words with punctuation stripped; numbers left as spoken digits."""
    words: list[str] = []
    for token in text.split():
        word = token.strip(" \t\n.,!?;:\"'()[]{}—-«»…").casefold()
        if word:
            words.append(word)
    return words


def _fuzzy_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO


class OwnSpeech:
    """Remembers the last few sentences Voxa spoke and strips echoes of them."""

    def __init__(self, keep_seconds: float = 6.0, clock=time.monotonic) -> None:
        self.keep_seconds = keep_seconds
        self.clock = clock
        self.texts: deque[list[str]] = deque(maxlen=3)
        self.speaking = False
        self.finished_at: float | None = None

    def started(self, text: str) -> None:
        words = normalise(text)
        if words:
            self.texts.append(words)
        self.speaking = True
        self.finished_at = None

    def finished(self) -> None:
        self.speaking = False
        self.finished_at = self.clock()

    def active(self) -> bool:
        if not self.texts:
            return False
        if self.speaking:
            return True
        if self.finished_at is None:
            return False
        return (self.clock() - self.finished_at) <= self.keep_seconds

    def _protected(self, protect: list[str] | None) -> set[str]:
        out = set(PROTECTED)
        if protect:
            out.update(normalise(" ".join(protect)))
        return out

    def _removed(self, heard_words: list[str], protect: set[str]) -> list[bool]:
        n = len(heard_words)
        removed = [False] * n
        min_run = MIN_RUN if n >= SHORT_HEARD_WORDS else SHORT_HEARD_RUN
        for words in self.texts:
            m = len(words)
            for i in range(n):
                for p in range(m):
                    if not _fuzzy_equal(heard_words[i], words[p]):
                        continue
                    run = 1
                    while i + run < n and p + run < m and _fuzzy_equal(heard_words[i + run], words[p + run]):
                        run += 1
                    if run >= min_run:
                        for k in range(i, i + run):
                            if heard_words[k] not in protect:
                                removed[k] = True
        return removed

    def strip(self, heard: str, protect: list[str] | None = None) -> tuple[str, float]:
        if not self.active():
            return (heard, 0.0)
        heard_words = normalise(heard)
        if not heard_words:
            return (heard, 0.0)
        protect_set = self._protected(protect)
        removed = self._removed(heard_words, protect_set)
        removed_count = sum(removed)
        if removed_count == 0:
            return (heard, 0.0)
        remaining = [w for w, gone in zip(heard_words, removed, strict=True) if not gone]
        share = removed_count / len(heard_words)
        return (" ".join(remaining), share)

    def is_echo(self, heard: str, threshold: float = 0.6, protect: list[str] | None = None) -> bool:
        if not self.active():
            return False
        heard_words = normalise(heard)
        if not heard_words:
            return False
        protect_set = self._protected(protect)
        removed = self._removed(heard_words, protect_set)
        share = sum(removed) / len(heard_words)
        remaining = sum(1 for gone in removed if not gone)
        return share >= threshold and remaining < 2
