"""Echo filter for Voxa's own voice (issue #88), as a FEATURE rather than a deletion rule.

Even with a perfect canceller some of her own voice leaks into the microphone, and she always
knows the sentence she is saying - so a transcript that is (part of) that sentence tells her
what an echo of THIS moment would look like. Handoff 5.3 (vetted) is explicit about what this
must not do: a user may repeat her words on purpose, and a fixed six-second lexical exclusion
window is not a room echo model. So:

* the remembered sentences and the match are kept as a diagnostic (:meth:`features`);
* a rewrite happens only when the AUDIO evidence for that span says echo
  (:meth:`strip` / :meth:`is_echo` take ``evidence``);
* a span already CONFIRMED as the user's (``confirmed_user=True``: it was attributed to a
  second source, or it continues a turn that was) is never rewritten, whatever the words look
  like;
* ``keep_seconds`` is how long her sentences stay remembered as evidence - bounded memory,
  not a window during which matching words get deleted.

Pure text work: no audio, no GTK.
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

# The attribution source that allows a rewrite at all.
ECHO = "echo"


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
    """Remembers the last few sentences Voxa spoke as evidence about what her echo looks like."""

    def __init__(self, keep_seconds: float = 6.0, clock=time.monotonic) -> None:
        # Evidence retention, bounded on purpose: how long a sentence stays in memory, NOT a
        # period during which matching words are deleted (handoff 5.3).
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
        """True while her sentences are still usable as evidence (speaking, or recently spoken)."""
        if not self.texts:
            return False
        if self.speaking:
            return True
        if self.finished_at is None:
            return False
        return (self.clock() - self.finished_at) <= self.keep_seconds

    def age_seconds(self) -> float | None:
        """How long since she finished speaking (None while she is still speaking)."""
        if self.speaking or self.finished_at is None:
            return None
        return round(self.clock() - self.finished_at, 3)

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

    def features(self, heard: str, protect: list[str] | None = None) -> dict:
        """The text overlap as a DIAGNOSTIC: what an echo of her recent words would look like.

        Nothing is removed here. The share, the longest matched run and the age of the evidence
        go to the trace and to the bench; they are inputs to a decision, never the decision.
        """
        heard_words = normalise(heard)
        if not heard_words or not self.texts:
            return {"share": 0.0, "longest_run": 0, "matched": 0, "words": len(heard_words),
                    "active": self.active(), "age_s": self.age_seconds()}
        protect_set = self._protected(protect)
        removed = self._removed(heard_words, protect_set)
        longest = run = 0
        for gone in removed:
            run = run + 1 if gone else 0
            longest = max(longest, run)
        return {"share": sum(removed) / len(heard_words), "longest_run": longest,
                "matched": sum(removed), "words": len(heard_words),
                "active": self.active(), "age_s": self.age_seconds()}

    def _may_rewrite(self, evidence, confirmed_user: bool) -> bool:
        """Only audio evidence that says ECHO may rewrite a transcript.

        ``evidence`` is a :class:`voxa.attribution.SourceDecision` (or its ``source`` string), or
        ``None`` when the capture path has no reference to compare the microphone with - the
        degraded text-only case, kept working so a machine without a reference sink still filters
        its echo, and logged as degraded.
        """
        if confirmed_user:
            # A confirmed near-end utterance is never rewritten by a text match (handoff 5.3).
            return False
        if evidence is None:
            return True
        source = getattr(evidence, "source", evidence)
        return source == ECHO

    def strip(self, heard: str, protect: list[str] | None = None, *, evidence=None,
              confirmed_user: bool = False) -> tuple[str, float]:
        """Remove echo words, but only when the evidence says the span is her own voice.

        Returns ``(text, share)``: the share is the text-overlap feature, reported whether or
        not anything was removed, so a kept user repetition is still visible in the trace.
        """
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
        share = removed_count / len(heard_words)
        if not self._may_rewrite(evidence, confirmed_user):
            # The words match her sentence and the audio says they are NOT her echo: keep them.
            return (heard, share)
        remaining = [w for w, gone in zip(heard_words, removed, strict=True) if not gone]
        return (" ".join(remaining), share)

    def is_echo(self, heard: str, threshold: float = 0.6, protect: list[str] | None = None,
                *, evidence=None, confirmed_user: bool = False) -> bool:
        """True only when BOTH the words and the audio say the whole utterance is her own."""
        if not self.active() or confirmed_user:
            return False
        heard_words = normalise(heard)
        if not heard_words:
            return False
        protect_set = self._protected(protect)
        removed = self._removed(heard_words, protect_set)
        share = sum(removed) / len(heard_words)
        left = [word for word, gone in zip(heard_words, removed, strict=False) if not gone]
        if any(word in protect_set for word in left):
            # "…her own sentence… stop": the one word left is the user interrupting, not echo.
            return False
        if not self._may_rewrite(evidence, confirmed_user):
            return False
        return share >= threshold and len(left) < 2
