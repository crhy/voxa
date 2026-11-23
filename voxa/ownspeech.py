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
from dataclasses import dataclass
from difflib import SequenceMatcher

PROTECTED = ("stop", "pause", "cancel", "quiet", "wait")

FUZZY_RATIO = 0.8
MIN_RUN = 3
SHORT_HEARD_RUN = 2
SHORT_HEARD_WORDS = 5

# Roadmap task 7: the ± window a HEARD word may sit from a played TTS word and still be a
# time-matched echo candidate. Report 78-87 (report line 181) names 200/400/600 ms as the sweep;
# 400 ms is the shipped default (Edge TTS word boundaries are estimates, and the audible moment is
# already shifted by task 1's measured output latency, so the slack is synthesis error + jitter).
WORD_WINDOW_MS = 400.0

# The attribution source that allows a rewrite at all.
ECHO = "echo"


@dataclass(slots=True)
class PlayedWord:
    """One TTS word Voxa's speakers played, and the audible span it occupied.

    ``start``/``end`` are already in playback (audible) time: the synthesis offset from Edge TTS'
    ``WordBoundary`` (``voxa/speech.py``) shifted by task 1's measured output latency (the render
    timeline's ``audible_time`` map). ``generation`` is the id the render timeline files this
    playback under, so a stopped reply's already-queued tail is not mistaken for heard echo.
    """

    word: str
    start: float
    end: float
    generation: str | None = None

    @property
    def span_seconds(self) -> float:
        return max(0.0, self.end - self.start)


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
        # Roadmap task 7: the word-timed timeline of what her speakers actually played. Bounded on
        # purpose (the same reasoning as keep_seconds): it is evidence about WHEN an echo of each
        # word would land, never a window during which matching words get deleted.
        self._played: deque[PlayedWord] = deque(maxlen=2048)

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

    # ------------------------------------------------------------------ word-timed timeline
    # Roadmap task 7: the whole point is that she knows WHICH word she was saying at each audible
    # moment, so a heard word that lands inside the span of a word she played is a time-matched
    # echo candidate. This is one weighted FEATURE (report 78-87); it never deletes a word of a
    # confirmed near-end utterance (handoff 5.3 / the A2 rule).

    def note_played(self, word: str, start: float, end: float,
                    generation: str | None = None) -> None:
        """Record one TTS word her speakers played, in audible (playback) time."""
        word = word.strip().casefold()
        if not word:
            return
        start, end = float(start), float(end)
        if end < start:
            start, end = end, start
        self._played.append(PlayedWord(word=word, start=start, end=end, generation=generation))

    def note_played_boundaries(self, boundaries, generation: str | None = None,
                               latency: float = 0.0) -> None:
        """Record Edge-TTS word boundaries shifted by the measured output latency.

        ``boundaries`` is the ``[(word, offset_seconds, duration_seconds)]`` ``voxa/speech.py``'s
        ``on_words`` callback hands over. ``latency`` is task 1's measured output latency (the
        seconds between the synthesis offset and the moment the sample is audible); the render
        timeline already folds it into ``audible_time`` on the live path, so a caller that hands
        audibility-corrected spans passes 0.0.
        """
        shift = float(latency)
        for entry in boundaries:
            word = str(entry[0])
            offset = float(entry[1]) + shift
            duration = float(entry[2]) if len(entry) > 2 else 0.0
            self.note_played(word, offset, offset + max(0.0, duration), generation)

    def played_words(self) -> list[PlayedWord]:
        return list(self._played)

    def item_being_read(self, at_time: float) -> PlayedWord | None:
        """The word her speakers were audible reading at ``at_time`` (roadmap task 7 step 4).

        The same timeline that times the echo also answers "what was she saying when I interrupted?"
        - so an anaphoric "that one" from the user resolves to the item on the tongue at that moment.
        """
        at_time = float(at_time)
        best: PlayedWord | None = None
        best_start = float("inf")
        for played in self._played:
            if played.start <= at_time <= played.end and played.start < best_start:
                best, best_start = played, played.start
        return best

    def timed_overlap(self, heard_with_times, window_ms: float = WORD_WINDOW_MS) -> dict:
        """The WORD-TIMED overlap as a feature: heard words that match a played word inside the time window.

        ``heard_with_times`` is an iterable of ``(word, onset_seconds, offset_seconds)`` for the
        recogniser's words (faster-whisper ``word_timestamps=True``). A heard word counts as
        time-matched when it fuzzy-matches one of her played words AND that word was audible within
        ``window_ms`` of the heard word's span. Returns a diagnostic share; nothing is removed.
        """
        heard = [
            (str(word).strip().casefold(), float(start), float(end))
            for word, start, end in heard_with_times
            if str(word).strip()
        ]
        if not heard or not self._played:
            return {"share": 0.0, "matched": 0, "words": len(heard),
                    "longest_run": 0, "window_ms": float(window_ms)}
        window = float(window_ms) / 1000.0
        matched = 0
        runs: list[bool] = []
        for word, onset, _offset in heard:
            # Time-matched when the word fuzzy-matches one of her played words AND she was audible
            # reading it within +- the window of the moment the recogniser placed this word: the
            # echo of THAT word could only land at that time (report 78-87: the synthesis offset is
            # already converted to playback time, so this is onset-to-onset within +-W ms).
            hit = any(
                _fuzzy_equal(word, played.word) and abs(onset - played.start) <= window
                for played in self._played
            )
            runs.append(hit)
            if hit:
                matched += 1
        longest = run = 0
        for hit in runs:
            run = run + 1 if hit else 0
            longest = max(longest, run)
        return {"share": matched / len(heard), "matched": matched, "words": len(heard),
                "longest_run": longest, "window_ms": float(window_ms)}

    def timed_matches(self, heard_with_times, window_ms: float = WORD_WINDOW_MS) -> list[bool]:
        """Per heard word: does it fuzzy-match one of her played words within ± the window?

        The same onset-to-onset test as :meth:`timed_overlap`, but per word - so a caller can
        drop exactly the matched words and keep everything else (:meth:`timed_removed`).
        """
        heard = [
            (str(word).strip().casefold(), float(start))
            for word, start, _end in heard_with_times
            if str(word).strip()
        ]
        if not heard or not self._played:
            return [False] * len(heard)
        window = float(window_ms) / 1000.0
        return [
            any(_fuzzy_equal(word, played.word) and abs(onset - played.start) <= window
                for played in self._played)
            for word, onset in heard
        ]

    def timed_removed(self, heard_with_times, window_ms: float = WORD_WINDOW_MS,
                      joint_closed_at=None) -> list[bool]:
        """The word-timed DROP rule (roadmap task 7 re-plan, task f175 step 3).

        A heard word is removed ONLY when BOTH hold: it fuzzy-matches a word Voxa was audibly
        reading within ± the window (her echo of THAT word could only land at that moment), AND
        the joint double-talk gate was CLOSED at that moment - it measured echo, nothing else
        (:class:`voxa.voicedetect.DoubleTalkGate`'s echo-consistent mirror verdict, shipped by
        task 5/f173). The gate open or unmeasurable never deletes: an unexplained word is the
        user's. This is why protected command words ("stop", "cancel", ...) are kept unless the
        rule fires around their exact moment: a "Stop" the user says while she reads a different
        word is her own command, not her echo.

        ``joint_closed_at`` maps a heard word's onset (audible time) to the joint gate's state
        there; missing (None) means never closed.
        """
        entries = [(str(word).strip(), float(start))
                   for word, start, _end in heard_with_times if str(word).strip()]
        matches = self.timed_matches(
            [(word, onset, onset) for word, onset in entries], window_ms=window_ms)
        removed: list[bool] = []
        for matched, (_word, onset) in zip(matches, entries, strict=True):
            if not matched:
                removed.append(False)
                continue
            closed = joint_closed_at(float(onset)) if callable(joint_closed_at) else False
            removed.append(bool(closed))
        return removed

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
              confirmed_user: bool = False, heard_with_times=None,
              joint_closed_at=None, window_ms: float = WORD_WINDOW_MS) -> tuple[str, float]:
        """Remove echo words, but only when the evidence says the span is her own voice.

        Without ``heard_with_times`` this is the degraded text-only path, unchanged: the lexical
        run decides (logged as degraded by the caller). With word times (roadmap task 7's
        re-plan) the lexical run stops deleting - it stays a diagnostic in :meth:`features` -
        and ONLY the word-timed rule does: a word is dropped when it time-matches one of her
        played words within ± the window AND the joint gate was closed at that moment
        (:meth:`timed_removed`). A confirmed near-end utterance is never rewritten either way.

        Returns ``(text, share)``: the share is the text-overlap feature, reported whether or
        not anything was removed, so a kept user repetition is still visible in the trace.
        """
        if not self.active():
            return (heard, 0.0)
        heard_words = normalise(heard)
        if not heard_words:
            return (heard, 0.0)
        protect_set = self._protected(protect)
        if heard_with_times is not None:
            entries = [(str(word).strip(), start, end)
                       for word, start, end in heard_with_times if str(word).strip()]
            if len(entries) == len(heard_words):
                removed = self.timed_removed(entries, window_ms=window_ms,
                                             joint_closed_at=joint_closed_at)
                removed_count = sum(removed)
                share = removed_count / len(heard_words)
                if removed_count == 0:
                    return (heard, share)
                if not self._may_rewrite(evidence, confirmed_user):
                    return (heard, share)
                remaining = [w for w, gone in zip(heard_words, removed, strict=True) if not gone]
                return (" ".join(remaining), share)
            # The word times and the transcript disagreed: never delete what cannot be aligned.
            return (heard, 0.0)
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
