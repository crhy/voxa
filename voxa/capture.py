"""Bounded, timestamped capture of the microphone, with pinning (handoff 6.4, milestone A).

The old code kept a five-second list of chunks in ``ConversationController`` (``_barge_buf``)
and DESTROYED it: ``candidate_text()`` joined it and cleared it, and ``mute()``/``unmute()``
cleared it too, so the words that proved an interruption never reached the request and the
user had to repeat them (handoff 5.1, vetted). This module keeps the same audio and makes
reading it non-destructive:

* every captured frame carries a **sequence id** and a timestamp, so loss and duplication
  can be measured instead of guessed (handoff 8.4: "test frame identifiers, not just
  transcript strings");
* both streams are kept: the **enhanced** chunk the ASR stage receives and, when the caller
  has it, the **raw** capture before the echo stages;
* **speech onset pins** the span, starting ``pre_roll_seconds`` (500 ms) before the onset;
  a rolling trim can never overwrite a pinned span;
* ``snapshot()`` reads a copy and never clears; ``promote()`` hands the pinned frames to the
  user turn and marks their ids consumed, so later audio is appended **exactly once**;
* a rejected candidate is released after its evidence window, and a long request keeps
  growing in bounded chunked storage instead of vanishing when the ring rolls over;
* every drop is counted (``dropped_frames``), never silent.

Nothing here runs a model, blocks, or allocates without a bound: ``add`` is called from the
capture path.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

SR = 16000  # the rate the capture path and the bench both use

# Handoff 6.4: "Keep approximately 0.5-1.0 seconds of pre-roll as an initial tuning range."
PRE_ROLL_SECONDS = 0.5
# The un-pinned rolling window: what the old ``_barge_buf`` held (160000 bytes ~ 5 s).
WINDOW_SECONDS = 5.0
# A confirmed user turn keeps growing (bounded chunked utterance storage) instead of
# being cut when the ring rolls over; this is the bound, and going past it is counted.
MAX_PINNED_SECONDS = 120.0
# How long a REJECTED candidate's pin survives before the span may be trimmed again.
EVIDENCE_SECONDS = 2.0


@dataclass(slots=True)
class Frame:
    """One captured moment: its sequence id, when it happened, and its audio."""

    seq: int
    t_s: float
    pcm: bytes
    level: float
    raw: bytes | None = None  # the same moment before the echo stages, when the caller has it


class CaptureRing:
    """Bounded timestamped capture with pinning. Thread-safe: the capture thread appends while
    the UI thread pins, snapshots and promotes.

    ``clock`` is the caller's clock (``time.monotonic`` in the app, the bench's simulated
    clock in the runner), so a run that replays files faster than real time keeps real
    ordering.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        rate: int = SR,
        pre_roll_seconds: float = PRE_ROLL_SECONDS,
        window_seconds: float = WINDOW_SECONDS,
        max_pinned_seconds: float = MAX_PINNED_SECONDS,
        evidence_seconds: float = EVIDENCE_SECONDS,
    ) -> None:
        self.clock = clock
        self.rate = rate
        self.pre_roll_seconds = pre_roll_seconds
        self.window_seconds = window_seconds
        self.max_pinned_seconds = max_pinned_seconds
        self.evidence_seconds = evidence_seconds
        self._frames: deque[Frame] = deque()
        self._bytes = 0
        self._last_seq = -1
        # The pinned span: ``_pin_from`` is the sequence id of its first frame (pre-roll included),
        # ``_pin_onset`` the id of the onset frame itself. None = nothing pinned, plain rolling window.
        self._pin_from: int | None = None
        self._pin_onset: int | None = None
        # Sequence ids already handed to a user turn: they must never be handed over twice.
        self._consumed: set[int] = set()
        # When a rejected candidate's pin may be dropped (0.0 = no release scheduled).
        self._release_at = 0.0
        # Counted drops: bounded memory is a rule, a dropped frame is a fact to report.
        self.dropped_frames = 0
        self.dropped_bytes = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ capture

    def add(
        self,
        pcm: bytes,
        level: float,
        *,
        seq: int | None = None,
        t_s: float | None = None,
        raw: bytes | None = None,
    ) -> Frame:
        """Append one captured frame, giving it a sequence id if the caller did not.

        Re-appending an id the ring already holds returns the stored frame unchanged: the
        same frame never exists twice, which is what makes "appended exactly once" testable.
        """
        with self._lock:
            if seq is not None and seq <= self._last_seq:
                for frame in self._frames:
                    if frame.seq == seq:
                        return frame
            elif seq is None:
                seq = self._last_seq + 1
            frame = Frame(seq=seq, t_s=(self.clock() if t_s is None else t_s),
                          pcm=pcm, level=level, raw=raw)
            self._frames.append(frame)
            self._last_seq = max(self._last_seq, seq)
            self._bytes += len(pcm) + (len(raw) if raw is not None else 0)
            self._expire_release()
            self._trim()
            return frame

    def was_consumed(self, seq: int) -> bool:
        """True when that frame has already been handed to a user turn (never queue it twice)."""
        return seq in self._consumed

    # ------------------------------------------------------------------ pinning

    def pin(
        self,
        seq: int | None = None,
        *,
        level_floor: float | None = None,
        pre_roll_seconds: float | None = None,
    ) -> bool:
        """Pin the span of a speech onset: the onset frame, back through the elevated frames
        that belong to the same run of speech, plus the pre-roll before it.

        ``seq`` is the onset frame when the caller knows it (the bench's frame id); otherwise
        the newest frame is the onset. ``level_floor`` is the level below which a frame is
        silence/noise rather than speech, in the same units the caller feeds levels in: walking
        back over elevated frames recovers where the user actually started, which is what keeps
        the first word of an interruption that began before the gate had time to open. Pass
        ``None`` for no walk-back: the span is exactly onset minus pre-roll. The walk-back can
        never reach past the retained window, so a pinned span stays bounded.
        """
        pre_roll = self.pre_roll_seconds if pre_roll_seconds is None else pre_roll_seconds
        with self._lock:
            if not self._frames:
                return False
            onset = self._frames[-1] if seq is None else self._frame_at(seq)
            if onset is None:
                # The onset frame itself is already gone (the span was pinned late): pin what
                # is left rather than nothing, so the rest of the utterance survives.
                onset = self._frames[0]
            # Walk back over the run of speech the onset belongs to.
            index = self._index_of(onset.seq)
            while level_floor is not None and index > 0:
                previous = self._frames[index - 1]
                if previous.level < level_floor:
                    break
                index -= 1
            start_t = self._frames[index].t_s - pre_roll
            while index > 0 and self._frames[index - 1].t_s >= start_t:
                index -= 1
            pinned_from = self._frames[index].seq
            if self._pin_from is None or pinned_from < self._pin_from:
                self._pin_from = pinned_from
                self._pin_onset = onset.seq
            self._release_at = 0.0
            self._trim()
            return True

    def pinned(self) -> list[Frame]:
        """The frames of the pinned span, oldest first (empty when nothing is pinned)."""
        with self._lock:
            if self._pin_from is None:
                return []
            return [frame for frame in self._frames if frame.seq >= self._pin_from]

    def snapshot(self, *, raw: bool = False) -> bytes:
        """The audio of the active span as ONE copy: the pinned span's frames that have not
        been handed to a user turn yet, otherwise the whole retained window. Never clears, never
        consumes, never raises.

        The span is what a classifier may look at; reading it leaves the audio in place so the
        same frames can become the request later (handoff 6.4: "a classifier reads a snapshot;
        it never consumes the only copy"). Frames already promoted are skipped, so the NEXT
        candidate judges only the audio that arrived since - the accepted words are not judged,
        transcribed or commanded twice.
        """
        with self._lock:
            frames = list(self._frames)
            start = self._pin_from
            pick = [(frame.raw if raw and frame.raw is not None else frame.pcm)
                    for frame in frames
                    if (start is None or frame.seq >= start) and frame.seq not in self._consumed]
            return b"".join(pick)

    def promote(self) -> list[Frame]:
        """Hand the pinned span to the user turn, exactly once.

        The frames keep their sequence ids, and their ids are recorded as consumed, so the
        caller that later receives the same frames again (or the next frames of the same
        turn) can append only what is new. The pin stays open: a request longer than the
        window keeps growing in this same span instead of losing its start.
        """
        with self._lock:
            if self._pin_from is None:
                return []
            fresh = [frame for frame in self._frames if frame.seq >= self._pin_from
                     and frame.seq not in self._consumed]
            self._consumed.update(frame.seq for frame in fresh)
            return fresh

    def note_rejected(self, *, now: float | None = None) -> None:
        """A candidate that was not the user: keep its audio for the evidence window, then release.

        Rejection is a decision about what to DO, never a licence to delete the user's speech.
        """
        self._release_at = (self.clock() if now is None else now) + self.evidence_seconds

    def release(self) -> None:
        """Drop the pin (the frames stay in the rolling window until a trim takes them)."""
        with self._lock:
            self._pin_from = None
            self._pin_onset = None
            self._release_at = 0.0

    # ------------------------------------------------------------------ bookkeeping

    def _frame_at(self, seq: int) -> Frame | None:
        for frame in self._frames:
            if frame.seq == seq:
                return frame
        return None

    def _index_of(self, seq: int) -> int:
        for index, frame in enumerate(self._frames):
            if frame.seq == seq:
                return index
        return len(self._frames) - 1

    def _expire_release(self) -> None:
        if self._release_at and self.clock() >= self._release_at:
            self._pin_from = None
            self._pin_onset = None
            self._release_at = 0.0

    def _trim(self) -> None:
        """Trim only what is neither pinned nor consumed, and count everything dropped."""
        while self._frames:
            newest = self._frames[-1].t_s
            oldest = self._frames[0]
            pinned = self._pin_from is not None and oldest.seq >= self._pin_from
            if pinned:
                # A pinned span is bounded too: a long request keeps growing, but only this far.
                if newest - oldest.t_s > self.max_pinned_seconds:
                    self._drop_oldest()
                    self._pin_from = self._frames[0].seq if self._frames else None
                    continue
                break
            if newest - oldest.t_s <= self.window_seconds:
                break
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        frame = self._frames.popleft()
        self.dropped_frames += 1
        self.dropped_bytes += len(frame.pcm) + (len(frame.raw) if frame.raw is not None else 0)
        self._bytes -= len(frame.pcm) + (len(frame.raw) if frame.raw is not None else 0)
        self._consumed.discard(frame.seq)

    # ------------------------------------------------------------------ reporting

    @property
    def frames_retained(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def span_seconds(self) -> float:
        """Seconds of audio the active span covers (the pinned span, or the whole window)."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return 0.0
        start = frames[0]
        if self._pin_from is not None:
            start = next((frame for frame in frames if frame.seq >= self._pin_from), frames[0])
        return max(0.0, frames[-1].t_s - start.t_s)

    @property
    def pin_from(self) -> int | None:
        with self._lock:
            return self._pin_from

    @property
    def consumed(self) -> set[int]:
        with self._lock:
            return set(self._consumed)

    def span_bounds(self) -> tuple[int, int]:
        """(first, last) sequence id of the span ``snapshot()`` would return."""
        with self._lock:
            frames = [frame for frame in self._frames
                      if (self._pin_from is None or frame.seq >= self._pin_from)
                      and frame.seq not in self._consumed]
            if not frames:
                return 0, -1
            return frames[0].seq, frames[-1].seq

    def span_start_time(self) -> float | None:
        """The arrival time of the oldest frame ``snapshot()`` would return (None: no span).

        Roadmap task 7 (f175): the recogniser's word times are segment-relative; a caller that
        wants them on the same clock as the word-timed playback timeline needs this anchor.
        """
        with self._lock:
            frames = [frame for frame in self._frames
                      if (self._pin_from is None or frame.seq >= self._pin_from)
                      and frame.seq not in self._consumed]
            if not frames:
                return None
            return float(frames[0].t_s)

