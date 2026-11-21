"""Barge-in off the frame clock (roadmap A3, step 2).

The GTK thread owns two things and nothing else: the frame clock (it knows when a level sample
was taken and how much audio it stands for) and the effects (only it can stop playback, unmute
the microphone or arm a prompt). Everything in between - the onset rule, the evidence windows,
the candidate transcript - runs here, on one worker thread, and comes back as a decision.

Two properties are load-bearing:

*BOUNDED QUEUES.* The frame clock must never wait for the decision, and a worker that falls
behind must not accumulate a backlog that decides something minutes later. The sample queue
holds what the rules need of the recent past (``SAMPLE_QUEUE`` frames); when it is full the
OLDEST sample is dropped, because the newest evidence is the one that says whether to yield
now. Dropped samples are counted, and ``voxa/bench`` reports the count as a transport drop.

*GENERATIONS.* Every sample and every decision carries the generation it belongs to. Yielding
bumps the generation: the samples queued for the reply that was just interrupted are stale, the
decisions already queued for it are stale, and the callbacks the interrupted reply's own
synthesis and prefetch still hold are stale. A stale decision is not applied, not half-applied.

Nothing in this module imports GTK: the caller decides how a decision reaches the frame clock
(``voxa/window.py`` hands it to ``idle()``, the bench drains ``poll()`` synchronously).
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .bargectl import BargeInController, BargeOutcome
from .bargein import FRAME_MS

SAMPLE_QUEUE = 8
DECISION_QUEUE = 8
POLL_SECONDS = 0.25


@dataclass(slots=True)
class HearingSample:
    """One frame's worth of evidence, as the frame clock hands it over.

    The numbers are read on the frame clock and JUDGED here. ``candidate`` and ``evidence`` are
    callables, not values: the transcript of the captured span is produced on the worker thread,
    only when the words stage actually needs it, and never on the frame clock.
    """

    level01: float = 0.0
    speaking: bool = False
    paused: bool = False
    muted: bool = False
    wake_word: str = "voxa"
    media_just_started: bool = False
    mic_db: float | None = None
    reference_db: float | None = None
    delay_ms: float | None = None
    dt_ms: float = FRAME_MS
    candidate: Callable[[], tuple[str, str]] | None = None
    evidence: Callable[[], object] | object | None = None
    generation: int = 0


@dataclass(slots=True)
class ReplyStarted:
    """A reply began being read out: the gate's baseline and the onset evidence restart.

    This is the command that replaced the grace period. The evidence starts over; the microphone
    does not get folded.
    """

    text: str = ""


@dataclass(slots=True)
class ReplyEnded:
    """The reply finished, was stopped or the mode closed."""


@dataclass(slots=True)
class PieceStarted:
    """A new piece of the SAME reply began being read out.

    The evidence restarts (the speaker onset is the loudest moment of every piece) while the
    words being spoken stay: the own-voice check still needs what is being said.
    """


class HearingDirector:
    """The barge-in decision, on its own thread, fed by bounded queues.

    ``start`` and ``stop`` bracket the thread; every other method is safe to call from the
    frame clock while it runs.
    """

    def __init__(
        self,
        *,
        controller: BargeInController | None = None,
        on_decision: Callable[[BargeOutcome], None] | None = None,
        sample_queue: int = SAMPLE_QUEUE,
        decision_queue: int = DECISION_QUEUE,
        trace: Callable[..., None] | None = None,
    ) -> None:
        self.controller = controller if controller is not None else BargeInController()
        self._on_decision = on_decision
        self._trace = trace
        self._events: queue.Queue = queue.Queue(maxsize=max(1, sample_queue))
        self._decisions: deque[BargeOutcome] = deque(maxlen=max(1, decision_queue))
        self._generation = 0
        self._dropped = 0
        self._stale = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain, name="voxa-hearing", daemon=True)
        self._thread.start()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        if self._thread is None:
            return
        if join:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=POLL_SECONDS)
            except queue.Empty:
                continue
            self._judge(event)

    def _judge(self, event: object) -> None:
        if isinstance(event, HearingSample):
            if event.generation != self._generation:
                self._stale += 1
                return
            outcome = self.controller.observe(
                event.level01,
                speaking=event.speaking,
                paused=event.paused,
                muted=event.muted,
                wake_word=event.wake_word,
                media_just_started=event.media_just_started,
                candidate=event.candidate,
                evidence=event.evidence,
                mic_db=event.mic_db,
                reference_db=event.reference_db,
                delay_ms=event.delay_ms,
                dt_ms=event.dt_ms,
            )
            outcome.generation = self._generation
            self._deliver(outcome)
        elif isinstance(event, ReplyStarted):
            self.controller.reply_started(event.text)
        elif isinstance(event, ReplyEnded):
            self.controller.reply_ended()
        elif isinstance(event, PieceStarted):
            self.controller.note_speaking()

    def _deliver(self, outcome: BargeOutcome) -> None:
        self._decisions.append(outcome)
        if self._on_decision is not None:
            self._on_decision(outcome)

    def _push(self, event: object) -> bool:
        """Queue one event, dropping the oldest when the worker has fallen behind."""
        try:
            self._events.put_nowait(event)
            return True
        except queue.Full:
            pass
        try:
            self._events.get_nowait()
        except queue.Empty:
            return False
        self._dropped += 1
        if self._trace is not None:
            try:
                self._trace("hearing-drop", dropped=self._dropped, queued=self._events.qsize())
            except Exception:
                pass
        try:
            self._events.put_nowait(event)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def submit(self, sample: HearingSample) -> bool:
        """Hand one frame's evidence over. Never blocks, whatever the worker is doing."""
        sample.generation = self._generation
        return self._push(sample)

    def reply_started(self, text: str) -> bool:
        return self._push(ReplyStarted(text=text))

    def reply_ended(self) -> bool:
        return self._push(ReplyEnded())

    def piece_started(self) -> bool:
        return self._push(PieceStarted())

    def generation(self) -> int:
        return self._generation

    def accept(self, outcome: BargeOutcome) -> bool:
        """Is this decision still about the reply the app is in?

        The frame clock asks this before it performs any effect. A decision that crossed a
        yield is dropped whole: applying half of it is what made an interruption twice.
        """
        return outcome.generation == self._generation

    def invalidate(self) -> int:
        """The floor was yielded: everything queued for this reply is stale.

        Returns the new generation, which is what the caller stamps on the callbacks its own
        synthesis and prefetch still hold.
        """
        self._generation += 1
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, HearingSample) and event.generation != self._generation:
                self._stale += 1
        self._decisions.clear()
        self.controller.invalidate()
        return self._generation

    def poll(self) -> BargeOutcome | None:
        """Take the oldest decision waiting for the frame clock."""
        return self._decisions.popleft() if self._decisions else None

    @property
    def dropped(self) -> int:
        """Samples thrown away because the worker could not keep up."""
        return self._dropped

    @property
    def stale(self) -> int:
        """Samples and decisions discarded because their generation was invalidated."""
        return self._stale

    @property
    def pending(self) -> int:
        """Events waiting to be judged."""
        return self._events.qsize()
