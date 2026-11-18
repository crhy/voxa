"""The barge-in decision, free of GTK and of the window (roadmap B2, step 1).

``voxa/window.py`` and the hearing bench (``voxa/bench/runner.py``) call this one class,
so what the bench measures is exactly what the app decides. The logic is the code that
used to live in ``MainWindow._maybe_barge_in`` (window.py:3152); behaviour is unchanged
by this move — the window keeps the effects (stopping speech, unmuting, arming the prompt)
and this module keeps the decision.

Everything here is pure: a clock the caller supplies, a level the caller measured, and a
callable that returns the words heard. No GTK, no threads, no audio devices.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .bargein import BargeInGate, is_own_voice
from .conversation import detect_addressed_exit_phrase, strip_wake_word

# Barge-in: loud sustained speech while a reply is being read interrupts it.
# A short grace period ignores the TTS itself starting (speaker onset is the loudest
# moment); the level gate in .bargein decides when the mic is loud enough, and the words
# heard decide whether it was really the user.
BARGE_IN_GRACE_SECONDS = 0.4
# After the level gate opens, how long to keep listening before judging the words.
BARGE_IN_LISTEN_SECONDS = 0.9

# Outcomes of one observation; the caller performs the effects.
IGNORED = "ignored"
OFFLINE = "offline"
OWN_VOICE = "own-voice"
INTERRUPT = "interrupt"

# The presenter of a station Voxa has just started is not the user, unless the words
# are an interrupt word or address her by name.
_MEDIA_EXCEPTION = re.compile(r"\b(?:stop|pause|cancel|quiet|enough|wait)\b")

# The capture path reports levels in int16 units; the gate works on the 0..1 scale the
# window builds by dividing by this (voxa/window.py: ``self._latest_level / 4000.0``).
LEVEL_SCALE = 4000.0


@dataclass(slots=True)
class BargeOutcome:
    """One barge-in observation: what to do, and the evidence that decided it."""

    action: str
    reason: str = ""
    heard: str = ""
    raw: str = ""
    level: float = 0.0
    gate_opened: bool = False
    listening_seconds: float = 0.0
    # True on the observation where the level gate first opened: independent speech onset.
    # The caller pins the captured span from here (voxa/conversation.py:note_onset), which is
    # what keeps the beginning of an interruption alive while the words are still being judged.
    onset: bool = False


class BargeInController:
    """State of the barge-in decision: the level gate, the grace and listening windows,
    the text Voxa is speaking (for the own-voice check) and the trace bookkeeping.

    ``observe`` is called once per level sample (the window does it every 50 ms, the bench
    once per 20 ms frame) and returns what should happen; the caller carries out the
    effects, because only the caller can stop speech or arm a prompt.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        grace_seconds: float = BARGE_IN_GRACE_SECONDS,
        listen_seconds: float = BARGE_IN_LISTEN_SECONDS,
        gate: BargeInGate | None = None,
        trace: Callable[..., None] | None = None,
    ) -> None:
        self.clock = clock
        self.grace_seconds = grace_seconds
        self.listen_seconds = listen_seconds
        self.gate = gate if gate is not None else BargeInGate()
        self.trace = trace
        # When the current reply started being read out (0.0 = not speaking).
        self.speaking_since = 0.0
        # When the level gate opened and the words are being collected (0.0 = not pending).
        self.pending_since = 0.0
        # Per-reply counters, traced to barge.log so a failed interruption can be explained.
        self.stats: dict = {}
        # What Voxa is saying now and the reply before it: the own-voice check's evidence.
        self.speaking_text = ""
        self.recent_spoken = ""

    def onset_level_floor(self) -> float:
        """The microphone level that separates speech from silence for THIS reply, in the
        caller's level units (the window and the bench divide the capture level by LEVEL_SCALE
        before feeding the gate, so the floor comes back scaled the same way).

        Used to walk the capture ring back over the run of speech that an onset belongs to,
        which is what recovers the first word of an interruption that started before the gate
        had time to open.
        """
        return max(self.gate.min_level, self.gate.ratio * self.gate.baseline) * LEVEL_SCALE

    def _trace(self, event: str, **fields) -> None:
        if self.trace is None:
            return
        try:
            self.trace(event, **fields)
        except Exception:  # noqa: BLE001 - tracing must never disturb the conversation
            pass

    def note_speaking(self) -> None:
        """Speech started being read out: the grace period restarts, the gate is cleared."""
        self.speaking_since = self.clock()
        self.gate.reset()

    def reply_started(self, text: str) -> None:
        """A reply began being read out (window.py:3004-3010 did exactly this before)."""
        self.recent_spoken = self.speaking_text
        self.speaking_text = text
        self.note_speaking()

    def reply_ended(self) -> None:
        """The reply finished, was stopped or the mode closed (window.py:3060-3061 and friends)."""
        self.speaking_since = 0.0
        self.gate.reset()

    def observe(
        self,
        level01: float,
        *,
        speaking: bool,
        paused: bool = False,
        muted: bool = False,
        wake_word: str = "voxa",
        media_just_started: bool = False,
        candidate: Callable[[], tuple[str, str]] | None = None,
    ) -> BargeOutcome:
        """Judge one level sample while a reply may be playing.

        ``level01`` is the microphone level on the 0..1 scale the gate uses (the window
        divides ``_latest_level`` by 4000). ``candidate`` returns ``(heard, raw)``: the
        transcript of the audio captured while muted, after and before the own-speech
        filter. ``speaking``, ``paused`` and ``muted`` are the caller's state, read the
        way ``_maybe_barge_in`` read them.
        """
        now = self.clock()
        stats = self.stats
        onset_open = False
        if speaking != stats.get("speaking", False):
            if not speaking:
                self._trace("reply-end", **{key: value for key, value in stats.items() if key != "speaking"})
            stats.clear()
            stats["speaking"] = speaking
        if speaking:
            stats["calls"] = stats.get("calls", 0) + 1
        if paused:
            if speaking:
                stats["paused_ticks"] = stats.get("paused_ticks", 0) + 1
            self.gate.reset()
            self.pending_since = 0.0
            return BargeOutcome(IGNORED, reason="paused", level=level01)
        if not muted:
            if speaking:
                stats["not_muted_ticks"] = stats.get("not_muted_ticks", 0) + 1
            self.gate.reset()
            self.pending_since = 0.0
            return BargeOutcome(IGNORED, reason="not-muted", level=level01)
        if not speaking:
            return BargeOutcome(IGNORED, reason="not-speaking", level=level01)
        if now - self.speaking_since < self.grace_seconds:
            stats["grace_ticks"] = stats.get("grace_ticks", 0) + 1
            self.gate.reset()
            self.pending_since = 0.0
            return BargeOutcome(IGNORED, reason="grace", level=level01)
        stats["ticks"] = stats.get("ticks", 0) + 1
        stats["max_level"] = max(stats.get("max_level", 0.0), level01)
        stats["baseline"] = float(self.gate.baseline)
        opened = self.gate.update(level01, self.gate.baseline)
        if self.pending_since <= 0.0:
            if not opened:
                # Still her voice (or noise): teach the gate what "normal" is.
                self.gate.note_speaking_level(level01)
                return BargeOutcome(IGNORED, reason="gate-closed", level=level01)
            self.pending_since = now
            stats["gate_opened"] = stats.get("gate_opened", 0) + 1
            self._trace("gate-open", level=level01, baseline=float(self.gate.baseline),
                        settled=self.gate.settled)
            onset_open = True
        # Always judge the words. A loudness shortcut was tried and interrupted her on her
        # own loudest syllables; the user is given a word or two first.
        if now - self.pending_since < self.listen_seconds:
            return BargeOutcome(IGNORED, reason="listening", level=level01, gate_opened=True,
                                onset=onset_open)
        started = self.clock()
        heard, raw = (candidate() if candidate is not None else ("", ""))
        # A spoken OFFLINE command is a kill switch even over Voxa's reply, and short
        # phrases such as "go offline" otherwise read as her own audio.
        if detect_addressed_exit_phrase(heard, wake_word) == "goodbye":
            self._trace("offline-interrupt", heard=heard, raw=raw)
            self.gate.reset()
            self.pending_since = 0.0
            return BargeOutcome(OFFLINE, heard=heard, raw=raw, level=level01)
        own = is_own_voice(heard, self.speaking_text, self.recent_spoken, raw=raw)
        self._trace("words", heard=heard, raw=raw, own_voice=own,
                    seconds=self.clock() - started, speaking=self.speaking_text[:60])
        if (
            not own
            and media_just_started
            and not _MEDIA_EXCEPTION.search(f"{heard} {raw}".casefold())
            and strip_wake_word(f"{heard} {raw}", wake_word) is None
        ):
            # The station she is announcing has already started: its presenter is not the user.
            own = True
        if own:
            # The loud sound was Voxa's own voice: carry on speaking.
            self.gate.reset()
            self.pending_since = 0.0
            return BargeOutcome(OWN_VOICE, heard=heard, raw=raw, level=level01)
        self.gate.reset()
        self.pending_since = 0.0
        self.speaking_since = 0.0
        return BargeOutcome(INTERRUPT, heard=heard, raw=raw, level=level01,
                            listening_seconds=self.clock() - started)
