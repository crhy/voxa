"""The barge-in decision, free of GTK and of the window (roadmap B2, step 1).

``voxa/window.py`` and the hearing bench (``voxa/bench/runner.py``) call this one class,
so what the bench measures is exactly what the app decides. The logic is the code that
used to live in ``MainWindow._maybe_barge_in`` (window.py:3152); behaviour is unchanged
by this move — the window keeps the effects (stopping speech, unmuting, arming the prompt)
and this module keeps the decision.

Everything here is pure: a clock the caller supplies, a level the caller measured, and a
callable that returns the words heard. No GTK, no threads, no audio devices. In the live app
``observe`` runs on the hearing worker's thread (``voxa/hearing.py``), which is what keeps the
candidate ASR off the GTK thread; the window only receives the decisions.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .attribution import ECHO, SourceEvidence
from .attribution import UNCERTAIN as SOURCE_UNCERTAIN
from .bargein import FRAME_MS, BargeInGate, source_decision
from .onset import OnsetDetector
from .conversation import detect_addressed_exit_phrase, strip_wake_word

# Barge-in: loud sustained speech while a reply is being read interrupts it.
# There is NO grace period any more (handoff 5.4, task A3): ``BARGE_IN_GRACE_SECONDS`` was a
# blind fold over the microphone for the first 0.4 s of every reply, which is exactly where an
# interruption that starts together with her speech lives. What replaces it is the evidence rule
# in ``voxa/onset.py``: the microphone holding sound the speaker path cannot explain, measured
# in milliseconds of audio, live from the first rendered frame.
# After the onset, how much audio to keep collecting before the WORDS are judged. This no longer
# blocks yielding: the decision to stop her stream is acoustic and fast (task A3); the transcript
# completes later and decides what happens to her words, not whether the floor was yielded.
BARGE_IN_LISTEN_MS = 900.0
BARGE_IN_LISTEN_SECONDS = BARGE_IN_LISTEN_MS / 1000.0  # kept for callers that still speak in seconds

# Outcomes of one observation; the caller performs the effects.
IGNORED = "ignored"
OFFLINE = "offline"
OWN_VOICE = "own-voice"
INTERRUPT = "interrupt"
HOLD = "hold"  # the evidence did not decide: keep the audio, strip nothing, run no command
# The fast, acoustic one: stop Voxa's playback NOW. The words are still being collected, and the
# later INTERRUPT / OWN_VOICE / HOLD outcome decides what happens to them (promoted, rejected,
# kept) - never whether the floor was yielded.
YIELD = "yield"

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
    # Which source the evidence supported, how strongly, and whether the path was degraded
    # (traced to barge.log and recorded by the bench; the reason a span was kept or dropped).
    source: str = ""
    confidence: float = 0.0
    degraded: bool = False
    # True on the observation where speech onset is confirmed: independent speech onset.
    # The caller pins the captured span from here (voxa/conversation.py:note_onset), which is
    # what keeps the beginning of an interruption alive while the words are still being judged.
    onset: bool = False
    # Where the run of unexplained sound started, how much of it there is (milliseconds of audio,
    # handoff 5.4), and how far the microphone stood over the speaker path.
    onset_audio_ms: float = 0.0
    evidence_ms: float = 0.0
    margin_db: float = 0.0
    # Stamped by ``voxa/hearing.py`` with the generation this decision belongs to: a decision
    # that crossed a yield is not applied at all (see ``HearingDirector.accept``).
    generation: int = 0


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
        listen_ms: float = BARGE_IN_LISTEN_MS,
        gate: BargeInGate | None = None,
        onset: OnsetDetector | None = None,
        trace: Callable[..., None] | None = None,
    ) -> None:
        self.clock = clock
        self.listen_ms = listen_ms
        self.gate = gate if gate is not None else BargeInGate()
        # Speech onset, in milliseconds of audio (voxa/onset.py). Live from the first frame of
        # every reply: no grace period hides the beginning of her speech any more.
        self.onset = onset if onset is not None else OnsetDetector()
        self.trace = trace
        # When the current reply started being read out (0.0 = not speaking).
        self.speaking_since = 0.0
        # When the candidate started being collected (0.0 = not pending), and how much AUDIO
        # has been collected since - the listening window is counted in milliseconds of sound,
        # not in callbacks.
        self.pending_since = 0.0
        self.pending_audio_ms = 0.0
        # How far the microphone stood above her own level at the moment the gate opened. The gate
        # watches 160 ms of samples and the words are judged about a second later, so the level of
        # the judging frame is not the level of the sound that made us listen.
        self.pending_ratio: float | None = None
        # Per-reply counters, traced to barge.log so a failed interruption can be explained.
        self.stats: dict = {}
        # What Voxa is saying now and the reply before it: the own-voice check's evidence.
        self.speaking_text = ""
        self.recent_spoken = ""
        # Has the floor already been yielded for this reply? Yielding happens once: a second
        # run of unexplained sound inside the same reply is the same interruption, not a new one.
        self._yielded = False

    def _clear_candidate(self) -> None:
        """Drop the pending candidate: the gate's streak and the level ratio it opened on.

        The ratio belongs to the sound that made us listen; once that sound is over, a new one
        has to earn it again.
        """
        self.pending_since = 0.0
        self.pending_ratio = None
        self.pending_audio_ms = 0.0

    def onset_level_floor(self) -> float:
        """The microphone level that separates speech from silence for THIS reply, in the
        caller's level units (the window and the bench divide the capture level by LEVEL_SCALE
        before feeding the gate, so the floor comes back scaled the same way).

        Used to walk the capture ring back over the run of speech that an onset belongs to,
        which is what recovers the first word of an interruption that started before the gate
        had time to open.
        """
        return max(self.gate.min_level, self.gate.ratio * self.gate.baseline) * LEVEL_SCALE

    def _open_candidate(self, now: float, level01: float, stats: dict) -> None:
        """Start collecting the candidate: the sound that made us listen is remembered."""
        self.pending_since = now
        self.pending_audio_ms = 0.0
        stats["gate_opened"] = stats.get("gate_opened", 0) + 1
        self.pending_ratio = self.gate.peak / max(self.gate.baseline, self.gate.min_level)
        self._trace("gate-open", level=level01, baseline=float(self.gate.baseline),
                    settled=self.gate.settled, ratio=round(self.pending_ratio, 2))

    def _trace(self, event: str, **fields) -> None:
        if self.trace is None:
            return
        try:
            self.trace(event, **fields)
        except Exception:  # noqa: BLE001 - tracing must never disturb the conversation
            pass

    def note_speaking(self) -> None:
        """Speech started being read out: the gate's settling window and the onset evidence restart.

        Clearing the onset detector is what makes it live from the FIRST frame of this reply
        instead of carrying a stale run of sound over from the previous one; clearing the gate's
        settling is what keeps her speaker's own onset from teaching the level gate that her
        voice is loud.
        """
        self.speaking_since = self.clock()
        self.gate.note_reply_start()
        self.onset.reset()
        self._yielded = False

    def reply_started(self, text: str) -> None:
        """A reply began being read out (window.py:3004-3010 did exactly this before)."""
        self.recent_spoken = self.speaking_text
        self.speaking_text = text
        self.note_speaking()

    def reply_ended(self) -> None:
        """The reply finished, was stopped or the mode closed (window.py:3060-3061 and friends)."""
        self.speaking_since = 0.0
        self.gate.reset()
        self.onset.reset()
        self._yielded = False

    def invalidate(self) -> None:
        """The floor was yielded: the evidence starts over, the reply's words do not.

        The gate's baseline and the onset detector's run of unexplained sound are cleared, and
        the once-per-reply yield flag is released for the NEXT reply. ``speaking_text`` and
        ``recent_spoken`` stay: the words she was saying are still the evidence the own-voice
        check is judged against, and a rejected interruption has to be rejected against them.
        """
        self.gate.reset()
        self.onset.reset()
        self._yielded = False

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
        evidence: Callable[[], SourceEvidence] | SourceEvidence | None = None,
        mic_db: float | None = None,
        reference_db: float | None = None,
        delay_ms: float | None = None,
        dt_ms: float = FRAME_MS,
        qualified: bool = True,
    ) -> BargeOutcome:
        """Judge one level sample while a reply may be playing.

        ``level01`` is the microphone level on the 0..1 scale the gate uses (the window
        divides ``_latest_level`` by 4000). ``candidate`` returns ``(heard, raw)``: the
        transcript of the audio captured while muted, after and before the own-speech
        filter. ``speaking``, ``paused`` and ``muted`` are the caller's state, read the
        way ``_maybe_barge_in`` read them. ``evidence`` is the capture path's
        :class:`voxa.attribution.SourceEvidence` for this moment (a callable is evaluated
        only once the words are actually judged); ``None`` means the caller has no reference
        to compare the microphone with, which puts the decision in its degraded text-only
        form.

        ``mic_db`` and ``reference_db`` are the microphone level and the reference level of this
        same moment, BOTH in the same dB scale (see ``voxa/onset.py``: an int16 RMS has to be
        brought to dBFS with ``INT16_DB_OFFSET`` before it can be compared with a float reference),
        and they are the inputs of the onset rule - the reason onset detection needs no grace period
        is that the speaker path says what it could have put in the microphone, and sound above
        that ceiling is not her echo. ``delay_ms`` is that path's echo travel time, or ``None``
        when it was never estimated, which turns the onset claim off rather than on. ``dt_ms`` is
        how much audio this sample stands for; every window in this controller is counted in those
        milliseconds.

        ``qualified`` comes from the capture path's transport health: ``False`` means the echo
        canceller failed and the microphone is RAW, so this audio may be Voxa's own voice with its
        echo in it. While a reply is playing no automatic command, duck or playback stop is taken
        from unqualified audio (handoff 6.7); the audio itself is kept and the span stays pinned,
        because a broken filter is never a licence to discard what the microphone heard.
        """
        now = self.clock()
        dt = max(0.0, float(dt_ms))
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
            self._clear_candidate()
            return BargeOutcome(IGNORED, reason="paused", level=level01)
        if not muted:
            if speaking:
                stats["not_muted_ticks"] = stats.get("not_muted_ticks", 0) + 1
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(IGNORED, reason="not-muted", level=level01)
        if not speaking:
            return BargeOutcome(IGNORED, reason="not-speaking", level=level01)
        if self.pending_since > 0.0:
            # The listening window is a duration of AUDIO, not a number of callbacks.
            self.pending_audio_ms += dt
        stats["ticks"] = stats.get("ticks", 0) + 1
        stats["max_level"] = max(stats.get("max_level", 0.0), level01)
        stats["baseline"] = float(self.gate.baseline)
        self.gate.note_reply_audio(dt)
        opened = self.gate.update(level01, self.gate.baseline, dt)
        if self.gate.listening and self.pending_since <= 0.0 and not opened:
            # Still her voice (or noise): teach the gate what "normal" is.
            self.gate.note_speaking_level(level01, dt)
        # --- speech onset: milliseconds of audio the speaker path cannot explain -----------------
        self.onset.note_delay(delay_ms)
        onset = self.onset.observe(mic_db, reference_db, dt)
        if not qualified:
            # The echo path failed while this reply plays: the microphone is raw, so nothing here
            # is qualified enough to stop her stream or to become a request. HOLD keeps the audio
            # and the pin (the user's words are still owed to the turn) and runs no command; the
            # degraded state itself is the caller's to show and to recover.
            self._trace("unqualified", level=round(level01, 4), onset=bool(onset.onset),
                        onset_audio_ms=round(onset.onset_audio_ms, 1))
            return BargeOutcome(HOLD, reason="unqualified-audio", level=level01,
                                onset=bool(onset.onset), gate_opened=bool(onset.onset))
        if onset.onset and not self._yielded:
            # The fast half of the decision (task A3): stop her playback NOW, before a single
            # word has been transcribed. The words are still being collected, and the outcome
            # that decides what happens to them comes later.
            self._yielded = True
            if self.pending_since <= 0.0:
                self._open_candidate(now, level01, stats)
            stats["onset"] = stats.get("onset", 0) + 1
            self._trace("onset", evidence_ms=round(onset.evidence_ms, 1),
                        margin_db=round(onset.margin_db, 2),
                        onset_audio_ms=round(onset.onset_audio_ms, 1),
                        level=round(level01, 4), speaking=self.speaking_text[:60])
            return BargeOutcome(YIELD, reason="onset", level=level01, onset=True, gate_opened=True,
                                onset_audio_ms=onset.onset_audio_ms,
                                evidence_ms=onset.evidence_ms, margin_db=onset.margin_db)
        if self.pending_since <= 0.0:
            if not opened:
                return BargeOutcome(IGNORED, reason="gate-closed", level=level01)
            self._open_candidate(now, level01, stats)
            onset_open = True
        # Always judge the words as well, on the path where no reference exists to compare the
        # microphone with (echo off, no reference sink). A loudness shortcut was tried and
        # interrupted her on her own loudest syllables; the user is given a word or two first.
        if self.pending_audio_ms < self.listen_ms:
            return BargeOutcome(IGNORED, reason="listening", level=level01, gate_opened=True,
                                onset=onset_open)
        started = self.clock()
        heard, raw = (candidate() if candidate is not None else ("", ""))
        # A spoken OFFLINE command is a kill switch even over Voxa's reply, and short
        # phrases such as "go offline" otherwise read as her own audio.
        if detect_addressed_exit_phrase(heard, wake_word) == "goodbye":
            self._trace("offline-interrupt", heard=heard, raw=raw)
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(OFFLINE, heard=heard, raw=raw, level=level01)
        audio = evidence() if callable(evidence) else evidence
        decision = source_decision(heard, self.speaking_text, self.recent_spoken, raw=raw,
                                   evidence=audio,
                                   level_ratio=(self.pending_ratio
                                                if self.pending_ratio is not None else
                                                self.gate.peak / max(self.gate.baseline,
                                                                     self.gate.min_level)))
        self._trace("words", heard=heard, raw=raw, source=decision.source,
                    confidence=round(decision.confidence, 2), degraded=decision.degraded,
                    reasons="; ".join(decision.reasons)[:200],
                    seconds=self.clock() - started, speaking=self.speaking_text[:60])
        if decision.source == SOURCE_UNCERTAIN:
            # Not her voice and not proven to be a person: hold. The captured span stays pinned
            # for its evidence window, nothing is stripped from the transcript, and no command
            # runs from audio this weak (handoff 6.5: an immediate output change for every raw
            # VAD event would reintroduce self-interruption). The gate is cleared so the candidate
            # is judged again on the next run of loud frames rather than staying stuck open.
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(HOLD, reason="uncertain", heard=heard, raw=raw, level=level01,
                                source=decision.source, confidence=decision.confidence,
                                degraded=decision.degraded)
        own = decision.source == ECHO
        if (
            not own
            and media_just_started
            and not _MEDIA_EXCEPTION.search(f"{heard} {raw}".casefold())
            and strip_wake_word(f"{heard} {raw}", wake_word) is None
        ):
            # The station she is announcing has already started: its presenter is not the user.
            own = True
        if own:
            # The loud sound was Voxa's own voice: carry on speaking (unless the floor was
            # already yielded by the onset, in which case the playback stopped and only the
            # words are rejected). This is the only branch that may strip words, and it is
            # reached on audio evidence - never because the transcript is short or happens to
            # contain a word she is saying herself (task A2: her saying "stop" alone must never
            # stop her, and a person saying "stop" over her saying "stop" must always stop her).
            self.gate.reset()
            self._clear_candidate()
            return BargeOutcome(OWN_VOICE, heard=heard, raw=raw, level=level01,
                                source=decision.source, confidence=decision.confidence,
                                degraded=decision.degraded)
        self.gate.reset()
        self._clear_candidate()
        self.speaking_since = 0.0
        return BargeOutcome(INTERRUPT, heard=heard, raw=raw, level=level01,
                            listening_seconds=self.clock() - started,
                            source=decision.source, confidence=decision.confidence,
                            degraded=decision.degraded)
