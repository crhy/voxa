"""The LIVE roadmap task-8 turn effects: duck on the joint suspect, yield on words, resume on a
false alarm.

``voxa/jointgate.py`` (f173) kept the joint *verdicts* in one place so the live path and the bench
could not drift; this module does the same for the *turn states* (report 182-183, handoff 6.5:218-236).
It is pure and importable without GTK - ``voxa/bargein.py`` supplies the state machine and the three
states, and this seam maps a hearing outcome onto effects on Voxa's OWN stream. ``voxa/window.py``
owns one :class:`LiveTurnDirector` and performs the returned effects on ``SpeechService`` and the LLM
history; the window itself is not unit-testable headless, so this seam is where the wiring is tested.

No model, no GTK, no audio device lives here: it only decides what to do with the reply. The system
volume is never touched - ``duck``/``resume`` act on the caller-supplied stream object alone (handoff
233, brief design rule: "duck her OWN stream, never lower system volume").
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .bargein import (
    DECISION_CONFIRM,
    DECISION_REJECT,
    DECISION_SUSPECT,
    SPEAKING,
    YIELDED,
    TurnStateMachine,
    rewind_to_last_word_boundary,
    truncate_to_heard,
)


class OwnStream(Protocol):
    """The slice of :class:`voxa.speech.SpeechService` the turn director acts on.

    ``duck()`` lowers this stream's own gain by 12 dB over the 20 ms ramp (never the system
    volume); ``resume()`` restores it; ``position()`` is the seconds of audio actually played;
    ``stop()`` hard-stops the reply. The live app passes the real service, the tests a fake that
    records the calls.
    """

    def duck(self, amount_db: float) -> bool: ...

    def resume(self) -> bool: ...

    def position(self) -> float: ...

    def stop(self) -> None: ...


class LiveTurnDirector:
    """One reply's explicit turn states and its own-stream duck, driven by the hearing pipeline.

    The window calls :meth:`note_reply` when a reply starts, :meth:`suspect` on the joint gate's
    suspect, and :meth:`confirm` / :meth:`reject` on the later word decision. ``confirm`` returns
    the assistant text to keep in the LLM context (the words audibly played); ``reject`` returns the
    text rewound to the last word boundary. A joint path that could not measure (INDETERMINATE /
    ``None``) never calls suspect, so the degraded behaviour is unchanged.
    """

    def __init__(self, machine: TurnStateMachine | None = None) -> None:
        self.machine = machine if machine is not None else TurnStateMachine()
        self.duck_position: float | None = None
        self.ducked = False
        self._spoke_a_word = False  # has at least one reply word been audible (for reject timing)

    # -- reply context (handoff 6.5:235: a short "yes" can be an ANSWER) --------------------
    def note_reply(self, text: str, asked_yes_no: bool = False) -> None:
        self.machine.note_speaking(text)
        self.machine.note_question(asked_yes_no)
        self.duck_position = None
        self.ducked = False
        self._spoke_a_word = False

    def note_question(self, asked_yes_no: bool) -> None:
        self.machine.note_question(asked_yes_no)

    # -- row 2: the joint-gate suspect -> a reversible duck of her OWN stream ---------------
    def suspect(self, stream: OwnStream) -> bool:
        """Pin the span and DUCK her own stream once, recording the sample position.

        The handoff's reversible pause (6.5:233): a suspect lowers her stream, it never mutes the
        mic and never touches the system volume. Already ducked -> nothing more to do. Returns
        whether a duck was actually applied to the stream.
        """
        if self.machine.user_onset() != DECISION_SUSPECT:
            return False
        if self.ducked:
            return False
        self.duck_position = stream.position()
        self.ducked = bool(stream.duck(12.0))
        return self.ducked

    # -- rows 3/4: CONFIRM yields, REJECT resumes, SUSPECT holds ----------------------------
    def confirm(self, stream: OwnStream, *, heard: str, gated_speech_ms: float = 0.0,
                no_speech_prob: float | None = None, spoken_reply: str = "",
                played_words: Iterable = ()) -> tuple[str, str | None]:
        """Decide the overlap candidate and apply the own-stream effect.

        Returns ``(decision, kept_context)``: on CONFIRM the stream is hard-stopped and
        ``kept_context`` is the assistant reply truncated to the words audibly played; on REJECT the
        duck is resumed and ``kept_context`` is that reply rewound to the last word boundary; on
        SUSPECT the stream stays ducked and collecting with ``kept_context`` ``None``.
        """
        audible = self._audible_prefix(played_words, stream.position())
        decision = self.machine.confirm(heard, gated_speech_ms=gated_speech_ms,
                                        no_speech_prob=no_speech_prob)
        if decision == DECISION_CONFIRM:
            stream.stop()
            self.ducked = False
            self.duck_position = None
            return decision, truncate_to_heard(spoken_reply, audible)
        if decision == DECISION_REJECT:
            self.reject(stream)
            return decision, rewind_to_last_word_boundary(spoken_reply, audible)
        return decision, None

    def reject(self, stream: OwnStream) -> bool:
        """Resume a ducked stream: a false alarm gives the floor back where she left off."""
        restored = bool(stream.resume())
        self.ducked = False
        self.duck_position = None
        return restored

    # -- the CONFIRM / REJECT effects, driven by the window's (bargectl) decision ----------
    # The live app's authoritative word decision is BargeInController.observe (attribution, the
    # own-voice check, wake-word, offline, media); the turn director performs the OWN-STREAM effect
    # that decision implies and computes the assistant-context truncation. decide_bargein (imported
    # above, used by confirm()/reject() and the tests) is the task-8 classifier these effects mirror.
    def yield_turn(self, stream: OwnStream, *, spoken_reply: str,
                   played_words: Iterable) -> str | None:
        """Task 8 CONFIRM effect: the floor is taken. Hard-stop her own stream and return the
        assistant reply truncated to the words audibly played (report 76:76)."""
        audible = self._audible_prefix(played_words, stream.position())
        stream.stop()
        self.ducked = False
        self.duck_position = None
        self.machine.state = YIELDED
        return truncate_to_heard(spoken_reply, audible)

    def resume_turn(self, stream: OwnStream, *, spoken_reply: str,
                    played_words: Iterable) -> str | None:
        """Task 8 REJECT effect: a false alarm. Resume the duck and return the reply rewound to
        the last word boundary (a long pause-then-reject never leaves a partial word in context)."""
        audible = self._audible_prefix(played_words, stream.position())
        self.reject(stream)
        self.machine.state = SPEAKING
        return rewind_to_last_word_boundary(spoken_reply, audible)

    # -- row 6/8 helpers (the window drives the acoustic side; these keep the state coherent) --
    def note_pause(self) -> None:
        self.machine.user_pause()

    def note_alignment_failed(self) -> None:
        self.machine.alignment_failed()

    @staticmethod
    def _audible_prefix(played_words: Iterable, upto: float) -> str:
        """The words her speakers were AUDIBLE reading by ``upto`` seconds, in order.

        ``played_words`` are :class:`voxa.ownspeech.PlayedWord`-shaped (``word`` and ``start`` in
        audible/playback time). Words that started after the stop position are not "audibly played"
        and are dropped (report 76:76: the context describes what was actually heard).
        """
        out: list[str] = []
        for played in played_words:
            word = getattr(played, "word", None)
            if word is None:
                continue
            if getattr(played, "start", 0.0) <= upto:
                out.append(word)
        return " ".join(out)
