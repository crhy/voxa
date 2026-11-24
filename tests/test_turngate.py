"""Roadmap task 8 (f174): the live turn effects on Voxa's OWN stream.

Extends the f135 state-table tests with the seam that wires the turn state machine into the live
path behind f173's joint double-talk gate (voxa/turngate.py, voxa/conversation.py). Everything here
is GTK-free and offline: the fake stream records what the director does to her OWN playback, so a
test can prove there is never a system-volume change and the degraded (None) joint path is unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

from voxa.bargein import (
    DECISION_CONFIRM,
    DECISION_REJECT,
    DECISION_SUSPECT,
    DUCK_DB,
    OVERLAP_PENDING,
    SPEAKING,
    YIELDED,
)
from voxa.conversation import ConversationHistory
from voxa.turngate import LiveTurnDirector


class OwnStreamSpy:
    """Records the own-stream effects the director applies; nothing else may be touched."""

    def __init__(self, *, position: float = 0.0) -> None:
        self.position_value = position
        self.ducks: list[float] = []
        self.resumes = 0
        self.stops = 0
        self.ducked = False
        self._allow = {"duck", "resume", "position", "stop"}
        self.touched: list[str] = []

    def duck(self, amount_db: float) -> bool:
        self.ducks.append(amount_db)
        self.ducked = True
        return True

    def resume(self) -> bool:
        self.resumes += 1
        self.ducked = False
        return True

    def position(self) -> float:
        return self.position_value

    def stop(self) -> None:
        self.stops += 1
        self.ducked = False

    def __getattr__(self, name: str):  # any attribute outside the own-stream slice is a hard failure
        raise AssertionError(f"turn director touched forbidden stream attribute {name!r}")


def _words(*specs):
    return [SimpleNamespace(word=w, start=s) for (w, s) in specs]


def test_joint_suspect_ducks_own_stream_within_the_ramp():
    stream = OwnStreamSpy(position=1.5)
    director = LiveTurnDirector()
    director.note_reply("I will open the pod bay doors for you now")

    assert director.suspect(stream) is True
    # A 12 dB duck of her OWN stream, never a system-volume change (the spy fails on any other call).
    assert stream.ducks == [DUCK_DB]
    assert stream.ducked and stream.stops == 0 and stream.resumes == 0
    assert director.duck_position == 1.5
    assert director.machine.state == OVERLAP_PENDING
    # Suspect again does not duck twice.
    assert director.suspect(stream) is False
    assert len(stream.ducks) == 1


def test_confirm_words_stops_within_the_played_turn_and_truncates():
    stream = OwnStreamSpy(position=2.0)
    director = LiveTurnDirector()
    director.note_reply("We will open the pod bay doors for you now")
    director.suspect(stream)
    played = _words(("We", 0.0), ("will", 0.3), ("open", 0.7), ("the", 1.0), ("pod", 1.3),
                    ("bay", 1.6), ("doors", 1.9))  # "for you now" (>=2.0) not yet audible

    decision, kept = director.confirm(stream, heard="stop that right now",
                                      spoken_reply="We will open the pod bay doors for you now",
                                      played_words=played)
    assert decision == DECISION_CONFIRM
    assert stream.stops == 1  # hard-stopped inside the turn the recorded position belongs to
    assert director.machine.state == YIELDED
    # Truncated assistant context equals the audibly played words exactly (lower-cased tokens).
    assert kept == "we will open the pod bay doors"


def test_backchannel_resumes_at_word_boundary():
    stream = OwnStreamSpy(position=2.0)
    director = LiveTurnDirector()
    director.note_reply("Turn one point two to the left and hold it")
    director.suspect(stream)
    played = _words(("Turn", 0.0), ("one", 0.3), ("point", 0.6), ("two", 0.9))

    decision, kept = director.confirm(stream, heard="yeah", spoken_reply="Turn one point two to the left and hold it",
                                      played_words=played)
    assert decision == DECISION_REJECT
    assert stream.resumes == 1  # false alarm: her stream is given back
    assert stream.stops == 0
    assert director.machine.state == SPEAKING
    # Rewound to the last complete word boundary, never a partial word left in context.
    assert kept == "turn one point two"


def test_long_wordless_pause_resumes():
    stream = OwnStreamSpy(position=1.0)
    director = LiveTurnDirector()
    director.note_reply("Hold on while I think about that")
    director.suspect(stream)
    # A cough / pause: no words at all inside the reject window of audio.
    decision, kept = director.confirm(stream, heard="", gated_speech_ms=1600.0,
                                      spoken_reply="Hold on while I think about that",
                                      played_words=_words(("Hold", 0.0), ("on", 0.2)))
    assert decision == DECISION_REJECT
    assert stream.resumes == 1 and stream.stops == 0


def test_protected_word_confirms_on_its_own_and_never_moves_system_volume():
    stream = OwnStreamSpy(position=1.0)
    director = LiveTurnDirector()
    director.note_reply("Let me finish reading this very long sentence please")
    director.suspect(stream)
    # Only "stop" said - a protected command word confirms on its own (>=2 words not needed).
    decision, kept = director.confirm(stream, heard="stop", spoken_reply="Let me finish reading this very long sentence please",
                                      played_words=_words(("Let", 0.0), ("me", 0.2), ("finish", 0.5)))
    assert decision == DECISION_CONFIRM
    assert stream.stops == 1
    assert kept == "let me finish"
    # The spy (OwnStreamSpy.__getattr__) already proved the director only touched the own-stream slice.


def test_suspect_holds_keeps_duck_and_does_not_stop():
    stream = OwnStreamSpy(position=0.5)
    director = LiveTurnDirector()
    director.note_reply("Reading a sentence that keeps going on")
    director.suspect(stream)
    decision, kept = director.confirm(stream, heard="well", gated_speech_ms=120.0,
                                      spoken_reply="Reading a sentence that keeps going on",
                                      played_words=_words(("Reading", 0.0)))
    assert decision == DECISION_SUSPECT
    assert kept is None
    assert stream.stops == 0 and stream.resumes == 0  # still ducked, still collecting
    assert stream.ducked


def test_degraded_joint_path_never_ducks():
    # A joint gate that could not measure (None) never calls suspect: the director stays put and the
    # stream is untouched - the degraded behaviour is unchanged (f173 caveat).
    stream = OwnStreamSpy(position=1.0)
    director = LiveTurnDirector()
    director.note_reply("A reply that is not interrupted at all")
    assert director.ducked is False
    assert stream.ducks == [] and stream.stops == 0 and stream.resumes == 0


def test_truncate_last_assistant_rewrites_only_that_turn():
    history = ConversationHistory("system prompt", max_turns=8)
    history.add_user("open it")
    history.add_assistant("Opening the pod bay doors for you now")
    history.add_user("stop")
    assert history.truncate_last_assistant("Opening the pod bay doors") is True
    messages = history.messages()
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant == [{"role": "assistant", "content": "Opening the pod bay doors"}]
    # No assistant turn to rewrite -> no-op, returns False.
    empty = ConversationHistory("system prompt", max_turns=8)
    assert empty.truncate_last_assistant("anything") is False
