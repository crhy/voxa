"""State-machine tests for the DICTATING assistant state (#84)."""

from __future__ import annotations

from voxa.ui.assistant_view import STATE_CAPTIONS, STATE_HINTS
from voxa.ui.state import ALLOWED_TRANSITIONS, AssistantModel, AssistantState, can_transition

_S = AssistantState


def test_dictating_transitions() -> None:
    for source in (_S.READY, _S.LISTENING, _S.SPEAKING):
        assert _S.DICTATING in ALLOWED_TRANSITIONS[source]
    assert ALLOWED_TRANSITIONS[_S.DICTATING] >= {_S.READY, _S.LISTENING, _S.PAUSED, _S.OFFLINE, _S.ERROR}
    assert not can_transition(_S.OFFLINE, _S.DICTATING)
    assert not can_transition(_S.THINKING, _S.DICTATING)
    assert not can_transition(_S.DICTATING, _S.THINKING)
    assert not can_transition(_S.DICTATING, _S.WORKING)
    assert not can_transition(_S.DICTATING, _S.WAITING)


def test_dictating_round_trip() -> None:
    model = AssistantModel()
    assert model.set_state(_S.READY)
    assert model.set_state(_S.DICTATING)
    assert model.state is _S.DICTATING
    assert model.set_state(_S.PAUSED)  # pause works from DICTATING
    assert model.set_state(_S.READY)
    assert model.set_state(_S.LISTENING)
    assert model.set_state(_S.DICTATING)
    assert model.set_state(_S.OFFLINE)  # offline works from DICTATING
    assert model.state is _S.OFFLINE
    assert not model.set_state(_S.DICTATING)  # OFFLINE refuses dictating
    assert model.state is _S.OFFLINE


def test_dictating_caption_and_hint() -> None:
    assert STATE_CAPTIONS[_S.DICTATING] == "Dictating"
    assert STATE_HINTS[_S.DICTATING] == "Say “stop dictation” when you are done."
