"""Tests for voxa.ui.state (issue #5 assistant state model)."""

from voxa.ui.state import STATUS_TEXT, AssistantState


def test_every_state_has_a_caption() -> None:
    assert set(STATUS_TEXT) == set(AssistantState)
    assert all(caption.strip() for caption in STATUS_TEXT.values())


def test_state_values_are_stable_strings() -> None:
    assert AssistantState("listening") is AssistantState.LISTENING
    assert AssistantState.OFFLINE.value == "offline"
    assert "…" in STATUS_TEXT[AssistantState.LISTENING]
