"""Family schedule commands route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.family import (
    _family_add_handler,
    _family_remove_handler,
    _family_week_handler,
)


def test_family_phrases_route() -> None:
    assert route("add soccer practice for Emma on Tuesday at 5 pm") == ToolCall(
        "family_add", {"person": "Emma", "event": "soccer practice", "when": "Tuesday at 5 pm"}
    )
    assert route("what's on for the family this week") == ToolCall("family_week", {"person": ""})
    assert route("remove Emma's soccer practice") == ToolCall(
        "family_remove", {"person": "Emma", "event": "soccer practice"}
    )


def test_family_rules_do_not_steal_calendar_or_lists() -> None:
    assert route("what does Emma have this week") == ToolCall("calendar_week", {"person": "Emma"})
    assert route("remove dentist from my calendar") == ToolCall("calendar_cancel", {"title": "dentist"})
    assert route("add milk to my shopping list") == ToolCall("list_add", {"item": "milk", "list": "shopping"})


def test_family_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.family._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.family._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _family_add_handler({"person": "Emma", "event": "soccer practice", "when": "tomorrow at 5 pm"})
    assert added.ok
    assert "Emma: soccer practice" in store["content"]

    week = _family_week_handler({"person": ""})
    assert week.ok and "Emma" in week.speech

    removed = _family_remove_handler({"person": "Emma", "event": "soccer practice"})
    assert removed.ok
    assert "Emma: soccer practice" not in store["content"]


def test_family_add_rejects_unparsable_when(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.family._read", lambda: "")
    monkeypatch.setattr("voxa.agent.tools.family._write", lambda content: None)
    result = _family_add_handler({"person": "Emma", "event": "soccer practice", "when": "soon"})
    assert not result.ok


def test_registry_has_family_tools() -> None:
    names = default_registry().names()
    for name in ("family_add", "family_week", "family_remove"):
        assert name in names
