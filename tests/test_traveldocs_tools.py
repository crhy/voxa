"""Voice commands for travel documents route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.traveldocs import (
    _travel_doc_add_handler,
    _travel_doc_forget_handler,
    _travel_doc_list_handler,
    _travel_doc_when_handler,
)


def test_travel_doc_phrases_route() -> None:
    assert route("remember my passport expires on June 5 2028") == ToolCall(
        "travel_doc_add", {"document": "passport", "expires": "June 5 2028"}
    )
    assert route("when does my passport expire") == ToolCall("travel_doc_when", {"document": "passport"})
    assert route("what travel documents do I have") == ToolCall("travel_doc_list", {})
    assert route("forget my passport") == ToolCall("travel_doc_forget", {"document": "passport"})


def test_travel_doc_rules_do_not_steal_other_tools() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_travel_doc_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.traveldocs._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.traveldocs._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _travel_doc_add_handler({"document": "passport", "expires": "June 5 2028"})
    assert added.ok
    assert "- Passport: expires 2028-06-05" in store["content"]

    read_back = _travel_doc_when_handler({"document": "passport"})
    assert read_back.ok
    assert "passport" in read_back.speech.lower()

    listed = _travel_doc_list_handler({})
    assert listed.ok
    assert "1 travel document" in listed.speech

    forgotten = _travel_doc_forget_handler({"document": "passport"})
    assert forgotten.ok
    assert "- Passport:" not in store["content"]


def test_travel_doc_add_rejects_bad_date() -> None:
    result = _travel_doc_add_handler({"document": "passport", "expires": "sometime"})
    assert not result.ok


def test_registry_has_travel_doc_tools() -> None:
    names = default_registry().names()
    assert "travel_doc_add" in names
    assert "travel_doc_when" in names
    assert "travel_doc_list" in names
    assert "travel_doc_forget" in names
