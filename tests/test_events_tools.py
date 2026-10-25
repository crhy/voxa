"""Voice commands that plan an event or gathering route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools import events as events_module
from voxa.agent.tools import lists as lists_module
from voxa.agent.tools.events import _event_plan_handler


def test_event_phrases_route() -> None:
    assert route("plan a birthday party for Saturday") == ToolCall(
        "event_plan", {"event": "birthday party", "when": "Saturday"}
    )
    assert route("help me organise a barbecue next Sunday") == ToolCall(
        "event_plan", {"event": "barbecue", "when": "next Sunday"}
    )
    assert route("plan a dinner party") == ToolCall("event_plan", {"event": "dinner party", "when": ""})


def test_event_rules_do_not_steal_ordinary_questions() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"


def test_event_plan_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr(lists_module, "_read", lambda: store["content"])
    monkeypatch.setattr(
        lists_module, "_write", lambda content: store.__setitem__("content", content)
    )
    monkeypatch.setattr(
        events_module,
        "ask_model",
        lambda messages: (
            "1. Invite the guests\n2. Send invitations\n3. Plan the food\n"
            "4. Pick the place\n5. Set up decorations\n6. Play some music"
        ),
    )

    planned = _event_plan_handler({"event": "birthday party", "when": ""})
    assert planned.ok
    assert planned.speech == (
        "I made a birthday party plan with 6 things to do and saved it as the "
        "Birthday Party Plan list. First: Invite the guests."
    )
    assert "## Birthday Party Plan" in store["content"]
    assert "Invite the guests" in store["content"]


def test_event_plan_with_a_date(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr(lists_module, "_read", lambda: store["content"])
    monkeypatch.setattr(
        lists_module, "_write", lambda content: store.__setitem__("content", content)
    )
    monkeypatch.setattr(
        events_module,
        "ask_model",
        lambda messages: (
            "1. Invite the guests\n2. Send invitations\n3. Plan the food\n"
            "4. Pick the place\n5. Set up decorations\n6. Play some music"
        ),
    )

    planned = _event_plan_handler({"event": "barbecue", "when": "next Sunday"})
    assert planned.ok
    assert planned.speech.startswith("I made a barbecue plan with 7 things to do")
    assert "First: Date: next Sunday." in planned.speech
    assert "Date: next Sunday" in store["content"]


def test_event_plan_handles_messy_brain(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr(lists_module, "_read", lambda: store["content"])
    monkeypatch.setattr(
        lists_module, "_write", lambda content: store.__setitem__("content", content)
    )
    monkeypatch.setattr(
        events_module,
        "ask_model",
        lambda messages: "Sure! Here is a plan.\n1. Invite guests\n2. Order food\n3. Choose a venue\nextra chatter",
    )
    planned = _event_plan_handler({"event": "dinner party", "when": ""})
    assert planned.ok
    assert "## Dinner Party Plan" in store["content"]
    assert "Invite guests" in store["content"]


def test_event_plan_empty_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(events_module, "ask_model", lambda messages: "")
    result = _event_plan_handler({"event": "picnic", "when": ""})
    assert not result.ok


def test_event_plan_without_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(events_module, "ask_model", None)
    result = _event_plan_handler({"event": "picnic", "when": ""})
    assert not result.ok


def test_registry_has_event_tools() -> None:
    names = default_registry().names()
    assert "event_plan" in names
