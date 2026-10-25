"""Voice commands that plan, read and forget trips route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools import trips as trips_module
from voxa.agent.tools.trips import _trip_forget_handler, _trip_plan_handler, _trip_read_handler


def test_trip_phrases_route() -> None:
    assert route("plan a three day trip to Rome") == ToolCall("trip_plan", {"place": "Rome", "days": "three"})
    assert route("plan a weekend in Chicago") == ToolCall("trip_plan", {"place": "Chicago", "days": "weekend"})
    assert route("make an itinerary for two days in Paris") == ToolCall("trip_plan", {"place": "Paris", "days": "two"})
    assert route("read my trip to Rome") == ToolCall("trip_read", {"place": "Rome"})
    assert route("forget my trip to Rome") == ToolCall("trip_forget", {"place": "Rome"})


def test_trip_rules_do_not_steal_ordinary_questions() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"


def test_trip_plan_read_forget_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr(trips_module, "_read", lambda: store["content"])
    monkeypatch.setattr(trips_module, "_write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr(
        trips_module,
        "ask_model",
        lambda messages: "Day 1: Arrive and see the Colosseum.\nDay 2: Museums and a long lunch.\nDay 3: Day trip and dinner.",
    )

    planned = _trip_plan_handler({"place": "Rome", "days": "three"})
    assert planned.ok
    assert planned.speech == (
        "I planned 3 days in Rome and saved it in Voxa Trips. Day 1: Arrive and see the Colosseum."
    )
    assert "## Rome (3 days)" in store["content"]
    assert "Arrive and see the Colosseum." in store["content"]

    read = _trip_read_handler({"place": "Rome"})
    assert read.ok and "Arrive and see the Colosseum" in read.speech

    forgot = _trip_forget_handler({"place": "Rome"})
    assert forgot.ok and forgot.speech == "I forgot your trip to Rome."
    assert "## Rome" not in store["content"]


def test_trip_plan_handles_messy_brain(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr(trips_module, "_read", lambda: store["content"])
    monkeypatch.setattr(trips_module, "_write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr(
        trips_module,
        "ask_model",
        lambda messages: "Sure! Here is a plan.\n1. Arrive in Rome.\n2. Visit the Vatican.\nextra chatter",
    )
    planned = _trip_plan_handler({"place": "Rome", "days": "two"})
    assert planned.ok
    assert "## Rome (2 days)" in store["content"]
    assert "Arrive in Rome." in store["content"]


def test_trip_plan_empty_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(trips_module, "ask_model", lambda messages: "")
    result = _trip_plan_handler({"place": "Rome", "days": "two"})
    assert not result.ok


def test_trip_plan_without_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(trips_module, "ask_model", None)
    result = _trip_plan_handler({"place": "Rome", "days": "two"})
    assert not result.ok


def test_registry_has_trip_tools() -> None:
    names = default_registry().names()
    for name in ("trip_plan", "trip_read", "trip_forget"):
        assert name in names
