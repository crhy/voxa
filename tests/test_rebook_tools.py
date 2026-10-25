"""A cancelled or delayed trip routes to travel_options and returns 3 options."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.rebook import _travel_options_handler


def test_travel_phrases_route() -> None:
    assert route("my flight to Denver was cancelled") == ToolCall(
        "travel_options", {"what": "flight", "place": "Denver", "problem": "cancelled"}
    )
    assert route("my train to London was delayed") == ToolCall(
        "travel_options", {"what": "train", "place": "London", "problem": "delayed"}
    )
    assert route("I missed my connection to Chicago") == ToolCall(
        "travel_options", {"what": "connection", "place": "Chicago", "problem": "missed"}
    )
    assert route("my bus to Paris was cancelled what are my options") == ToolCall(
        "travel_options", {"what": "bus", "place": "Paris", "problem": "cancelled"}
    )


def test_travel_options_does_not_steal_other_intents() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("add milk to my shopping list") == ToolCall(
        "list_add", {"item": "milk", "list": "shopping"}
    )
    assert route("plan a trip to Rome") != ToolCall("travel_options", {})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_travel_options_handler_with_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.rebook.websearch.search",
        lambda q: [
            websearch_result("United cancels flights", "United Airlines cancelled flight help"),
            websearch_result("Rebook options", "How to rebook a cancelled flight"),
        ],
    )
    monkeypatch.setattr(
        "voxa.agent.tools.rebook.ask_model",
        lambda messages: "1. Call United and ask for a rebook.\n2. Use the United app to rebook.\n3. Ask for a refund or hotel voucher.",
    )
    result = _travel_options_handler({"what": "flight", "place": "Denver", "problem": "cancelled"})
    assert result.ok
    assert result.speech == (
        "Sorry about your flight. Your options: Call United and ask for a rebook.; "
        "Use the United app to rebook.; Ask for a refund or hotel voucher."
    )


def test_travel_options_handler_fallback_without_results(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.rebook.websearch.search", lambda q: [])
    result = _travel_options_handler({"what": "train", "place": "London", "problem": "delayed"})
    assert result.ok
    assert "options" in result.speech
    assert "Call the carrier" in result.speech
    assert "Rebook in the carrier" in result.speech
    assert "refund or a hotel voucher" in result.speech


def test_travel_options_handler_fallback_when_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.rebook.websearch.search",
        lambda q: [websearch_result("x", "y")],
    )
    monkeypatch.setattr("voxa.agent.tools.rebook.ask_model", None)
    result = _travel_options_handler({"what": "bus", "place": "Paris", "problem": "cancelled"})
    assert result.ok
    assert "options" in result.speech


def test_travel_options_handler_rejects_missing_args() -> None:
    result = _travel_options_handler({"what": "", "place": "Denver", "problem": "cancelled"})
    assert not result.ok
    result2 = _travel_options_handler({"what": "flight", "place": "", "problem": "cancelled"})
    assert not result2.ok


def test_registry_has_travel_options() -> None:
    assert "travel_options" in default_registry().names()


def websearch_result(title: str, snippet: str):
    from voxa.websearch import SearchResult

    return SearchResult(title=title, url="https://example.com", snippet=snippet)
