"""Spoken hotel phrases route to hotel_options and return 3 booking options, never a reservation."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.hotel import _hotel_options_handler


def test_hotel_phrases_route() -> None:
    assert route("find a hotel in Denver") == ToolCall("hotel_options", {"city": "Denver"})
    assert route("book a hotel room near the airport for 2 nights") == ToolCall(
        "hotel_options", {"city": "the airport", "guests": "2"}
    )
    assert route("find a hotel in Rome from Friday to Sunday") == ToolCall(
        "hotel_options", {"city": "Rome", "check_in": "Friday", "check_out": "Sunday"}
    )
    assert route("look for a hotel around Seattle with 3 guests") == ToolCall(
        "hotel_options", {"city": "Seattle", "guests": "3"}
    )


def test_hotel_options_does_not_steal_travel_or_other_intents() -> None:
    assert route("my flight to Denver was cancelled") == ToolCall(
        "travel_options", {"what": "flight", "place": "Denver", "problem": "cancelled"
        }
    )
    assert route("I missed my connection to Chicago") == ToolCall(
        "travel_options", {"what": "connection", "place": "Chicago", "problem": "missed"}
    )
    assert route("plan a trip to Rome") == ToolCall("trip_plan", {"place": "Rome", "days": ""})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("book a dentist appointment for Friday") == ToolCall(
        "calendar_add", {"title": "a dentist appointment", "when": "Friday"}
    )


def test_hotel_options_handler_with_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.hotel.websearch.search",
        lambda q: [
            websearch_result("Hotel deals in Denver", "Compare last minute hotel deals in Denver, CO"),
            websearch_result("Best budget hotels in Denver", "The 10 best budget hotels in Denver"),
        ],
    )
    monkeypatch.setattr(
        "voxa.agent.tools.hotel.ask_model",
        lambda messages: (
            "1. Grand Hyatt, luxury, book on Booking.com.\n"
            "2. Lavelle Inn, budget, book on Expedia.\n"
            "3. Hostel Fish, budget, book on Hostelworld."
        ),
    )
    result = _hotel_options_handler({"city": "Denver", "check_in": "Friday", "check_out": "Sunday"})
    assert result.ok
    assert result.speech == (
        "Here are hotel options for Denver. I have not booked anything. "
        "Your options: Grand Hyatt, luxury, book on Booking.com.; "
        "Lavelle Inn, budget, book on Expedia.; Hostel Fish, budget, book on Hostelworld."
    )


def test_hotel_options_handler_fallback_without_results(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.hotel.websearch.search", lambda q: [])
    result = _hotel_options_handler({"city": "Rome"})
    assert result.ok
    assert "hotel options for Rome" in result.speech
    assert "Booking.com" in result.speech
    assert "hotel's own website" in result.speech
    assert "front desk" in result.speech


def test_hotel_options_handler_fallback_when_search_fails(monkeypatch) -> None:
    def boom(q: str) -> list:
        raise RuntimeError("no network")

    monkeypatch.setattr("voxa.agent.tools.hotel.websearch.search", boom)
    monkeypatch.setattr(
        "voxa.agent.tools.hotel.ask_model",
        lambda messages: "1. Alpha Hotel.\n2. Beta Inn.\n3. Gamma Lodge.",
    )
    result = _hotel_options_handler({"city": "Denver"})
    assert result.ok
    assert "Alpha" not in result.speech
    assert "Compare rooms" in result.speech


def test_hotel_options_handler_fallback_when_brain_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.hotel.websearch.search",
        lambda q: [websearch_result("x", "y")],
    )
    monkeypatch.setattr("voxa.agent.tools.hotel.ask_model", None)
    result = _hotel_options_handler({"city": "Paris", "guests": "4"})
    assert result.ok
    assert "hotel options for Paris" in result.speech


def test_hotel_options_handler_rejects_missing_city() -> None:
    result = _hotel_options_handler({"city": ""})
    assert not result.ok
    result2 = _hotel_options_handler({})
    assert not result2.ok


def test_registry_has_hotel_options() -> None:
    assert "hotel_options" in default_registry().names()


def websearch_result(title: str, snippet: str):
    from voxa.websearch import SearchResult

    return SearchResult(title=title, url="https://example.com", snippet=snippet)
