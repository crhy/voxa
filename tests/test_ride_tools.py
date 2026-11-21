"""crhy/voxa#50: open a ride provider with the destination pre-filled and read
the quote back. Parsing, the provider URL, the quote reader, the ride_quote tool
through the FakeSession pattern, and the router rule."""

from __future__ import annotations

from datetime import date

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.rides import (
    describe_ride,
    parse_ride_request,
    provider_url,
    quote_from_text,
)
from voxa.agent.tools import default_registry
from voxa.agent.tools.rides import ride_quote, ride_tools
from voxa.agent.tools.web import set_session

TODAY = date(2026, 10, 6)
HOME = "Los Angeles"


# --- parsing ----------------------------------------------------------------


def test_provider_chosen_by_voice() -> None:
    assert parse_ride_request("get me a Lyft to the airport", TODAY, HOME).provider == "Lyft"
    assert parse_ride_request("get an Uber from home to the station at 6am tomorrow", TODAY, HOME).provider == "Uber"
    assert parse_ride_request("call a taxi to O'Hare", TODAY, HOME).provider == "taxi"


def test_home_resolves_to_stored_location() -> None:
    req = parse_ride_request("get an Uber from home to the station at 6am tomorrow", TODAY, HOME)
    assert req is not None
    assert req.pickup == HOME, "from home must resolve to the stored home_location"


def test_destination_and_time_captured() -> None:
    req = parse_ride_request("get an Uber from home to the station at 6am tomorrow", TODAY, HOME)
    assert req is not None
    assert req.destination == "station"
    assert req.at == "6am tomorrow"
    assert req.when == date(2026, 10, 7)


def test_named_pickup_and_destination() -> None:
    req = parse_ride_request("call a taxi from the museum to O'Hare tonight", TODAY, HOME)
    assert req is not None
    assert req.pickup == "the museum"
    assert req.destination == "O'Hare"
    assert req.when == TODAY


def test_rejects_other_intents() -> None:
    for text in (
        "rent a car in Denver next week",
        "find a cheap rental car in Lisbon",
        "get me the cheapest ticket to Hawaii",
        "find me a cheap hotel in Tokyo",
        "reserve a table at the Olive Garden",
        "get milk",
    ):
        assert parse_ride_request(text, TODAY, HOME) is None, text


# --- provider url -----------------------------------------------------------


def test_provider_url_lyft_deep_link() -> None:
    req = parse_ride_request("get me a Lyft from home to the airport", TODAY, HOME)
    assert req is not None
    url = provider_url(req, HOME)
    assert "lyft.com" in url
    assert "Los%20Angeles" in url or "Los Angeles" in url
    assert "airport" in url


def test_provider_url_names_provider_and_places() -> None:
    for phrase, provider in (("call a taxi to the airport", "taxi"), ("get an Uber to the station", "Uber")):
        req = parse_ride_request(phrase, TODAY, HOME)
        assert req is not None
        url = provider_url(req, HOME)
        assert provider.lower() in url.lower()
        assert "Los Angeles" in url or "Los%20Angeles" in url
        assert "airport" in url or "station" in url


def test_describe_ride() -> None:
    req = parse_ride_request("get me a Lyft from home to the airport at 6am tomorrow", TODAY, HOME)
    assert req is not None
    spoken = describe_ride(req)
    assert "Lyft" in spoken
    assert "Los Angeles" in spoken
    assert "airport" in spoken


# --- quote reader -----------------------------------------------------------

PROVIDER_PAGE = (
    "Lyft. Standard fare from your location to the airport is $18.47. "
    "A driver is 6 minutes away. Arrives at 6:15 PM."
)

NO_QUOTE_PAGE = "Sign in with your account to continue booking this trip."


def test_quote_from_text_reads_price_and_eta() -> None:
    price, eta = quote_from_text(PROVIDER_PAGE)
    assert price == "$18.47"
    assert eta == "arrives at 6:15 PM"


def test_quote_from_text_minutes_fallback() -> None:
    price, eta = quote_from_text("Your fare is $22. A car 12 minutes away.")
    assert price == "$22"
    assert eta == "12 minutes away"


def test_quote_from_text_none_when_neither() -> None:
    price, eta = quote_from_text(NO_QUOTE_PAGE)
    assert price is None
    assert eta is None


# --- the tool, driven by a fake session -------------------------------------


class FakeRideSession:
    def __init__(self, text: str) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._text = text

    def goto(self, url, wait=15.0) -> None:
        self.calls.append(("goto", {"url": url}))

    def text(self, limit=6000) -> str:
        self.calls.append(("text", {"limit": limit}))
        return self._text[:limit]

    def click(self, text):
        self.calls.append(("click", {"text": text}))
        return text


@pytest.fixture(autouse=True)
def _isolate_session(monkeypatch):
    set_session(None)

    class _Store:
        class _Settings:
            home_location = HOME

        def load(self):
            return self._Settings()

    from voxa.agent.tools import rides as rides_tools_module

    monkeypatch.setattr(rides_tools_module, "ConfigStore", _Store)
    monkeypatch.setattr(rides_tools_module, "detect_location", lambda: HOME)
    yield
    set_session(None)


def test_ride_quote_reads_price_and_eta_without_ordering() -> None:
    session = FakeRideSession(PROVIDER_PAGE)
    set_session(session)
    result = ride_quote({"request": "call a taxi to the airport"})
    assert result.ok
    assert "$18.47" in result.speech
    assert "arrives at 6:15 PM" in result.speech
    # the provider page it opened named the provider and the destination
    opened = [c for name, c in session.calls if name == "goto"]
    assert opened, "the tool must record the URL it opened"
    url = opened[0]["url"]
    assert "taxi" in url.lower()
    assert "airport" in url
    # nothing was clicked, so no ride was ever requested
    assert not any(name == "click" for name, _ in session.calls)


def test_ride_quote_says_when_no_quote() -> None:
    session = FakeRideSession(NO_QUOTE_PAGE)
    set_session(session)
    result = ride_quote({"request": "call a taxi to the airport"})
    assert result.ok
    assert "no price or time" in result.speech
    assert not any(name == "click" for name, _ in session.calls)


# --- registration and routing ----------------------------------------------


def test_ride_tools_registered() -> None:
    names = default_registry().names()
    for tool in ride_tools():
        assert tool.name in names
    tool = default_registry().get("ride_quote")
    assert int(tool.risk) == 0  # READ_ONLY


def test_ride_routes_to_ride_quote() -> None:
    assert route("get me a lyft to the airport") == ToolCall(
        "ride_quote", {"request": "get me a lyft to the airport"}
    )
    assert route("call a taxi to O'Hare") == ToolCall("ride_quote", {"request": "call a taxi to O'Hare"})


def test_rental_car_still_routes_to_find_deal() -> None:
    call = route("find a cheap rental car in Lisbon")
    assert call is not None
    assert call.tool == "find_deal", call
