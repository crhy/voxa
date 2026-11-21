from __future__ import annotations

from datetime import date

import pytest

from voxa.agent.cdp import CdpError
from voxa.agent.deal_results import parse_results
from voxa.agent.deals import build_url, describe, parse_request
from voxa.agent.intents import route
from voxa.agent.result import ToolResult
from voxa.agent.tools import default_registry
from voxa.agent.tools.deals import read_deal_results, set_session
from voxa.agent.tools.web import get_session
from voxa.config import Settings

FLIGHT_PAGE = (
    "Flights from Los Angeles to Dublin Depart October 15 Return October 22 "
    "United Airlines 1 stop 4 h 20 m $312 via Chicago "
    "KLM 1 stop 6 h 30 m $425 via Amsterdam "
    "Delta 1 stop 5 h 10 m $389 via Cleveland "
    "American Airlines nonstop 8 h 45 m $610 "
    "Prices starting from $289 Sorted by Price"
)

HOTEL_PAGE = (
    "Hotels in Dublin checking in October 15 for 7 nights "
    "Premier Inn Dublin City Centre 4 star $89.00 per night "
    "Motel 6 Dublin budget $54.00 per night "
    "Grand Hotel Dublin 5 star $120.00 per night"
)

PRODUCT_PAGE = (
    "4070 Ti Super for sale "
    "NVIDIA 4070 Ti Super graphics card $1,199.99 Best price "
    "4070 Ti Super by Acer $1,349 "
    "4070 Ti Super 8GB $1,099"
)

EMPTY_PAGE = "Please enable JavaScript to use this site."


class FakeSession:
    def __init__(self):
        self.calls = []
        self._url = "https://example.com/"
        self._text = EMPTY_PAGE
        self._outline: list[dict] = []
        self._goto_raises = False

    def ensure(self):
        self.calls.append(("ensure", {}))

    def goto(self, url, wait=15.0):
        self.calls.append(("goto", {"url": url}))
        if self._goto_raises:
            raise CdpError("navigation failed")
        self._url = url

    def url(self):
        self.calls.append(("url", {}))
        return self._url

    def text(self, limit=6000):
        self.calls.append(("text", {"limit": limit}))
        return self._text[:limit]

    def outline(self, limit=60):
        self.calls.append(("outline", {"limit": limit}))
        return self._outline


def _session(page: str, outline: list[dict] | None = None) -> FakeSession:
    session = FakeSession()
    session._text = page
    session._outline = outline or []
    set_session(session)
    assert get_session() is session
    return session


def _known_place(monkeypatch) -> None:
    """A home location, so no test reaches the geolocation services."""

    class Store:
        def __init__(self, path=None):
            self.path = path

        def load(self, *, migrate: bool = True) -> Settings:
            return Settings(home_location="Los Angeles")

        def save(self, settings: Settings) -> None:
            pass

    monkeypatch.setattr("voxa.agent.tools.deals.ConfigStore", Store)


def _nowhere(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.deals.ConfigStore", lambda path=None: _EmptyStore())
    monkeypatch.setattr("voxa.agent.tools.deals.detect_location", lambda fetch=None: "")


class _EmptyStore:
    def load(self, *, migrate: bool = True) -> Settings:
        return Settings()

    def save(self, settings: Settings) -> None:
        pass


def _request(text: str) -> str:
    req = parse_request(text, date.today())
    assert req is not None, text
    return build_url(req, "Los Angeles")[0]


def test_flight_page_parses_three_with_price_duration_stops() -> None:
    results = parse_results(FLIGHT_PAGE, "flight")
    assert len(results) == 3
    assert [entry["price"] for entry in results] == [312.0, 389.0, 425.0]
    assert [entry["label"] for entry in results] == ["United Airlines", "Delta", "KLM"]
    assert [entry["stops"] for entry in results] == [1, 1, 1]
    assert [entry["duration"] for entry in results] == [260, 310, 390]


def test_flight_page_parses_at_most_three() -> None:
    assert len(parse_results(FLIGHT_PAGE, "flight")) == 3
    assert len(parse_results(FLIGHT_PAGE, "flight", limit=2)) == 2
    assert parse_results(FLIGHT_PAGE, "flight", limit=0) == []


def test_read_back_speaks_the_cheapest_first(monkeypatch) -> None:
    _known_place(monkeypatch)
    session = _session(FLIGHT_PAGE)
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert result.ok
    assert result.speech.startswith("Cheapest: United Airlines, 1 stop, 4 hours 20 minutes, 312 dollars.")
    assert "Second: Delta" in result.speech
    assert "Third: KLM" in result.speech
    assert result.speech.count("stop") == 3
    assert "hours" in result.speech
    assert ("goto", {"url": _request("cheapest flight to Dublin")}) in session.calls


def test_read_back_keeps_three_lines_in_detail(monkeypatch) -> None:
    _known_place(monkeypatch)
    _session(FLIGHT_PAGE)
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert "United Airlines, 1 stop, 4 h 20 m, $312" in result.detail
    assert "Delta, 1 stop, 5 h 10 m, $389" in result.detail
    assert "KLM, 1 stop, 6 h 30 m, $425" in result.detail
    assert result.detail.count("\n") == 3


def test_goto_uses_the_url_build_url_produces(monkeypatch) -> None:
    _known_place(monkeypatch)
    session = _session(FLIGHT_PAGE)
    read_deal_results({"request": "get me the cheapest ticket to Hawaii"})
    url, _site = build_url(parse_request("get me the cheapest ticket to Hawaii", date.today()), "Los Angeles")
    assert ("goto", {"url": url}) in session.calls
    assert session.calls[0] == ("ensure", {})


def test_hotel_page_parses_label_and_price() -> None:
    results = parse_results(HOTEL_PAGE, "hotel")
    assert [entry["price"] for entry in results] == [54.0, 89.0, 120.0]
    assert results[0]["label"] == "Motel 6 Dublin budget"
    assert all(entry["duration"] is None and entry["stops"] is None for entry in results)


def test_hotel_page_read_back(monkeypatch) -> None:
    _known_place(monkeypatch)
    _session(HOTEL_PAGE)
    result = read_deal_results({"request": "find me a cheap hotel in Dublin"})
    assert result.ok
    assert result.speech.startswith("Cheapest: Motel 6 Dublin budget, 54 dollars.")
    assert "stop" not in result.speech


def test_product_page_parses_label_and_price() -> None:
    results = parse_results(PRODUCT_PAGE, "product")
    assert [entry["price"] for entry in results] == [1099.0, 1199.99, 1349.0]
    assert results[0]["label"] == "4070 Ti Super 8GB"
    assert all(entry["duration"] is None and entry["stops"] is None for entry in results)


def test_product_page_read_back(monkeypatch) -> None:
    _known_place(monkeypatch)
    _session(PRODUCT_PAGE)
    result = read_deal_results({"request": "best price on a 4070 Ti Super"})
    assert result.ok
    assert result.speech.startswith("Cheapest: 4070 Ti Super 8GB, 1099 dollars.")


def test_outline_only_page_reads_back(monkeypatch) -> None:
    _known_place(monkeypatch)
    outline = [
        {"n": 1, "kind": "link", "text": "United Airlines 1 stop 4 h 20 m $312", "placeholder": "", "href": "/1"},
        {"n": 2, "kind": "link", "text": "Delta 1 stop 5 h 10 m $389", "placeholder": "", "href": "/2"},
        {"n": 3, "kind": "link", "text": "KLM 1 stop 6 h 30 m $425", "placeholder": "", "href": "/3"},
    ]
    _session(EMPTY_PAGE, outline)
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert result.ok
    assert result.speech.startswith("Cheapest: United Airlines, 1 stop, 4 hours 20 minutes, 312 dollars.")


def test_unparseable_page_falls_back_to_describe(monkeypatch) -> None:
    _known_place(monkeypatch)
    session = _session(EMPTY_PAGE)
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert result.ok
    req = parse_request("cheapest flight to Dublin", date.today())
    assert req is not None
    assert result.speech == f"{describe(req, 'Los Angeles')} The page itself still needs your eyes."
    assert result.detail == _request("cheapest flight to Dublin")
    assert ("goto", {"url": _request("cheapest flight to Dublin")}) in session.calls


def test_cdp_error_gives_a_failure(monkeypatch) -> None:
    _known_place(monkeypatch)
    session = _session(FLIGHT_PAGE)
    session._goto_raises = True
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert result.speech == "The browser did not open that page."
    assert result.detail == "navigation failed"


def test_unknown_location_reuses_no_location(monkeypatch) -> None:
    _nowhere(monkeypatch)
    session = _session(FLIGHT_PAGE)
    result = read_deal_results({"request": "cheapest flight to Dublin"})
    assert not result.ok
    assert result.speech == "I couldn't tell where you are. Say \u201cI'm in <city>\u201d."
    assert session.calls == []


def test_a_request_that_is_not_a_search_reads_the_open_page(monkeypatch) -> None:
    _known_place(monkeypatch)
    session = _session(FLIGHT_PAGE)
    session._url = "https://www.expedia.com/Flights-Search?trip=roundtrip"
    result = read_deal_results({"request": "compare them"})
    assert result.ok
    assert result.speech.startswith("Cheapest: United Airlines, 1 stop, 4 hours 20 minutes, 312 dollars.")
    assert session.calls == [("text", {"limit": 6000}), ("outline", {"limit": 60}), ("url", {})]


def test_a_request_that_is_not_a_search_with_nothing_open(monkeypatch) -> None:
    _known_place(monkeypatch)
    _session(EMPTY_PAGE)
    result = read_deal_results({"request": "compare them"})
    assert not result.ok
    assert result.speech == "Nothing on the page could be read back."


def test_deal_results_in_registry() -> None:
    names = default_registry().names()
    assert "deal_results" in names
    tool = default_registry().get("deal_results")
    assert tool.risk == 0
    assert tool.required == ("request",)


@pytest.mark.parametrize(
    "phrase",
    [
        "read back the flights",
        "read back the top three",
        "what are the cheapest flights",
        "compare them",
        "tell me the top three",
        "compare the options",
    ],
)
def test_read_back_phrases_route_to_deal_results(phrase: str) -> None:
    call = route(phrase)
    assert call is not None
    assert call.tool == "deal_results"
    assert call.args == {"request": phrase}


@pytest.mark.parametrize(
    "phrase",
    [
        "get me the cheapest ticket to Hawaii",
        "best price on a 4070 Ti Super",
        "cheapest flight to Hawaii",
        "find me a cheap hotel in Tokyo for 4 nights",
    ],
)
def test_plain_searches_still_route_to_find_deal(phrase: str) -> None:
    call = route(phrase)
    assert call is not None
    assert call.tool == "find_deal"
