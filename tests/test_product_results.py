from __future__ import annotations

from datetime import date

import pytest

import voxa.agent.tools.deals as deal_tools
from voxa.agent.cdp import CdpError
from voxa.agent.deal_results import parse_products, spoken_products
from voxa.agent.deals import build_url, describe, parse_request
from voxa.agent.tools import default_registry
from voxa.agent.tools.deals import NO_LOCATION, read_products
from voxa.agent.tools.web import set_session
from voxa.config import Settings

TODAY = date(2026, 10, 6)
LOCATION = "Los Angeles"
REQUEST = "best price on a 4070 Ti Super"

# A shopping results page as BrowserSession.text() returns it: one line, whitespace
# collapsed, page furniture between the rows.
PAGE = (
    "Best price on a 4070 Ti Super - Shopping results Sponsored Starting from $25 "
    "Anker 5-in-1 hub $31 4.6 out of 5 1,204 reviews from Amazon in stock "
    "Baseus 4-port USB-C charger $28.99 4.2 out of 5 803 reviews on eBay "
    "UGREEN 20W power bank $45 4.9 out of 5 2,310 reviews from Amazon "
    "Sign in More results Compare prices"
)
PAGE_NO_RATINGS = (
    "Standing desk results "
    "Bambicore single arm $18.99 from Amazon in stock "
    "Sitris desk $24.50 on eBay "
    "Adjustable riser $30 at Walmart"
)
PAGE_JUNK = "Please sign in to continue browsing. Loading more matches."


class FakeSession:
    def __init__(self):
        self.calls = []
        self._url = "https://example.com/"
        self._text = PAGE
        self._outline = [
            {
                "n": 1,
                "kind": "link",
                "text": "Anker 5-in-1 hub $31 4.6 out of 5 1,204 reviews from Amazon",
                "placeholder": "",
                "href": "/a",
            },
            {
                "n": 2,
                "kind": "link",
                "text": "Baseus 4-port USB-C charger $28.99 4.2 out of 5 803 reviews on eBay",
                "placeholder": "",
                "href": "/b",
            },
            {
                "n": 3,
                "kind": "link",
                "text": "UGREEN 20W power bank $45 4.9 out of 5 2,310 reviews from Amazon",
                "placeholder": "",
                "href": "/c",
            },
        ]
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


class _FakeStore:
    def __init__(self, home_location: str) -> None:
        self._settings = Settings(home_location=home_location)

    def load(self) -> Settings:
        return self._settings

    def save(self, settings: Settings) -> None:
        pass


@pytest.fixture(autouse=True)
def _home_location(monkeypatch):
    """A known city, so no location lookup reaches the network."""
    monkeypatch.setattr(deal_tools, "ConfigStore", lambda: _FakeStore(LOCATION))


def _fresh_session() -> FakeSession:
    session = FakeSession()
    set_session(session)
    return session


def _request():
    req = parse_request(REQUEST, TODAY)
    assert req is not None
    return req


def test_parse_products_carries_price_and_rating() -> None:
    items = parse_products(PAGE)
    assert [item["label"] for item in items] == [
        "Baseus 4-port USB-C charger",
        "Anker 5-in-1 hub",
        "UGREEN 20W power bank",
    ]
    assert [item["price"] for item in items] == [28.99, 31.0, 45.0]
    assert [item["rating"] for item in items] == [4.2, 4.6, 4.9]
    assert [item["seller"] for item in items] == ["eBay", "Amazon", "Amazon"]


def test_parse_products_stops_at_three_and_puts_the_cheapest_first() -> None:
    items = parse_products(PAGE)
    assert len(items) == 3
    assert len(parse_products(PAGE, limit=2)) == 2
    assert items[0]["price"] == min(item["price"] for item in items)


def test_parse_products_of_an_unparseable_page_is_empty() -> None:
    assert parse_products(PAGE_JUNK) == []
    assert parse_products("") == []


def test_spoken_products_names_the_cheapest_first_with_its_rating() -> None:
    speech = spoken_products(parse_products(PAGE), _request())
    assert speech.startswith("Cheapest: Baseus 4-port USB-C charger, $28.99, rated 4.2 out of 5")
    assert "Second: Anker 5-in-1 hub, $31" in speech
    assert "Third: UGREEN 20W power bank, $45" in speech
    assert "on eBay" in speech  # the page named the shop each row came from


def test_spoken_products_says_the_search_site_when_the_page_names_no_shop() -> None:
    page = "Coffee maker results Hamilton brewer $41.00 4.7 out of 5 612 reviews"
    assert spoken_products(parse_products(page), _request()) == (
        "Cheapest: Hamilton brewer, $41, rated 4.7 out of 5 on Google Shopping."
    )


def test_read_products_speaks_the_top_three() -> None:
    _fresh_session()
    result = read_products({"request": REQUEST})
    assert result.ok
    assert result.speech == spoken_products(parse_products(PAGE), _request())
    assert result.speech.startswith("Cheapest: Baseus 4-port USB-C charger")
    assert "rated 4.2 out of 5" in result.speech
    lines = result.detail.splitlines()
    assert len(lines) == 3
    assert "Baseus 4-port USB-C charger" in lines[0]
    assert "Anker 5-in-1 hub" in lines[1]
    assert "UGREEN 20W power bank" in lines[2]


def test_read_products_opens_the_url_build_url_produces() -> None:
    session = _fresh_session()
    req = parse_request(REQUEST, TODAY)
    assert req is not None
    url, _site = build_url(req, LOCATION)
    result = read_products({"request": REQUEST})
    assert result.ok
    assert ("ensure", {}) in session.calls
    assert ("goto", {"url": url}) in session.calls
    assert ("text", {"limit": 6000}) in session.calls
    assert url == "https://www.google.com/search?tbm=shop&q=4070%20Ti%20Super&tbs=p_ord:p"


def test_read_products_without_ratings_says_no_rating() -> None:
    session = _fresh_session()
    session._text = PAGE_NO_RATINGS
    result = read_products({"request": "the lowest price for a standing desk"})
    assert result.ok
    assert result.speech.startswith("Cheapest: Bambicore single arm, $18.99")
    assert "rated" not in result.speech
    assert len(result.detail.splitlines()) == 3


def test_read_products_falls_back_to_describe_when_the_page_will_not_parse() -> None:
    session = _fresh_session()
    session._text = PAGE_JUNK
    session._outline = []
    result = read_products({"request": REQUEST})
    assert result.ok
    assert result.speech == f"{describe(_request(), LOCATION)} The page itself still needs your eyes."
    assert result.detail == build_url(_request(), LOCATION)[0]


def test_read_products_reads_the_outline_when_the_text_holds_nothing() -> None:
    session = _fresh_session()
    session._text = "Shopping results"
    result = read_products({"request": REQUEST})
    assert result.ok
    assert result.speech.startswith("Cheapest: Baseus 4-port USB-C charger")


def test_read_products_reports_a_browser_failure() -> None:
    session = _fresh_session()
    session._goto_raises = True
    result = read_products({"request": REQUEST})
    assert not result.ok
    assert result.speech == "The browser did not open that page."
    assert result.detail == "navigation failed"


def test_read_products_without_a_location(monkeypatch) -> None:
    monkeypatch.setattr(deal_tools, "ConfigStore", lambda: _FakeStore(""))
    monkeypatch.setattr(deal_tools, "detect_location", lambda fetch=None: "")
    _fresh_session()
    result = read_products({"request": REQUEST})
    assert not result.ok
    assert result.speech == NO_LOCATION


def test_read_products_rejects_a_booking_request() -> None:
    _fresh_session()
    result = read_products({"request": "get me the cheapest ticket to Hawaii"})
    assert not result.ok
    assert result.speech == "That doesn't look like a product or price search."


def test_product_results_is_registered() -> None:
    assert "product_results" in default_registry().names()
