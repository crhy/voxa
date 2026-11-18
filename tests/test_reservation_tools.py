"""crhy/voxa#47's first step: parse a table booking, open the restaurant's booking page,
pre-fill the fields that page offers. A test booking page stands in for the real one:
no browser, no network, no audio, and nothing is ever submitted."""

from __future__ import annotations

from datetime import date

from voxa.agent.cdp import CdpError
from voxa.agent.intents import ToolCall, route
from voxa.agent.reservations import booking_url, form_values, parse_reservation
from voxa.agent.tools import default_registry
from voxa.agent.tools.reservations import reservation_fill, reservation_plan
from voxa.agent.tools.web import set_session

TODAY = date(2026, 10, 7)
REQUEST = "book a table for four at Luigi's on Friday at seven"


class FakeSession:
    """A test booking page: canned outline, recorded calls, no browser."""

    def __init__(self):
        self.calls = []
        self._title = "Luigi's — book a table"
        self._url = "about:blank"
        self._text = "Reserve your table at Luigi's."
        self._outline = [
            {"n": 1, "kind": "link", "text": "Menu", "placeholder": "", "href": "/menu"},
            {"n": 2, "kind": "button", "text": "Book now", "placeholder": "", "href": ""},
            {"n": 3, "kind": "field", "text": "Party size", "placeholder": "Party size", "href": ""},
            {"n": 4, "kind": "field", "text": "Reservation date", "placeholder": "Reservation date", "href": ""},
            {"n": 5, "kind": "field", "text": "Time of day", "placeholder": "Time of day", "href": ""},
            {"n": 6, "kind": "field", "text": "Restaurant name", "placeholder": "Restaurant name", "href": ""},
        ]
        self._type_ok = True
        self._goto_raises = False

    def ensure(self):
        self.calls.append(("ensure", {}))

    def goto(self, url, wait=15.0):
        self.calls.append(("goto", {"url": url}))
        if self._goto_raises:
            raise CdpError("navigation failed")
        self._url = url

    def title(self):
        self.calls.append(("title", {}))
        return self._title

    def url(self):
        self.calls.append(("url", {}))
        return self._url

    def outline(self, limit=60):
        self.calls.append(("outline", {"limit": limit}))
        return self._outline

    def click(self, text):
        self.calls.append(("click", {"text": text}))
        return text

    def type_into(self, field, value, submit=False):
        self.calls.append(("type_into", {"field": field, "value": value, "submit": submit}))
        return self._type_ok

    def wait_for_text(self, text, timeout=15.0):
        self.calls.append(("wait_for_text", {"text": text}))
        return True


class _Frozen:
    """A fixed "today" so the spoken day resolves the same on any date."""

    @staticmethod
    def today():
        return TODAY


def _booking_page(monkeypatch, city: str = "") -> FakeSession:
    session = FakeSession()
    set_session(session)
    monkeypatch.setattr("voxa.agent.tools.reservations._location", lambda: city)
    monkeypatch.setattr("voxa.agent.tools.reservations.date", _Frozen)
    return session


def test_parse_takes_party_day_and_time_from_the_example_sentence():
    res = parse_reservation(REQUEST, TODAY)
    assert res is not None
    assert res.restaurant == "Luigi's"
    assert res.party_size == 4
    assert res.when == date(2026, 10, 9)
    assert res.at == "7:00 PM"
    assert res.site == ""


def test_form_values_are_the_labels_a_booking_page_expects():
    res = parse_reservation(REQUEST, TODAY)
    assert form_values(res) == [
        ("Party size", "4"),
        ("Date", "2026-10-09"),
        ("Time", "7:00 PM"),
        ("Restaurant", "Luigi's"),
    ]


def test_the_booking_page_url_is_the_one_goto_opens(monkeypatch):
    session = _booking_page(monkeypatch)
    result = reservation_fill({"request": REQUEST})
    assert result.ok
    assert ("goto", {"url": booking_url(parse_reservation(REQUEST, TODAY), "")}) in session.calls


def test_every_field_the_page_offers_is_typed_once_with_the_right_value(monkeypatch):
    session = _booking_page(monkeypatch)
    result = reservation_fill({"request": REQUEST})
    assert result.ok
    typed = [call[1] for call in session.calls if call[0] == "type_into"]
    assert typed == [
        {"field": "Party size", "value": "4", "submit": False},
        {"field": "Reservation date", "value": "2026-10-09", "submit": False},
        {"field": "Time of day", "value": "7:00 PM", "submit": False},
        {"field": "Restaurant name", "value": "Luigi's", "submit": False},
    ]
    assert result.speech.startswith("Filled Party size, Date, Time, and Restaurant on the web.")
    assert "Date -> Reservation date=2026-10-09" in result.detail


def test_a_field_the_page_does_not_offer_is_reported_not_guessed(monkeypatch):
    session = _booking_page(monkeypatch)
    session._outline = [item for item in session._outline if item["text"] != "Time of day"]
    result = reservation_fill({"request": REQUEST})
    assert result.ok
    typed = [call[1]["field"] for call in session.calls if call[0] == "type_into"]
    assert "Time of day" not in typed
    assert "It did not offer Time." in result.speech
    assert "Filled Party size" in result.speech


def test_nothing_is_ever_submitted_and_nothing_is_clicked(monkeypatch):
    session = _booking_page(monkeypatch)
    reservation_fill({"request": REQUEST})
    reservation_plan({"request": REQUEST})
    names = [call[0] for call in session.calls]
    assert "click" not in names
    assert "wait_for_text" not in names
    assert "search_site" not in names
    assert "back" not in names
    for call in session.calls:
        if call[0] == "type_into":
            assert call[1]["submit"] is False
    assert "Nothing was submitted." in reservation_fill({"request": REQUEST}).speech


def test_a_sentence_that_names_no_restaurant_is_refused(monkeypatch):
    _booking_page(monkeypatch)
    for spoken in ("book a table for four", "book a table", "reserve a table for two on Friday"):
        for handler in (reservation_plan, reservation_fill):
            result = handler({"request": spoken})
            assert not result.ok
            assert "table booking" in result.speech


def test_the_plan_says_the_booking_back_and_asks_for_what_is_missing(monkeypatch):
    _booking_page(monkeypatch)
    full = reservation_plan({"request": REQUEST})
    assert full.ok
    assert full.speech == "A table for 4 at Luigi's on Friday at 7:00 PM."

    partial = reservation_plan({"request": "reserve a table at Luigi's"})
    assert partial.ok
    assert partial.speech.startswith("A table at Luigi's.")
    for gap in ("the party size", "the day", "the time"):
        assert gap in partial.speech


def test_the_named_site_narrows_the_page_and_is_spoken(monkeypatch):
    _booking_page(monkeypatch)
    spoken = "book a table at Luigi's on OpenTable for tomorrow at 7 for four"
    res = parse_reservation(spoken, TODAY)
    assert res is not None
    assert res.site == "opentable"
    assert "site%3Aopentable.com" in booking_url(res, "")
    result = reservation_fill({"request": spoken})
    assert result.ok
    assert "on OpenTable" in result.speech


def test_the_page_that_stops_answering_is_reported_honestly(monkeypatch):
    session = _booking_page(monkeypatch)
    session._goto_raises = True
    result = reservation_fill({"request": REQUEST})
    assert not result.ok
    assert result.speech == "The browser did not open that page."


def test_reservation_plan_never_opens_the_browser(monkeypatch):
    session = _booking_page(monkeypatch)
    reservation_plan({"request": REQUEST})
    assert session.calls == []


def test_both_tools_are_registered_at_the_right_risk():
    registry = default_registry()
    assert {"reservation_plan", "reservation_fill"} <= set(registry.names())
    assert int(registry.get("reservation_plan").risk) == 0
    assert int(registry.get("reservation_fill").risk) == 1


def test_the_spoken_phrasings_reach_the_reservation_tools():
    assert route(REQUEST) == ToolCall("reservation_fill", {"request": REQUEST})
    assert route("Voxa, book a table for four at Luigi's on Friday at seven") == ToolCall(
        "reservation_fill", {"request": REQUEST}
    )
    assert route("reserve a table at Luigi's") == ToolCall(
        "reservation_plan", {"request": "reserve a table at Luigi's"}
    )
    assert route("book Luigi's for tonight") == ToolCall(
        "reservation_plan", {"request": "book Luigi's for tonight"}
    )


def test_other_bookings_keep_their_own_routes():
    assert route("book a hotel in Tokyo") == ToolCall("find_deal", {"request": "book a hotel in Tokyo"})
    assert route("book a flight to Cancun tomorrow") == ToolCall(
        "find_deal", {"request": "book a flight to Cancun tomorrow"}
    )
    assert route("book dentist on my calendar for friday at 3 pm") == ToolCall(
        "calendar_add", {"title": "dentist", "when": "friday at 3 pm"}
    )
    assert route("nearest pharmacy") == ToolCall("find_nearby", {"what": "pharmacy"})
    assert route("give me directions to the San Diego Zoo") == ToolCall(
        "directions", {"destination": "the San Diego Zoo"}
    )
