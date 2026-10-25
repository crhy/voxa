"""Monthly bills, subscriptions and services route and work end to end."""

from __future__ import annotations

from datetime import date

from voxa.agent.dates import next_occurrence, parse_monthly, spoken_upcoming, spoken_when, upcoming
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.dates import _date_remember_handler, _date_upcoming_handler

TODAY = date(2026, 10, 10)


def test_parse_monthly_forms() -> None:
    assert parse_monthly("the 15th of every month") == 15
    assert parse_monthly("on the 15th of every month") == 15
    assert parse_monthly("every month on the 3rd") == 3
    assert parse_monthly("monthly on the 3rd") == 3
    assert parse_monthly("on the 15th each month") == 15
    assert parse_monthly("the 31st of every month") == 31
    assert parse_monthly("the 32nd of every month") is None
    assert parse_monthly("banana") is None


def test_monthly_next_occurrence() -> None:
    assert next_occurrence("15", TODAY) == date(2026, 10, 15)
    assert next_occurrence("03", TODAY) == date(2026, 11, 3)
    assert next_occurrence("31", date(2026, 1, 31)) == date(2026, 1, 31)
    assert next_occurrence("31", date(2026, 2, 1)) == date(2026, 2, 28)
    assert next_occurrence("31", date(2026, 2, 28)) == date(2026, 2, 28)


def test_spoken_when_new_kinds() -> None:
    assert spoken_when(("15", "bill", "Electricity"), TODAY) == (
        "The electricity bill is due on the 15th of each month; next in 5 days."
    )
    assert spoken_when(("03", "subscription", "Netflix"), TODAY) == (
        "Your Netflix subscription renews on the 3rd of each month; next in 24 days."
    )
    assert spoken_when(("2027-03-03", "service", "car service"), TODAY) == (
        "The car service is due on March 3rd, in 144 days."
    )


def test_upcoming_kind_filter() -> None:
    content = "- 15 | bill | Electricity\n- 03 | subscription | Netflix\n- 2027-03-03 | service | car service\n"
    assert upcoming(content, TODAY, days=30) == [
        (date(2026, 10, 15), "bill", "Electricity"),
        (date(2026, 11, 3), "subscription", "Netflix"),
    ]
    assert upcoming(content, TODAY, days=30, kind="bill") == [(date(2026, 10, 15), "bill", "Electricity")]
    assert upcoming(content, TODAY, days=365, kind="subscription") == [(date(2026, 11, 3), "subscription", "Netflix")]
    assert (
        spoken_upcoming(content, TODAY, days=30)
        == "Coming up: The electricity bill on Thursday, and Your Netflix subscription on November 3rd."
    )


def test_new_phrases_route() -> None:
    assert route("the electricity bill is due on the 15th of every month") == ToolCall(
        "date_remember", {"kind": "bill", "name": "electricity", "when": "the 15th of every month"}
    )
    assert route("netflix subscription renews on the 3rd of every month") == ToolCall(
        "date_remember", {"kind": "subscription", "name": "netflix", "when": "the 3rd of every month"}
    )
    assert route("the car service is due on march 3rd") == ToolCall(
        "date_remember", {"kind": "service", "name": "car service", "when": "march 3rd"}
    )
    assert route("what bills do i have coming up") == ToolCall("date_upcoming", {"kind": "bill"})
    assert route("what subscriptions do i have") == ToolCall("date_upcoming", {"kind": "subscription", "days": "365"})
    assert route("when is the electricity bill due") == ToolCall("date_when", {"name": "electricity"})
    assert route("when does netflix renew") == ToolCall("date_when", {"name": "netflix"})
    assert route("windows netflix renew") == ToolCall("date_when", {"name": "netflix"})
    assert route("when's the netflix subscription renew") == ToolCall("date_when", {"name": "netflix"})
    assert route("forget the electricity bill") == ToolCall("date_forget", {"name": "electricity"})


def test_cap61_misheard_service_due_phrases_route() -> None:
    # the small brain hears "the oil change is due on" as "the oil changes do 1"
    assert route("the oil changes do 1 March 3rd") == ToolCall(
        "date_remember", {"kind": "service", "name": "oil change", "when": "March 3rd"}
    )
    assert route("the oil change does 1 March 3rd") == ToolCall(
        "date_remember", {"kind": "service", "name": "oil change", "when": "March 3rd"}
    )
    assert route("the car service 2 March 3rd") == ToolCall(
        "date_remember", {"kind": "service", "name": "car service", "when": "March 3rd"}
    )


def test_new_rules_do_not_steal_other_phrases() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("the report is due Friday") == ToolCall("date_remember", {"kind": "deadline", "name": "report", "when": "Friday"})
    assert route("windows") is None or route("windows").tool != "date_when"


def test_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.dates._write", lambda content: store.__setitem__("content", content))

    bill = _date_remember_handler({"kind": "bill", "name": "electricity", "when": "on the 15th of every month"})
    assert bill.ok and bill.speech.startswith("Got it.")
    assert store["content"] == "- 15 | bill | Electricity\n"
    sub = _date_remember_handler({"kind": "subscription", "name": "netflix", "when": "on the 3rd of every month"})
    assert sub.ok
    assert store["content"] == "- 15 | bill | Electricity\n- 03 | subscription | Netflix\n"
    soon = _date_upcoming_handler({"kind": "bill"})
    assert soon.ok
    assert soon.speech.startswith("Coming up")


def test_registry_has_date_tools() -> None:
    names = default_registry().names()
    for name in ("date_remember", "date_when", "date_upcoming", "date_forget"):
        assert name in names
