"""Recurring charges found in the spending log: detection, the due window, spoken answers, routing.

Issue #60. Detection runs on the rows the spending store keeps, because #45's statement
importer is not merged yet.
"""

from __future__ import annotations

from datetime import date, timedelta

from voxa.agent.bills import (
    Subscription,
    due_within,
    next_due,
    recurring_charges,
    spoken_due,
    spoken_subscriptions,
)
from voxa.agent.expenses import entries
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry

TODAY = date(2026, 10, 10)

# A sample spending log: a monthly utility bill, a monthly internet charge that wobbles
# a little, groceries that swing (not a bill), a movie seen twice in one month (not a
# bill) and a gas card paid every other month.
LOG = """- 2026-07-12 | 120.00 | home | Electricity
- 2026-08-12 | 120.00 | home | Electricity
- 2026-09-12 | 120.00 | home | Electricity
- 2026-10-12 | 120.00 | home | Electricity
- 2026-07-05 | 30.00 | home | Internet
- 2026-08-05 | 31.50 | home | Internet
- 2026-09-05 | 30.00 | home | Internet
- 2026-08-03 | 45.00 | food | Groceries
- 2026-09-04 | 12.00 | food | Groceries
- 2026-10-03 | 60.00 | food | Groceries
- 2026-10-05 | 12.00 | fun | Movie night
- 2026-10-22 | 12.00 | fun | Movie night
- 2026-04-15 | 40.00 | transport | Gas card
- 2026-06-15 | 40.00 | transport | Gas card
- 2026-08-15 | 40.00 | transport | Gas card
"""

# A charge due just over the month boundary: seen on the 3rd, today is the 27th.
BOUNDARY_TODAY = date(2026, 10, 27)
BOUNDARY_LOG = """- 2026-08-03 | 30.00 | home | Internet
- 2026-09-03 | 30.00 | home | Internet
- 2026-10-03 | 30.00 | home | Internet
- 2026-08-12 | 120.00 | home | Electricity
- 2026-09-12 | 120.00 | home | Electricity
- 2026-10-12 | 120.00 | home | Electricity
"""


def _by_label(subs: list[Subscription]) -> dict[str, Subscription]:
    return {sub.label: sub for sub in subs}


def test_a_monthly_bill_becomes_a_subscription() -> None:
    subs = recurring_charges(entries(LOG), TODAY)
    assert {"Electricity", "Internet", "Gas card"} == set(_by_label(subs))
    power = _by_label(subs)["Electricity"]
    assert power.amount == 120.0
    assert power.category == "home"
    assert power.cadence == "monthly"
    assert power.last_seen == date(2026, 10, 12)
    assert power.next_due == date(2026, 11, 12)


def test_a_charge_that_wobbles_within_ten_percent_is_still_one_bill() -> None:
    internet = _by_label(recurring_charges(entries(LOG), TODAY))["Internet"]
    assert internet.amount == 30.0
    assert internet.cadence == "monthly"


def test_a_one_off_trip_is_not_a_subscription() -> None:
    found = _by_label(recurring_charges(entries(LOG), TODAY))
    assert "Groceries" not in found  # one grocery trip a month, different amounts
    assert "Movie night" not in found  # twice in one month is not a bill


def test_a_charge_that_swings_more_than_ten_percent_is_not_a_bill() -> None:
    content = (
        "- 2026-07-01 | 120.00 | home | Water\n"
        "- 2026-08-01 | 120.00 | home | Water\n"
        "- 2026-09-01 | 145.00 | home | Water\n"
        "- 2026-10-01 | 120.00 | home | Water\n"
    )
    assert recurring_charges(entries(content), TODAY) == []


def test_three_different_months_are_needed_by_default() -> None:
    content = (
        "- 2026-09-12 | 120.00 | home | Electricity\n"
        "- 2026-10-12 | 120.00 | home | Electricity\n"
    )
    assert recurring_charges(entries(content), TODAY) == []
    assert len(recurring_charges(entries(content), TODAY, min_times=2)) == 1


def test_two_entries_in_the_same_month_count_once() -> None:
    content = (
        "- 2026-08-04 | 30.00 | home | Internet\n- 2026-08-20 | 30.00 | home | Internet\n"
        "- 2026-09-04 | 30.00 | home | Internet\n"
    )
    assert recurring_charges(entries(content), TODAY) == []  # two months, not three
    assert len(recurring_charges(entries(content), TODAY, min_times=2)) == 1  # August counted once


def test_a_steady_gap_sets_the_cadence() -> None:
    gas = _by_label(recurring_charges(entries(LOG), TODAY))["Gas card"]
    assert gas.cadence == "every 2 months"
    assert gas.next_due == date(2026, 10, 15)


def test_next_due_rolls_the_same_day_of_month_forward() -> None:
    assert next_due(date(2026, 9, 12), TODAY) == date(2026, 10, 12)
    assert next_due(date(2026, 10, 12), TODAY) == date(2026, 11, 12)
    assert next_due(date(2026, 1, 31), date(2026, 2, 1)) == date(2026, 2, 28)  # no 31st
    assert next_due(date(2026, 11, 30), date(2026, 11, 1)) == date(2026, 12, 30)


def test_next_due_never_returns_a_past_date() -> None:
    for offset in range(0, 400, 7):
        today = date(2026, 1, 1) + timedelta(days=offset)
        due = next_due(date(2025, 12, 15), today)
        assert due >= today, (today, due)


def test_the_due_window_crosses_a_month_boundary() -> None:
    subs = recurring_charges(entries(BOUNDARY_LOG), BOUNDARY_TODAY)
    assert _by_label(subs)["Internet"].next_due == date(2026, 11, 3)
    assert [sub.label for sub in due_within(subs, BOUNDARY_TODAY, days=7)] == ["Internet"]
    assert due_within(subs, BOUNDARY_TODAY, days=4) == []
    assert [sub.label for sub in due_within(subs, BOUNDARY_TODAY, days=20)] == ["Internet", "Electricity"]


def test_spoken_subscriptions_names_amounts_and_cadence() -> None:
    spoken = spoken_subscriptions(recurring_charges(entries(LOG), TODAY))
    assert spoken.startswith("You pay for 3 subscriptions:")
    assert "electricity 120 dollars every month" in spoken
    assert "internet 30 dollars every month" in spoken
    assert "gas card 40 dollars every 2 months" in spoken
    assert spoken_subscriptions([]) == "I have not found any subscriptions in your spending log."


def test_spoken_due_answers_the_question() -> None:
    subs = recurring_charges(entries(BOUNDARY_LOG), BOUNDARY_TODAY)
    assert spoken_due(subs, BOUNDARY_TODAY) == (
        "One bill is due this week: your internet bill on November 3rd for 30 dollars."
    )
    assert spoken_due(subs, BOUNDARY_TODAY, days=4) == "Nothing you pay for is due in the next 4 days."
    assert spoken_due([], BOUNDARY_TODAY) == "Nothing you pay for is due this week."


def test_phrases_route() -> None:
    assert route("what bills are due this week") == ToolCall("bills_due", {})
    assert route("what do I have to pay for this week") == ToolCall("bills_due", {})
    assert route("what subscriptions do I pay for") == ToolCall("subscriptions_list", {})
    assert route("remind me about my bills") == ToolCall("bills_remind", {})
    assert route("set up bill reminders") == ToolCall("bills_remind", {})


def test_bill_rules_do_not_steal_the_date_answers() -> None:
    assert route("what bills are due this month") == ToolCall("date_upcoming", {"kind": "bill"})
    assert route("what bills are coming up") == ToolCall("date_upcoming", {"kind": "bill"})
    assert route("what subscriptions do I have") == ToolCall("date_upcoming", {"kind": "subscription", "days": "365"})
    assert route("what's due this week") == ToolCall("date_upcoming", {})
    assert route("the electricity bill is due on the 15th of every month") == ToolCall(
        "date_remember", {"kind": "bill", "name": "electricity", "when": "the 15th of every month"}
    )


def test_bill_tools_are_registered() -> None:
    names = default_registry().names()
    for name in ("bills_due", "subscriptions_list", "bills_remind"):
        assert name in names
