"""Spending by voice: pure module, routing and end-to-end handlers."""

from __future__ import annotations

from datetime import date

from voxa.agent.expenses import (
    add,
    category_for,
    entries,
    parse_amount,
    period_start,
    remove_last,
    spoken_breakdown,
    spoken_money,
    spoken_total,
    total,
)
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.expenses import (
    _expense_add_handler,
    _expense_breakdown_handler,
    _expense_total_handler,
    _expense_undo_handler,
)

TODAY = date(2026, 10, 10)


def test_parse_amount() -> None:
    assert parse_amount("12 dollars") == 12.0
    assert parse_amount("$12.50") == 12.5
    assert parse_amount("12.50") == 12.5
    assert parse_amount("twelve dollars") == 12.0
    assert parse_amount("12 dollars and 50 cents") == 12.5
    assert parse_amount("a hundred bucks") == 100.0
    assert parse_amount("45 bucks") == 45.0
    assert parse_amount("no amount here") is None


def test_category_for() -> None:
    assert category_for("Lunch") == "food"
    assert category_for("gas") == "transport"
    assert category_for("movie night") == "fun"
    assert category_for("unknown thing") == "other"


def test_add_and_entries() -> None:
    content = add("", TODAY, 12.5, "lunch")
    assert content == "- 2026-10-10 | 12.50 | food | Lunch\n"
    assert entries(content) == [(date(2026, 10, 10), 12.5, "food", "Lunch")]


def test_remove_last() -> None:
    content = "- 2026-10-10 | 12.50 | food | Lunch\n"
    new, removed = remove_last(content)
    assert new == ""
    assert removed == (date(2026, 10, 10), 12.5, "food", "Lunch")
    assert remove_last("") == ("", None)


def test_total() -> None:
    items = [
        (date(2026, 10, 10), 12.5, "food", "Lunch"),
        (date(2026, 10, 5), 30.0, "transport", "Gas"),
    ]
    assert total(items, date(2026, 10, 5)) == 42.5
    assert total(items, date(2026, 10, 5), "food") == 12.5
    assert total(items, date(2026, 10, 11)) == 0.0


def test_period_start() -> None:
    assert period_start("today", TODAY) == (date(2026, 10, 10), "today")
    assert period_start("this week", TODAY) == (date(2026, 10, 5), "this week")
    assert period_start("this month", TODAY) == (date(2026, 10, 1), "this month")
    assert period_start("this year", TODAY) == (date(2026, 1, 1), "this year")
    assert period_start("", TODAY) == (date(2026, 10, 1), "this month")


def test_spoken_money() -> None:
    assert spoken_money(12.5) == "12 dollars and 50 cents"
    assert spoken_money(12.0) == "12 dollars"
    assert spoken_money(1.0) == "1 dollar"


def test_spoken_total() -> None:
    items = [
        (date(2026, 10, 10), 12.5, "food", "Lunch"),
        (date(2026, 10, 5), 30.0, "transport", "Gas"),
    ]
    assert spoken_total(items, TODAY, "this month", "food") == "You have spent 12 dollars and 50 cents on food this month."
    assert spoken_total(items, TODAY, "this week", None) == "You have spent 42 dollars and 50 cents this week."
    assert spoken_total([], TODAY, "this week", None) == "You have not spent anything this week."


def test_spoken_breakdown() -> None:
    items = [
        (date(2026, 10, 10), 45.0, "food", "Lunch"),
        (date(2026, 10, 5), 30.0, "transport", "Gas"),
    ]
    assert spoken_breakdown(items, TODAY, "this month") == "This month: food 45 dollars, transport 30 dollars."
    assert spoken_breakdown([], TODAY, "this month") == "You have not spent anything this month."


def test_phrases_route() -> None:
    assert route("i spent 12 dollars on lunch") == ToolCall("expense_add", {"amount": "12 dollars", "what": "lunch"})
    assert route("how much have i spent this week") == ToolCall("expense_total", {"period": "this week", "category": ""})
    assert route("show my spending this month") == ToolCall("expense_breakdown", {"period": "this month"})
    assert route("remove the last expense") == ToolCall("expense_undo", {})


def test_guard_phrases_still_route() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.expenses._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.expenses._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _expense_add_handler({"amount": "12 dollars and 50 cents", "what": "lunch"})
    assert added.ok and added.speech == "Noted: 12 dollars and 50 cents for lunch, under food."
    assert "| food | Lunch" in store["content"]

    total_result = _expense_total_handler({"period": "this month", "category": "food"})
    assert total_result.speech == "You have spent 12 dollars and 50 cents on food this month."

    breakdown = _expense_breakdown_handler({"period": "this month"})
    assert breakdown.speech == "This month: food 12 dollars and 50 cents."

    undone = _expense_undo_handler({})
    assert undone.ok and undone.speech == "I removed the last expense: 12 dollars and 50 cents for lunch."
    assert store["content"] == ""

    again = _expense_undo_handler({})
    assert not again.ok and again.speech == "There is no expense to remove."


def test_add_rejects_unparsable_amount() -> None:
    result = _expense_add_handler({"amount": "a lot", "what": "lunch"})
    assert not result.ok and result.speech == "How much was it?"


def test_registry_has_expense_tools() -> None:
    names = default_registry().names()
    for name in ("expense_add", "expense_total", "expense_breakdown", "expense_undo"):
        assert name in names
