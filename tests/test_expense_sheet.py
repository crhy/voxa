"""The spending log written out as a CSV spreadsheet and opened."""

from __future__ import annotations

from datetime import date

from voxa.agent.expenses import to_csv
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.expenses import _expense_spreadsheet_handler

TODAY = date(2026, 10, 10)


def test_to_csv() -> None:
    items = [
        (date(2026, 10, 10), 12.5, "food", "Lunch"),
        (date(2026, 10, 5), 30.0, "transport", "Gas"),
    ]
    assert to_csv(items) == (
        "Date,Amount,Category,What\n"
        "2026-10-10,12.50,food,Lunch\n"
        "2026-10-05,30.00,transport,Gas\n"
    )
    assert to_csv([]) == "Date,Amount,Category,What\n"


def test_phrases_route() -> None:
    assert route("make a spreadsheet of my expenses") == ToolCall("expense_spreadsheet", {})
    assert route("create a sheet with my spending") == ToolCall("expense_spreadsheet", {})
    assert route("export my expenses") == ToolCall("expense_spreadsheet", {})
    assert route("export my spending to a spreadsheet") == ToolCall("expense_spreadsheet", {})


def test_guard_phrases_still_route() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_handler_end_to_end(monkeypatch) -> None:
    store = {"content": "- 2026-10-10 | 12.50 | food | Lunch\n- 2026-10-05 | 30.00 | transport | Gas\n"}
    monkeypatch.setattr("voxa.agent.tools.expenses._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.expenses._write_csv", lambda content: None)
    monkeypatch.setattr("voxa.agent.tools.expenses.spawn", lambda command: True)
    monkeypatch.setattr("voxa.agent.tools.expenses.filemanage._folder", lambda name: "/tmp/docs")
    result = _expense_spreadsheet_handler({})
    assert result.ok
    assert result.speech == "I made a spreadsheet of 2 expenses and opened it."


def test_handler_with_no_expenses(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.expenses._read", lambda: "")
    result = _expense_spreadsheet_handler({})
    assert not result.ok
    assert result.speech == "There are no expenses to put in a spreadsheet."


def test_registry_has_spreadsheet_tool() -> None:
    assert "expense_spreadsheet" in default_registry().names()
