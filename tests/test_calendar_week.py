"""C65: the week ahead, per person."""

from __future__ import annotations

import datetime as _dt

from voxa.agent.calendar_local import person_of, spoken_count, spoken_week, week
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.calendar_local import (
    _calendar_busy_handler,
    _calendar_week_handler,
)

TODAY = _dt.date(2026, 10, 10)


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: TODAY)

    def _datetime_shim(*args):
        return _dt.datetime(*args)

    _datetime_shim.now = lambda: _dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


def test_person_of() -> None:
    assert person_of("Emma's soccer practice") == "Emma"
    assert person_of("dentist") is None
    assert person_of("Sam's lunch with team") == "Sam"


def test_week_pure() -> None:
    items = [
        (_dt.date(2026, 10, 10), (9, 0), "Emma's soccer practice"),
        (_dt.date(2026, 10, 12), (15, 0), "dentist"),
        (_dt.date(2026, 10, 16), (18, 0), "Sam's dinner"),
        (_dt.date(2026, 10, 17), (10, 0), "out of range"),
    ]
    result = week(items, TODAY)
    assert result == [
        (_dt.date(2026, 10, 10), (9, 0), "Emma's soccer practice"),
        (_dt.date(2026, 10, 12), (15, 0), "dentist"),
        (_dt.date(2026, 10, 16), (18, 0), "Sam's dinner"),
    ]


def test_week_from_content() -> None:
    content = "- 2026-10-11 12:00 | Lunch with Sam\n- every Monday 09:00 | Team call\n"
    result = week(content, TODAY)
    assert result == [
        (_dt.date(2026, 10, 11), (12, 0), "Lunch with Sam"),
        (_dt.date(2026, 10, 12), (9, 0), "Team call"),
    ]


def test_spoken_week_all() -> None:
    entries_list = [
        (_dt.date(2026, 10, 10), (9, 0), "Emma's soccer practice"),
        (_dt.date(2026, 10, 12), (15, 0), "dentist"),
    ]
    text = spoken_week(entries_list, TODAY)
    assert "This week:" in text
    assert "today" in text
    assert "Emma's soccer practice at 9 AM" in text


def test_spoken_week_person() -> None:
    entries_list = [
        (_dt.date(2026, 10, 10), (9, 0), "Emma's soccer practice"),
        (_dt.date(2026, 10, 12), (15, 0), "dentist"),
    ]
    text = spoken_week(entries_list, TODAY, person="Emma")
    assert "Emma has 1 thing this week:" in text
    assert "dentist" not in text


def test_spoken_week_empty_person() -> None:
    entries_list = [
        (_dt.date(2026, 10, 12), (15, 0), "dentist"),
    ]
    text = spoken_week(entries_list, TODAY, person="Emma")
    assert text == "Emma has nothing this week."


def test_spoken_count() -> None:
    entries_list = [
        (_dt.date(2026, 10, 10), (9, 0), "Emma's soccer practice"),
        (_dt.date(2026, 10, 10), (15, 0), "dentist"),
        (_dt.date(2026, 10, 12), (18, 0), "Sam's dinner"),
    ]
    text = spoken_count(entries_list)
    assert "3 appointments" in text
    assert "busiest day is Saturday" in text


def test_spoken_count_empty() -> None:
    assert spoken_count([]) == "You have nothing this week."


def test_week_phrases_route() -> None:
    assert route("what's on this week") == ToolCall("calendar_week", {"person": ""})
    assert route("what does my week look like") == ToolCall("calendar_week", {"person": ""})
    assert route("what's the family schedule") == ToolCall("calendar_week", {"person": ""})
    assert route("what does Emma have this week") == ToolCall("calendar_week", {"person": "Emma"})
    assert route("what's Sam doing this week") == ToolCall("calendar_week", {"person": "Sam"})
    assert route("what's on Emma's calendar") == ToolCall("calendar_week", {"person": "Emma"})
    assert route("how busy am i") == ToolCall("calendar_busy", {})
    assert route("how many appointments do i have this week") == ToolCall("calendar_busy", {})


def test_book_plan_verbs_route() -> None:
    assert route("book dentist on my calendar for friday at 3 pm") == ToolCall(
        "calendar_add", {"title": "dentist", "when": "friday at 3 pm"}
    )
    assert route("plan lunch with Sam on my calendar tomorrow at noon") == ToolCall(
        "calendar_add", {"title": "lunch with Sam", "when": "tomorrow at noon"}
    )


def test_week_guard_phrases() -> None:
    assert route("what's on my calendar tomorrow") == ToolCall("calendar_day", {"day": "tomorrow"})
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"


def test_calendar_week_handler(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._read",
        lambda: "- 2026-10-10 09:00 | Emma's soccer practice\n- 2026-10-12 15:00 | dentist\n",
    )
    result = _calendar_week_handler({"person": "Emma"})
    assert result.ok
    assert "Emma has 1 thing this week:" in result.speech


def test_calendar_busy_handler(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._read",
        lambda: "- 2026-10-10 09:00 | Emma's soccer practice\n- 2026-10-10 15:00 | dentist\n- 2026-10-12 18:00 | Sam's dinner\n",
    )
    result = _calendar_busy_handler({})
    assert result.ok
    assert "3 appointments" in result.speech
    assert "busiest day is Saturday" in result.speech
