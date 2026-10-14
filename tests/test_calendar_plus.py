"""C13: conflicts, auto-reminders, and recurring weekly calendar entries."""

from __future__ import annotations

import datetime as _dt

from voxa.agent.calendar_local import add_weekly, conflicts, entries_on, weekly_entries
from voxa.agent.intents import ToolCall, route
from voxa.agent.reminders import Reminder, ReminderStore
from voxa.agent.tools.calendar_local import (
    _calendar_add_handler,
    _calendar_add_weekly_handler,
    _calendar_cancel_handler,
    _calendar_conflicts_handler,
)


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: _dt.date(2026, 10, 10))

    def _datetime_shim(*args):
        return _dt.datetime(*args)

    _datetime_shim.now = lambda: _dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


def test_conflicts_pure() -> None:
    items = [
        (_dt.date(2026, 10, 11), (15, 0), "Dentist"),
        (_dt.date(2026, 10, 11), (15, 30), "Team call"),
        (_dt.date(2026, 10, 11), (18, 0), "Dinner"),
        (_dt.date(2026, 10, 12), (15, 0), "Other day"),
    ]
    assert conflicts(items, _dt.date(2026, 10, 11), (15, 15)) == ["Dentist", "Team call"]
    assert conflicts(items, _dt.date(2026, 10, 11), (17, 0)) == ["Dinner"]
    assert conflicts(items, _dt.date(2026, 10, 11), None) == []


def test_weekly_lines_round_trip() -> None:
    content = "- 2026-10-11 12:00 | Lunch with Sam\n"
    with_weekly = add_weekly(content, 0, (9, 0), "team call")
    assert with_weekly == "- 2026-10-11 12:00 | Lunch with Sam\n- every Monday 09:00 | Team call\n"
    assert weekly_entries(with_weekly) == [(0, (9, 0), "Team call")]
    monday = _dt.date(2026, 10, 12)
    assert entries_on(with_weekly, monday) == [(monday, (9, 0), "Team call")]
    assert entries_on(with_weekly, _dt.date(2026, 10, 11)) == [
        (_dt.date(2026, 10, 11), (12, 0), "Lunch with Sam")
    ]


def test_weekly_phrases_route() -> None:
    assert route("add team call every monday at 9 am") == ToolCall(
        "calendar_add_weekly", {"title": "team call", "weekday": "monday", "time": "9 am"}
    )
    assert route("i have lunch every friday at noon") == ToolCall(
        "calendar_add_weekly", {"title": "lunch", "weekday": "friday", "time": "noon"}
    )
    assert route("do i have any conflicts this week") == ToolCall("calendar_conflicts", {})
    assert route("are there any clashes on my calendar") == ToolCall("calendar_conflicts", {})
    assert route("check my calendar for clashes") == ToolCall("calendar_conflicts", {})


def test_weekly_and_conflict_guards_stay_put() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
    assert route("add lunch with Sam to my calendar for tomorrow at noon") == ToolCall(
        "calendar_add", {"title": "lunch with Sam", "when": "tomorrow at noon"}
    )


def test_add_reminder_and_cancel_removes_it(monkeypatch, tmp_path) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local.ReminderStore",
        lambda: ReminderStore(tmp_path / "r.json"),
    )
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _calendar_add_handler({"title": "lunch with Sam", "when": "tomorrow at noon"})
    assert added.ok and "I'll remind you 15 minutes before." in added.speech
    due = _dt.datetime(2026, 10, 11, 11, 45)
    reminder_id = f"cal{int(due.timestamp())}"
    assert ReminderStore(tmp_path / "r.json").upcoming() == [
        Reminder(id=reminder_id, due=due, text="lunch with Sam is in 15 minutes", kind="reminder")
    ]

    cancelled = _calendar_cancel_handler({"title": "lunch with Sam"})
    assert cancelled.ok
    assert ReminderStore(tmp_path / "r.json").upcoming() == []


def test_add_skips_past_reminders(monkeypatch, tmp_path) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local.ReminderStore",
        lambda: ReminderStore(tmp_path / "r.json"),
    )
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: None)
    result = _calendar_add_handler({"title": "coffee", "when": "today at 8 am"})
    assert result.ok and "I'll remind you 15 minutes before." not in result.speech
    assert ReminderStore(tmp_path / "r.json").upcoming() == []


def test_conflict_note_on_add(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    store = {"content": "- 2026-10-11 15:00 | Dentist\n"}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )
    result = _calendar_add_handler({"title": "team call", "when": "tomorrow at 3:30 pm"})
    assert result.ok
    assert result.speech == (
        "I put Team call on your calendar for tomorrow at 3:30 PM."
        " Note: that is close to Dentist at 3 PM."
        " I'll remind you 15 minutes before."
    )
    assert store["content"] == "- 2026-10-11 15:00 | Dentist\n- 2026-10-11 15:30 | Team call\n"


def test_conflicts_handler(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._read",
        lambda: "- 2026-10-11 15:00 | Dentist\n- 2026-10-11 15:30 | Team call\n",
    )
    clash = _calendar_conflicts_handler({})
    assert clash.ok and "Dentist at 3 PM is close to Team call at 3:30 PM." in clash.speech
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._read",
        lambda: "- 2026-10-11 15:00 | Dentist\n- 2026-10-11 18:00 | Dinner\n",
    )
    clear = _calendar_conflicts_handler({})
    assert clear.ok and clear.speech == "I see no clashes in the next 7 days."


def test_add_weekly_handler(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )
    result = _calendar_add_weekly_handler({"title": "team call", "weekday": "monday", "time": "9 am"})
    assert result.ok
    assert result.speech == "I put Team call on your calendar every Monday at 9 AM."
    assert store["content"] == "- every Monday 09:00 | Team call\n"
    missing = _calendar_add_weekly_handler({"title": "team call", "weekday": "someday", "time": "9 am"})
    assert not missing.ok and missing.speech == "Which weekday?"
