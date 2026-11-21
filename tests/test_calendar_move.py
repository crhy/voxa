"""Moving an appointment by voice: one line, sorted, and its reminder moves with it."""

from __future__ import annotations

import datetime as _dt

from voxa.agent.calendar_local import move
from voxa.agent.intents import ToolCall, route
from voxa.agent.reminders import ReminderStore
from voxa.agent.tools import default_registry
from voxa.agent.tools.calendar_local import _calendar_add_handler, _calendar_move_handler


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: _dt.date(2026, 10, 10))

    def _datetime_shim(*args):
        return _dt.datetime(*args)

    _datetime_shim.now = lambda: _dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


class _NullStore:
    def add(self, reminder) -> None:
        pass

    def remove(self, reminder_id: str) -> None:
        pass


def _ids(tmp_path) -> set[str]:
    return {reminder.id for reminder in ReminderStore(tmp_path / "r.json").upcoming()}


def test_calendar_move_phrases_route() -> None:
    assert route("move my dentist to thursday at 4") == ToolCall(
        "calendar_move", {"title": "dentist", "when": "thursday at 4"}
    )
    assert route("move the lunch appointment to tomorrow") == ToolCall(
        "calendar_move", {"title": "lunch appointment", "when": "tomorrow"}
    )
    assert route("reschedule dentist for friday at 3 pm") == ToolCall(
        "calendar_move", {"title": "dentist", "when": "friday at 3 pm"}
    )
    assert route("change my team call to tomorrow at 9 am") == ToolCall(
        "calendar_move", {"title": "team call", "when": "tomorrow at 9 am"}
    )


def test_calendar_move_rules_do_not_steal_other_intents() -> None:
    assert route("cancel my reminders") == ToolCall("cancel_reminders", {})
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("cancel my dentist appointment") == ToolCall("calendar_cancel", {"title": "dentist"})
    assert route("move the window from left to right") is None
    assert route("move the file from documents to notes") is None
    assert route("change to Windows 7 dark theme").tool == "change_theme"


def test_move_pure() -> None:
    content = "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-11 15:00 | The dentist\n"
    new_content, moved, old_day, old_time = move(content, "dentist", _dt.date(2026, 10, 15), (16, 0))
    assert moved == "The dentist"
    assert (old_day, old_time) == (_dt.date(2026, 10, 11), (15, 0))
    assert new_content == "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-15 16:00 | The dentist\n"
    assert new_content.count("dentist") == 1
    assert move(content, "gym", _dt.date(2026, 10, 15), (16, 0)) == (content, None, None, None)


def test_move_handler_moves_the_reminder(monkeypatch, tmp_path) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.ReminderStore", lambda: ReminderStore(tmp_path / "r.json"))
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )

    _calendar_add_handler({"title": "lunch with Sam", "when": "tomorrow at noon"})
    _calendar_add_handler({"title": "the dentist", "when": "tomorrow at 3 pm"})
    assert store["content"] == "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-11 15:00 | The dentist\n"
    old_due = _dt.datetime(2026, 10, 11, 14, 45)
    assert f"cal{int(old_due.timestamp())}" in _ids(tmp_path)

    moved = _calendar_move_handler({"title": "dentist", "when": "thursday at 4 pm"})
    assert moved.ok
    assert moved.speech == (
        "I moved The dentist from tomorrow at 3 PM to thursday at 4 PM."
        " I'll remind you 15 minutes before."
    )
    assert store["content"] == "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-15 16:00 | The dentist\n"
    new_due = _dt.datetime(2026, 10, 15, 15, 45)
    assert _ids(tmp_path) == {
        f"cal{int(_dt.datetime(2026, 10, 11, 11, 45).timestamp())}",
        f"cal{int(new_due.timestamp())}",
    }
    assert ReminderStore(tmp_path / "r.json").upcoming()[1].text == "The dentist is in 15 minutes"


def test_move_refuses_an_unknown_title(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2026-10-11 12:00 | Lunch with Sam\n")
    written = []
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: written.append(content))
    result = _calendar_move_handler({"title": "gym", "when": "tomorrow at noon"})
    assert not result.ok and result.speech == "I can't find gym on your calendar."
    assert written == []


def test_move_keeps_one_line_and_the_no_time_convention(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.ReminderStore", lambda: _NullStore())
    store = {"content": "- 2026-10-11 --:-- | Errand\n- 2026-10-12 09:00 | Dentist\n"}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: store.__setitem__("content", content))
    result = _calendar_move_handler({"title": "errand", "when": "thursday"})
    assert result.ok
    assert result.speech == "I moved Errand from tomorrow to thursday."
    assert store["content"] == "- 2026-10-12 09:00 | Dentist\n- 2026-10-15 --:-- | Errand\n"


def test_move_notes_a_clash(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.ReminderStore", lambda: _NullStore())
    store = {"content": "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-11 15:00 | The dentist\n"}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: store.__setitem__("content", content))
    result = _calendar_move_handler({"title": "dentist", "when": "tomorrow at 12:30 pm"})
    assert result.ok
    assert result.speech == (
        "I moved The dentist from tomorrow at 3 PM to tomorrow at 12:30 PM."
        " Note: that is close to Lunch with Sam at noon."
        " I'll remind you 15 minutes before."
    )
    assert store["content"] == "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-11 12:30 | The dentist\n"


def test_move_asks_when_the_new_time_is_unparsable(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2026-10-11 12:00 | Lunch with Sam\n")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: None)
    result = _calendar_move_handler({"title": "lunch", "when": "somewhen"})
    assert not result.ok and result.speech == "When should it be instead?"


def test_registry_has_calendar_move() -> None:
    assert "calendar_move" in default_registry().names()
