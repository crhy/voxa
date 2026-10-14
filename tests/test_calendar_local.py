"""Voice commands for the local calendar route and work end to end."""

from __future__ import annotations

import datetime as _dt

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.calendar_local import (
    _calendar_add_handler,
    _calendar_cancel_handler,
    _calendar_day_handler,
    _calendar_next_handler,
)


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: _dt.date(2026, 10, 10))

    def _datetime_shim(*args):
        return _dt.datetime(*args)

    _datetime_shim.now = lambda: _dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


def test_calendar_phrases_route() -> None:
    assert route("add lunch with Sam to my calendar for tomorrow at noon") == ToolCall(
        "calendar_add", {"title": "lunch with Sam", "when": "tomorrow at noon"}
    )
    assert route("schedule the dentist for friday at 3 pm") == ToolCall(
        "calendar_add", {"title": "the dentist", "when": "friday at 3 pm"}
    )
    assert route("what's on my calendar tomorrow") == ToolCall("calendar_day", {"day": "tomorrow"})
    assert route("what's my next appointment") == ToolCall("calendar_next", {})
    assert route("cancel my dentist appointment") == ToolCall("calendar_cancel", {"title": "dentist"})
    assert route("cancel my lunch with Sam appointment") == ToolCall("calendar_cancel", {"title": "lunch with Sam"})


def test_calendar_rules_do_not_steal_other_intents() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
    assert route("cancel my reminders") == ToolCall("cancel_reminders", {})
    assert route("what's on my calendar") == ToolCall("calendar_day", {"day": ""})


def test_calendar_handlers_end_to_end(monkeypatch, tmp_path) -> None:
    _freeze_clock(monkeypatch)
    from voxa.agent.reminders import ReminderStore

    monkeypatch.setattr("voxa.agent.tools.calendar_local.ReminderStore", lambda: ReminderStore(tmp_path / "r.json"))
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _calendar_add_handler({"title": "lunch with Sam", "when": "tomorrow at noon"})
    assert added.ok and added.speech == "I put Lunch with Sam on your calendar for tomorrow at noon. I'll remind you 15 minutes before."
    assert _calendar_add_handler({"title": "the dentist", "when": "tomorrow at 3 pm"}).ok
    assert store["content"] == "- 2026-10-11 12:00 | Lunch with Sam\n- 2026-10-11 15:00 | The dentist\n"

    day = _calendar_day_handler({"day": "tomorrow"})
    assert day.ok and day.speech == "Tomorrow you have 2 things: Lunch with Sam at noon, and The dentist at 3 PM."
    nxt = _calendar_next_handler({})
    assert nxt.ok and nxt.speech == "Your next appointment is Lunch with Sam, tomorrow at noon."

    cancelled = _calendar_cancel_handler({"title": "dentist"})
    assert cancelled.ok and cancelled.speech == "I took The dentist off your calendar."
    assert store["content"] == "- 2026-10-11 12:00 | Lunch with Sam\n"


def test_calendar_add_rejects_unparsable_when(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: None)
    result = _calendar_add_handler({"title": "lunch", "when": "somewhen"})
    assert not result.ok and result.speech == "When is it?"


def test_calendar_cancel_reports_missing_title(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2026-10-11 12:00 | Lunch with Sam\n")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._write", lambda content: None)
    result = _calendar_cancel_handler({"title": "dentist"})
    assert not result.ok and result.speech == "I can't find dentist on your calendar."


def test_registry_has_calendar_tools() -> None:
    names = default_registry().names()
    for name in ("calendar_add", "calendar_day", "calendar_next", "calendar_cancel"):
        assert name in names
