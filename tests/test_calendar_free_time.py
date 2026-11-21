"""Free time on the calendar: the first gap of the length you asked for, spoken."""

from __future__ import annotations

import datetime as _dt

from voxa.agent.calendar_local import free_slot
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.calendar_local import _calendar_free_handler

TODAY = _dt.date(2026, 10, 10)


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: TODAY)

    def _datetime_shim(*args):
        return _dt.datetime(*args)

    _datetime_shim.now = lambda: _dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


def test_free_slot_skips_a_busy_morning() -> None:
    content = "- 2026-10-11 09:00 | Gym\n- 2026-10-11 10:00 | Lunch\n- 2026-10-11 11:00 | Dentist\n"
    assert free_slot(content, _dt.date(2026, 10, 11), 60) == (12, 0)
    assert free_slot(content, _dt.date(2026, 10, 11), 30) == (12, 0)
    assert free_slot(content, _dt.date(2026, 10, 11), 90) == (12, 0)


def test_free_slot_honours_a_weekly_entry() -> None:
    content = "- every Sunday 09:00 | Team call\n"
    assert free_slot(content, _dt.date(2026, 10, 11), 60) == (10, 0)
    assert free_slot(content, _dt.date(2026, 10, 12), 60) == (8, 0)


def test_free_slot_returns_none_on_a_full_day() -> None:
    content = "".join(f"- 2026-10-11 {hour:02d}:00 | Chore {hour}\n" for hour in range(8, 22))
    assert free_slot(content, _dt.date(2026, 10, 11), 60) is None
    assert free_slot(content, _dt.date(2026, 10, 11), 30) is None


def test_free_slot_boundaries() -> None:
    assert free_slot("", _dt.date(2026, 10, 11), 60) == (8, 0)
    assert free_slot("- 2026-10-11 09:00 | Gym\n", _dt.date(2026, 10, 11), 60, (9, 0)) == (10, 0)
    assert free_slot("- 2026-10-11 --:-- | Errand\n", _dt.date(2026, 10, 11), 60) == (8, 0)
    assert free_slot("- 2026-10-11 21:00 | Late\n", _dt.date(2026, 10, 11), 60) == (8, 0)
    assert free_slot("- 2026-10-11 21:00 | Late\n", _dt.date(2026, 10, 11), 60, (21, 0)) is None


def test_free_phrases_route() -> None:
    assert route("when am I free tomorrow") == ToolCall("calendar_free", {"day": "tomorrow", "minutes": ""})
    assert route("find free time this afternoon") == ToolCall("calendar_free", {"day": "", "minutes": ""})
    assert route("when do I have an hour free") == ToolCall("calendar_free", {"day": "", "minutes": "an hour"})
    assert route("when am I free for 90 minutes tomorrow") == ToolCall(
        "calendar_free", {"day": "tomorrow", "minutes": "90 minutes"}
    )
    assert route("show me free time on friday") == ToolCall("calendar_free", {"day": "friday", "minutes": ""})


def test_free_rules_do_not_steal_other_intents() -> None:
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("cancel my reminders") == ToolCall("cancel_reminders", {})
    assert route("when is the report due") == ToolCall("date_when", {"name": "report"})
    assert route("when does my passport expire") == ToolCall("travel_doc_when", {"document": "passport"})
    assert route("what's on my calendar tomorrow") == ToolCall("calendar_day", {"day": "tomorrow"})
    assert route("what's my next appointment") == ToolCall("calendar_next", {})


def test_free_handler_names_the_gap(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._read",
        lambda: "- 2026-10-11 09:00 | Gym\n- 2026-10-11 10:00 | Lunch\n- 2026-10-11 11:00 | Dentist\n",
    )
    result = _calendar_free_handler({"day": "tomorrow", "minutes": "an hour"})
    assert result.ok and result.speech == "You have an hour free from noon to 1 PM tomorrow."


def test_free_handler_counts_a_weekly_entry(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- every Sunday 09:00 | Team call\n")
    result = _calendar_free_handler({"day": "tomorrow", "minutes": "an hour"})
    assert result.ok and result.speech == "You have an hour free from 10 AM to 11 AM tomorrow."


def test_free_handler_says_when_the_day_is_full(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    content = "".join(f"- 2026-10-11 {hour:02d}:00 | Chore {hour}\n" for hour in range(8, 22))
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: content)
    result = _calendar_free_handler({"day": "tomorrow", "minutes": "an hour"})
    assert result.ok and result.speech == "You have no free stretch of an hour tomorrow."


def test_free_handler_defaults_to_an_hour(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2026-10-10 09:00 | Gym\n")
    result = _calendar_free_handler({"day": "today", "minutes": ""})
    assert result.ok and result.speech == "You have an hour free from 10 AM to 11 AM today."


def test_free_handler_asks_for_an_unparsable_amount(monkeypatch) -> None:
    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "")
    result = _calendar_free_handler({"day": "tomorrow", "minutes": "a while"})
    assert not result.ok and result.speech == "How much free time am I looking for?"


def test_registry_has_calendar_free() -> None:
    assert "calendar_free" in default_registry().names()
