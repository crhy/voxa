"""Heads-up tools route and work end to end."""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime as dt_cls

from voxa.agent.headsup import forgotten, meeting_brief, suggest_next, today_summary
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.headsup import (
    _heads_up_handler,
    _meeting_brief_handler,
    _suggest_next_action_handler,
    _what_did_i_forget_handler,
)


class _FakeDT:
    @staticmethod
    def now():
        return dt_cls(2025, 10, 15, 10, 0)


class _FakeDate:
    @staticmethod
    def today():
        return date_cls(2025, 10, 15)


def test_today_summary_with_calendar_and_dates() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    calendar = "- 2025-10-15 15:00 | Dentist\n"
    dates_text = "- 2025-10-16 | deadline | Report\n"
    lists_text = "## To-do\n- [ ] Buy stamps\n- [x] Done item\n"
    result = today_summary(calendar, dates_text, lists_text, now)
    assert "Dentist" in result
    assert "Report" in result
    assert "1 thing open" in result


def test_today_summary_empty() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    result = today_summary("", "", "", now)
    assert result == "Nothing needs your attention today."


def test_forgotten_overdue_deadline() -> None:
    today = dt_cls(2025, 10, 15, 10, 0).date()
    dates_text = "- 2025-10-13 | deadline | Report\n"
    result = forgotten(dates_text, "", "", today)
    assert "The report was due 2 days ago." in result


def test_forgotten_open_to_dos() -> None:
    today = dt_cls(2025, 10, 15, 10, 0).date()
    lists_text = "## To-do\n- [ ] Buy stamps\n- [ ] Call the vet\n"
    result = forgotten("", lists_text, "", today)
    assert "Still open on your to-do list: Buy stamps and Call the vet." in result


def test_forgotten_empty() -> None:
    today = dt_cls(2025, 10, 15, 10, 0).date()
    result = forgotten("", "", "", today)
    assert result == "I don't see anything you have forgotten."


def test_suggest_next_appointment_within_two_hours() -> None:
    now = dt_cls(2025, 10, 15, 14, 0)
    calendar = "- 2025-10-15 15:00 | Dentist\n"
    result = suggest_next("", "", calendar, now)
    assert "Get ready: Dentist at 3 PM." in result


def test_suggest_next_overdue_date() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    dates_text = "- 2025-10-13 | deadline | Report\n"
    result = suggest_next(dates_text, "", "", now)
    assert "I would start with the report: it is overdue." in result


def test_suggest_next_first_open_to_do() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    lists_text = "## To-do\n- [ ] Buy stamps\n"
    result = suggest_next("", lists_text, "", now)
    assert "Next on your list: Buy stamps." in result


def test_suggest_next_all_caught_up() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    result = suggest_next("", "", "", now)
    assert result == "You are all caught up."


def test_meeting_brief_found() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    calendar = "- 2025-10-15 15:00 | Dentist\n"
    facts_text = "- 2025-10-10 | Dentist is on Main Street\n"
    lists_text = "## To-do\n- [ ] Call the dentist\n"
    result = meeting_brief(calendar, lists_text, facts_text, "dentist", now)
    assert "Dentist is today at 3 PM." in result
    assert "You told me: Dentist is on Main Street." in result
    assert "Still open: Call the dentist." in result


def test_meeting_brief_not_found() -> None:
    now = dt_cls(2025, 10, 15, 10, 0)
    result = meeting_brief("", "", "", "dentist", now)
    assert result == "I can't find that on your calendar."


def test_heads_up_phrases_route() -> None:
    assert route("what should I know") == ToolCall("heads_up", {})
    assert route("what should I know today") == ToolCall("heads_up", {})
    assert route("what's important today") == ToolCall("heads_up", {})
    assert route("anything I should know") == ToolCall("heads_up", {})
    assert route("give me a heads up") == ToolCall("heads_up", {})
    assert route("give me your hands up") == ToolCall("heads_up", {})


def test_forgotten_phrases_route() -> None:
    assert route("what am I forgetting") == ToolCall("what_did_i_forget", {})
    assert route("did I forget anything") == ToolCall("what_did_i_forget", {})
    assert route("have I forgotten anything") == ToolCall("what_did_i_forget", {})
    assert route("what have I missed") == ToolCall("what_did_i_forget", {})


def test_suggest_next_phrases_route() -> None:
    assert route("what do you recommend") == ToolCall("suggest_next_action", {})
    assert route("what do you suggest I do next") == ToolCall("suggest_next_action", {})
    assert route("what would you do next") == ToolCall("suggest_next_action", {})


def test_meeting_brief_phrases_route() -> None:
    assert route("brief me on my dentist appointment") == ToolCall(
        "meeting_brief", {"title": "dentist"}
    )
    assert route("prepare me for dentist meeting") == ToolCall(
        "meeting_brief", {"title": "dentist"}
    )
    assert route("tell me about the dentist appointment") == ToolCall(
        "meeting_brief", {"title": "dentist"}
    )


def test_general_tell_me_about_request_is_not_stolen_by_calendar() -> None:
    assert route("tell me about Saturn") is None
    assert route("brief me on the election") is None


def test_list_phrases_not_stolen() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})


def test_time_and_capital_and_app_not_stolen() -> None:
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_registry_has_headsup_tools() -> None:
    names = default_registry().names()
    for name in ("heads_up", "what_did_i_forget", "suggest_next_action", "meeting_brief"):
        assert name in names


def test_heads_up_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.headsup.datetime", _FakeDT)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2025-10-15 15:00 | Dentist\n")
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: "- 2025-10-16 | deadline | Report\n")
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: "## To-do\n- [ ] Buy stamps\n")
    result = _heads_up_handler({})
    assert result.ok
    assert "Dentist" in result.speech
    assert "Report" in result.speech
    assert "1 thing open" in result.speech


def test_what_did_i_forget_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.headsup.date", _FakeDate)
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: "- 2025-10-13 | deadline | Report\n")
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: "## To-do\n- [ ] Buy stamps\n")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "")
    result = _what_did_i_forget_handler({})
    assert result.ok
    assert "The report was due 2 days ago." in result.speech
    assert "Buy stamps" in result.speech


def test_suggest_next_action_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.headsup.datetime", _FakeDT)
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: "")
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: "## To-do\n- [ ] Buy stamps\n")
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "")
    result = _suggest_next_action_handler({})
    assert result.ok
    assert "Next on your list: Buy stamps." in result.speech


def test_meeting_brief_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.headsup.datetime", _FakeDT)
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: "- 2025-10-15 15:00 | Dentist\n")
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: "## To-do\n- [ ] Call the dentist\n")
    monkeypatch.setattr("voxa.agent.tools.memory_facts._read", lambda: "- 2025-10-10 | Dentist is on Main Street\n")
    result = _meeting_brief_handler({"title": "dentist"})
    assert result.ok
    assert "Dentist is today at 3 PM." in result.speech
    assert "You told me: Dentist is on Main Street." in result.speech
    assert "Still open: Call the dentist." in result.speech


def test_meeting_brief_handler_rejects_empty_title() -> None:
    result = _meeting_brief_handler({"title": "  "})
    assert not result.ok
