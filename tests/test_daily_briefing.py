import json
from datetime import date, datetime
from types import SimpleNamespace

from voxa.agent import briefing, intents
from voxa.agent.result import ToolResult
from voxa.agent.tools import briefing as tool

WEATHER = json.dumps({
    "current_condition": [{"temp_F": "76", "temp_C": "25", "weatherDesc": [{"value": "Sunny"}]}],
    "weather": [{"maxtempF": "77", "mintempF": "63", "maxtempC": "25", "mintempC": "17"}],
    "nearest_area": [{"areaName": [{"value": "Oceanside"}], "country": [{"value": "United States of America"}]}],
})


def test_parts():
    assert briefing.parse_weather(WEATHER) == "In Oceanside it is sunny and 76 degrees, with a high of 77 and a low of 63."
    assert "25 degrees" in briefing.parse_weather(WEATHER, fahrenheit=False)
    assert briefing.parse_weather("nope") == "" and briefing.uses_fahrenheit(WEATHER)
    now = datetime(2026, 10, 7, 9, 0)
    items = [SimpleNamespace(text="call Bob", due=datetime(2026, 10, 7, 15, 0)),
             SimpleNamespace(text="old", due=datetime(2026, 10, 7, 8, 0)),
             SimpleNamespace(text="tomorrow", due=datetime(2026, 10, 8, 9, 0))]
    assert briefing.reminders_today(items, now) == "You have 1 reminder today: call Bob at 3:00 PM."
    assert briefing.reminders_today([], now) == ""
    assert briefing.greeting_for(9) == "Good morning." and briefing.greeting_for(20) == "Good evening."


def test_tool_survives_missing_parts(monkeypatch):
    monkeypatch.setattr(tool.system, "tell_date", lambda args: ToolResult.success("It's Wednesday, the 7th of October 2026."))
    monkeypatch.setattr(tool, "ReminderStore", lambda: SimpleNamespace(upcoming=lambda: []))
    monkeypatch.setattr(tool, "get_location", lambda: "")
    now = datetime(2026, 10, 7, 9, 0)
    full = tool.daily_briefing({}, now=now, fetch=lambda url: WEATHER)
    assert full.speech == ("Good morning. It's Wednesday, the 7th of October 2026. In Oceanside it is sunny and 76 degrees, "
                           "with a high of 77 and a low of 63. Nothing is on your reminder list today.")

    def offline(url):
        raise OSError("no network")

    partial = tool.daily_briefing({}, now=now, fetch=offline)
    assert partial.ok and "Oceanside" not in partial.speech and partial.speech.startswith("Good morning. It's Wednesday")


def test_money_line():
    today = date(2026, 10, 7)
    items = [(date(2026, 10, 3), 12.5, "food", "Lunch"), (date(2026, 9, 2), 40.0, "food", "Groceries")]
    assert briefing.money_line(items, {}, today) == "You have spent 12 dollars and 50 cents this month."
    assert briefing.money_line(items, {"food": 100.0}, today) == "You have spent 12 dollars and 50 cents this month."
    assert briefing.money_line(items, {"food": 10.0}, today) == (
        "You have spent 12 dollars and 50 cents this month. "
        "You are 2 dollars and 50 cents over your 10 dollar food budget this month."
    )
    assert briefing.money_line([], {}, today) == ""
    assert briefing.money_line(items, {}, date(2026, 11, 7)) == ""


def _quiet_briefing_parts(monkeypatch):
    """Date only: no reminders, no calendar, no weather of its own."""
    monkeypatch.setattr(tool.system, "tell_date", lambda args: ToolResult.success("It's Wednesday, the 7th of October 2026."))
    monkeypatch.setattr(tool, "ReminderStore", lambda: SimpleNamespace(upcoming=lambda: []))
    monkeypatch.setattr(tool, "get_location", lambda: "")
    monkeypatch.setattr(tool, "_read", lambda: "")


BRIEF_HEAD = ("Good morning. It's Wednesday, the 7th of October 2026. In Oceanside it is sunny and 76 degrees, "
              "with a high of 77 and a low of 63. Nothing is on your reminder list today.")


def test_bills_join_the_briefing(monkeypatch):
    _quiet_briefing_parts(monkeypatch)
    monkeypatch.setattr(tool, "_expenses_read", lambda: "- 2026-10-03 | 12.50 | food | Lunch\n- budget | food | 10\n")
    now = datetime(2026, 10, 7, 9, 0)
    result = tool.daily_briefing({}, now=now, fetch=lambda url: WEATHER)
    assert result.speech == (
        f"{BRIEF_HEAD} You have spent 12 dollars and 50 cents this month."
        " You are 2 dollars and 50 cents over your 10 dollar food budget this month."
    )


def test_bills_left_out_when_the_ledger_is_empty_or_unreadable(monkeypatch):
    _quiet_briefing_parts(monkeypatch)
    now = datetime(2026, 10, 7, 9, 0)
    monkeypatch.setattr(tool, "_expenses_read", lambda: "")
    empty = tool.daily_briefing({}, now=now, fetch=lambda url: WEATHER)
    assert empty.speech == BRIEF_HEAD

    def unreadable():
        raise OSError("no documents folder")

    monkeypatch.setattr(tool, "_expenses_read", unreadable)
    broken = tool.daily_briefing({}, now=now, fetch=lambda url: WEATHER)
    assert broken.ok and broken.speech == BRIEF_HEAD


def test_router():
    for phrase in ("good morning", "brief me", "daily briefing", "what does my day look like"):
        assert intents.route(phrase).tool == "daily_briefing", phrase
