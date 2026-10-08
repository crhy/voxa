import json
from datetime import datetime
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


def test_router():
    for phrase in ("good morning", "brief me", "daily briefing", "what does my day look like"):
        assert intents.route(phrase).tool == "daily_briefing", phrase
