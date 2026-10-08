"""The morning briefing: the day, the weather, and what is due — each part optional."""

from __future__ import annotations

import json
from datetime import datetime


def parse_weather(json_text: str, fahrenheit: bool = True) -> str:
    """One sentence from wttr.in's ``format=j1`` answer; "" on anything unexpected."""
    try:
        data = json.loads(json_text)
        now = data["current_condition"][0]
        today = data["weather"][0]
        place = data["nearest_area"][0]["areaName"][0]["value"]
        unit = "F" if fahrenheit else "C"
        sky = now["weatherDesc"][0]["value"].strip().lower()
        return (
            f"In {place} it is {sky} and {now['temp_' + unit]} degrees, "
            f"with a high of {today['maxtemp' + unit]} and a low of {today['mintemp' + unit]}."
        )
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def uses_fahrenheit(json_text: str) -> bool:
    """Fahrenheit for the United States, Celsius elsewhere."""
    try:
        country = json.loads(json_text)["nearest_area"][0]["country"][0]["value"]
    except (ValueError, KeyError, IndexError, TypeError):
        return True
    return "united states" in country.lower()


def reminders_today(reminders, now: datetime) -> str:
    """"You have 2 reminders today: call Bob at 3:00 PM, and …"; "" when there are none."""
    due = sorted((r for r in reminders if r.due.date() == now.date() and r.due >= now), key=lambda r: r.due)
    if not due:
        return ""
    parts = [f"{r.text} at {r.due.strftime('%-I:%M %p')}" for r in due[:3]]
    listed = ", and ".join(parts) if len(parts) > 1 else parts[0]
    more = f", and {len(due) - 3} more" if len(due) > 3 else ""
    noun = "reminder" if len(due) == 1 else "reminders"
    return f"You have {len(due)} {noun} today: {listed}{more}."


def compose(greeting: str, date_sentence: str, weather: str, reminders: str) -> str:
    parts = [greeting, date_sentence, weather, reminders or "Nothing is on your reminder list today."]
    return " ".join(part.strip() for part in parts if part and part.strip())


def greeting_for(hour: int) -> str:
    if hour < 12:
        return "Good morning."
    return "Good afternoon." if hour < 18 else "Good evening."
