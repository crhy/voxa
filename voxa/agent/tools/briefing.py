from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime

from voxa.agent import briefing
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.reminders import ReminderStore
from voxa.agent.result import ToolResult
from voxa.agent.tools import system

get_location: Callable[[], str] | None = None   # set by the window: the saved place, or "" to detect it


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Voxa"})
    with urllib.request.urlopen(request, timeout=6) as response:  # noqa: S310 - fixed https address
        return response.read().decode("utf-8", "replace")


def daily_briefing(args: dict[str, str], now: datetime | None = None, fetch=None) -> ToolResult:
    now = now or datetime.now()
    date_sentence = ""
    try:
        told = system.tell_date({})
        date_sentence = told.speech if told.ok else ""
    except Exception:  # noqa: BLE001 - every part of the briefing is optional
        pass
    weather = ""
    try:
        place = get_location() if get_location is not None else ""
        text = (fetch or _fetch)(f"https://wttr.in/{urllib.parse.quote(place or '')}?format=j1")
        weather = briefing.parse_weather(text, briefing.uses_fahrenheit(text))
    except Exception:  # noqa: BLE001 - no network: leave the weather out
        pass
    reminders = ""
    try:
        reminders = briefing.reminders_today(ReminderStore().upcoming(), now)
    except Exception:  # noqa: BLE001
        pass
    return ToolResult.success(briefing.compose(briefing.greeting_for(now.hour), date_sentence, weather, reminders))


def briefing_tools() -> list[Tool]:
    return [
        Tool(name="daily_briefing", description="Speak a short briefing: date, weather, today's reminders.",
             parameters={}, risk=RiskLevel.READ_ONLY, handler=daily_briefing),
    ]
