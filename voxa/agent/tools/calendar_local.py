from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import date, datetime, timedelta

from voxa import simulation
from voxa.agent.calendar_local import (
    FILE_NAME,
    _day_word,
    add,
    add_weekly,
    conflicts,
    entries_on,
    free_slot,
    move,
    parse_time,
    remove,
    split_when,
    spoken_count,
    spoken_day,
    spoken_next,
    spoken_time,
    spoken_week,
    week,
)
from voxa.agent.dates import WEEKDAY_NAMES, WEEKDAYS
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.reminders import Reminder, ReminderStore
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your calendar."

_MINUTE_NUMBER = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty five": 45,
}


def _minutes(spoken: str) -> int | None:
    """Minutes from "an hour", "90 minutes", "two hours", "half an hour"; None when unparsable."""
    s = " ".join(spoken.lower().strip().split()).replace("-", " ")
    s = " ".join(s.split())
    if not s:
        return None
    if s in ("half an hour", "half hour"):
        return 30
    m = re.fullmatch(r"(?P<count>[a-z0-9]+(?: five)?) (hours?|minutes?)", s)
    if not m:
        return None
    raw = m.group("count")
    count = int(raw) if raw.isdigit() else _MINUTE_NUMBER.get(raw)
    if count is None:
        return None
    return count * 60 if m.group(2).startswith("h") else count


def _spoken_minutes(minutes: int) -> str:
    """"an hour", "2 hours", "90 minutes", "30 minutes"."""
    if minutes == 60:
        return "an hour"
    if minutes > 60 and minutes % 60 == 0:
        return f"{minutes // 60} hours"
    return f"{minutes} minutes"


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def _path() -> str | None:
    folder = filemanage._folder("documents")
    if folder is None:
        return None
    return os.path.join(folder, FILE_NAME)


def _read() -> str:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    done = _run(["cat", path], check=False)
    if done.returncode != 0:
        return ""
    return done.stdout


def _write(content: str) -> None:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    _run(["sh", "-c", 'cat > "$1"', "sh", path], stdin=content)


def _when_phrase(day: date, time: tuple[int, int] | None, today: date) -> str:
    word = _day_word(day, today)
    if word.startswith("On "):
        word = word[3:]
    word = word.lower()
    if time:
        return f"{word} at {spoken_time(time)}"
    return word


def _calendar_add_handler(args: dict[str, str]) -> ToolResult:
    title = args["title"].strip()
    if not title:
        return ToolResult.failure("What is the appointment?")
    try:
        parsed = split_when(args.get("when", ""), date.today())
        if parsed is None:
            return ToolResult.failure("When is it?")
        day, time = parsed
        content = _read()
        clash = conflicts(entries_on(content, day), day, time) if time else []
        _write(add(content, day, time, title))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    when = _when_phrase(day, time, date.today())
    speech = f"I put {title[0].upper() + title[1:]} on your calendar for {when}."
    if clash:
        clash_time = None
        for _item_day, item_time, item_title in entries_on(content, day):
            if item_title == clash[0]:
                clash_time = item_time
                break
        speech += f" Note: that is close to {clash[0]} at {spoken_time(clash_time)}."
    if time:
        due = datetime(day.year, day.month, day.day, time[0], time[1]) - timedelta(minutes=15)
        if due > datetime.now():
            ReminderStore().add(
                Reminder(id=f"cal{int(due.timestamp())}", due=due, text=f"{title} is in 15 minutes", kind="reminder")
            )
            speech += " I'll remind you 15 minutes before."
    return ToolResult.success(speech)


def _calendar_day_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    spoken = args.get("day", "").strip()
    if not spoken or spoken.lower() == "today":
        day = today
    else:
        parsed = split_when(spoken, today)
        day = parsed[0] if parsed else today
    try:
        items = entries_on(_read(), day)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_day(items, day, today))


def _calendar_next_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_next(content, datetime.now()))


def _calendar_conflicts_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    today = date.today()
    for offset in range(7):
        day = today + timedelta(days=offset)
        timed = [(time, title) for _, time, title in entries_on(content, day) if time]
        for i in range(len(timed)):
            for j in range(i + 1, len(timed)):
                a, b = timed[i], timed[j]
                if abs((a[0][0] * 60 + a[0][1]) - (b[0][0] * 60 + b[0][1])) <= 60:
                    word = WEEKDAY_NAMES[day.weekday()]
                    return ToolResult.success(
                        f"{word}: {a[1]} at {spoken_time(a[0])} is close to {b[1]} at {spoken_time(b[0])}."
                    )
    return ToolResult.success("I see no clashes in the next 7 days.")


def _calendar_add_weekly_handler(args: dict[str, str]) -> ToolResult:
    title = args["title"].strip()
    if not title:
        return ToolResult.failure("What is the appointment?")
    weekday = WEEKDAYS.get(args.get("weekday", "").strip().lower())
    if weekday is None:
        return ToolResult.failure("Which weekday?")
    time = parse_time(args.get("time", ""))
    try:
        content = _read()
        _write(add_weekly(content, weekday, time, title))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    when = f"every {WEEKDAY_NAMES[weekday]}"
    if time:
        when += f" at {spoken_time(time)}"
    return ToolResult.success(f"I put {title[0].upper() + title[1:]} on your calendar {when}.")


def _calendar_cancel_handler(args: dict[str, str]) -> ToolResult:
    title = args["title"].strip()
    if not title:
        return ToolResult.failure("Which appointment should I cancel?")
    try:
        content = _read()
        new_content, removed, day, time = remove(content, title)
        if removed is None:
            return ToolResult.failure(f"I can't find {title} on your calendar.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    if day and time:
        due = datetime(day.year, day.month, day.day, time[0], time[1]) - timedelta(minutes=15)
        ReminderStore().remove(f"cal{int(due.timestamp())}")
    return ToolResult.success(f"I took {removed} off your calendar.")


def _calendar_move_handler(args: dict[str, str]) -> ToolResult:
    title = args["title"].strip()
    if not title:
        return ToolResult.failure("Which appointment should I move?")
    parsed = split_when(args.get("when", ""), date.today())
    if parsed is None:
        return ToolResult.failure("When should it be instead?")
    day, time = parsed
    today = date.today()
    try:
        content = _read()
        new_content, moved, old_day, old_time = move(content, title, day, time)
        if moved is None:
            return ToolResult.failure(f"I can't find {title} on your calendar.")
        others = [item for item in entries_on(new_content, day) if item[2] != moved]
        clash = conflicts(others, day, time) if time else []
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    if old_day and old_time:
        old_due = datetime(old_day.year, old_day.month, old_day.day, old_time[0], old_time[1]) - timedelta(minutes=15)
        ReminderStore().remove(f"cal{int(old_due.timestamp())}")
    if old_day:
        speech = f"I moved {moved} from {_when_phrase(old_day, old_time, today)} to {_when_phrase(day, time, today)}."
    else:
        speech = f"I moved {moved} to {_when_phrase(day, time, today)}."
    if clash:
        clash_time = None
        for _item_day, item_time, item_title in others:
            if item_title == clash[0]:
                clash_time = item_time
                break
        speech += f" Note: that is close to {clash[0]} at {spoken_time(clash_time)}."
    if time:
        due = datetime(day.year, day.month, day.day, time[0], time[1]) - timedelta(minutes=15)
        if due > datetime.now():
            ReminderStore().add(
                Reminder(id=f"cal{int(due.timestamp())}", due=due, text=f"{moved} is in 15 minutes", kind="reminder")
            )
            speech += " I'll remind you 15 minutes before."
    return ToolResult.success(speech)


def _calendar_free_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    spoken = args.get("day", "").strip()
    if not spoken or spoken.lower() == "today":
        day = today
    else:
        parsed = split_when(spoken, today)
        day = parsed[0] if parsed else today
    asked_for = args.get("minutes", "").strip()
    minutes = 60 if not asked_for else _minutes(asked_for)
    if minutes is None:
        return ToolResult.failure("How much free time am I looking for?")
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    slot = free_slot(content, day, minutes)
    if slot is None:
        return ToolResult.success(f"You have no free stretch of {_spoken_minutes(minutes)} {_when_phrase(day, None, today)}.")
    end = (slot[0] * 60 + slot[1] + minutes)
    speech = (
        f"You have {_spoken_minutes(minutes)} free from {spoken_time(slot)} "
        f"to {spoken_time((end // 60, end % 60))} {_when_phrase(day, None, today)}."
    )
    return ToolResult.success(speech)


def _calendar_week_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    person = args.get("person", "").strip()
    return ToolResult.success(spoken_week(week(content, today), today, person or None))


def _calendar_busy_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_count(week(content, today)))


def calendar_local_tools() -> list[Tool]:
    return [
        Tool(
            name="calendar_add",
            description="Add an appointment to the Voxa Calendar file.",
            parameters={"title": "the appointment", "when": "the day and time, e.g. tomorrow at noon"},
            risk=RiskLevel.REVERSIBLE,
            handler=_calendar_add_handler,
            required=("title", "when"),
        ),
        Tool(
            name="calendar_day",
            description="Read back the appointments on a day of the calendar.",
            parameters={"day": "which day, e.g. today or tomorrow"},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_day_handler,
        ),
        Tool(
            name="calendar_next",
            description="Speak the next appointment coming up on the calendar.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_next_handler,
        ),
        Tool(
            name="calendar_move",
            description="Move a calendar appointment to another day and time, and move its reminder.",
            parameters={"title": "the appointment to move", "when": "the new day and time, e.g. thursday at 4 pm"},
            risk=RiskLevel.REVERSIBLE,
            handler=_calendar_move_handler,
            required=("title", "when"),
        ),
        Tool(
            name="calendar_free",
            description="Find the first stretch of free time on a day of the calendar.",
            parameters={"day": "which day, e.g. today or tomorrow", "minutes": "how much free time you need, e.g. an hour"},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_free_handler,
        ),
        Tool(
            name="calendar_cancel",
            description="Remove an appointment from the calendar.",
            parameters={"title": "the appointment to cancel"},
            risk=RiskLevel.REVERSIBLE,
            handler=_calendar_cancel_handler,
            required=("title",),
        ),
        Tool(
            name="calendar_conflicts",
            description="Check the calendar for appointments that clash in the next 7 days.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_conflicts_handler,
        ),
        Tool(
            name="calendar_add_weekly",
            description="Add a recurring weekly appointment to the Voxa Calendar file.",
            parameters={"title": "the appointment", "weekday": "the weekday, e.g. monday", "time": "the time, e.g. 9 am"},
            risk=RiskLevel.REVERSIBLE,
            handler=_calendar_add_weekly_handler,
            required=("title", "weekday", "time"),
        ),
        Tool(
            name="calendar_week",
            description="Read back the appointments in the week ahead, optionally for one person.",
            parameters={"person": "whose week, e.g. Emma"},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_week_handler,
        ),
        Tool(
            name="calendar_busy",
            description="Count the appointments this week and name the busiest day.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_calendar_busy_handler,
        ),
    ]
