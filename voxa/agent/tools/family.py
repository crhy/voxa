from __future__ import annotations

import logging
import subprocess
from datetime import date

from voxa.agent.calendar_local import (
    add,
    person_of,
    remove,
    split_when,
    spoken_week,
    week,
)
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.calendar_local import REACH_ERROR, _read, _when_phrase, _write

log = logging.getLogger(__name__)


def _title(person: str, event: str) -> str:
    person = person.strip()
    event = event.strip()
    if person:
        return f"{person}: {event}"
    return event


def _family_add_handler(args: dict[str, str]) -> ToolResult:
    person = args["person"].strip()
    event = args["event"].strip()
    if not event:
        return ToolResult.failure("What should I add?")
    if not person:
        return ToolResult.failure("Who is it for?")
    try:
        parsed = split_when(args.get("when", ""), date.today())
        if parsed is None:
            return ToolResult.failure("When is it?")
        day, time = parsed
        title = _title(person, event)
        content = _read()
        _write(add(content, day, time, title))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    when = _when_phrase(day, time, date.today())
    return ToolResult.success(f"I put {title} on {person}'s schedule for {when}.")


def _family_week_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    person = args.get("person", "").strip()
    items = week(content, today)
    if person:
        want = person.lower()
        items = [item for item in items if (person_of(item[2]) or "").lower() == want]
        return ToolResult.success(spoken_week(items, today, person))
    groups: dict[str, list] = {}
    order: list[str] = []
    for item in items:
        who = person_of(item[2]) or "the family"
        if who not in groups:
            groups[who] = []
            order.append(who)
        groups[who].append(item)
    if not items:
        return ToolResult.success("Nothing is on the family calendar this week.")
    return ToolResult.success(" ".join(spoken_week(groups[who], today, who) for who in order))


def _family_remove_handler(args: dict[str, str]) -> ToolResult:
    person = args["person"].strip()
    event = args["event"].strip()
    if not event:
        return ToolResult.failure("What should I remove?")
    title = _title(person, event)
    try:
        content = _read()
        new_content, removed, _day, _time = remove(content, title)
        if removed is None and person:
            new_content, removed, _day, _time = remove(content, event)
        if removed is None:
            return ToolResult.failure(f"I can't find {event} on {person or 'the family'}'s schedule.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I took {removed} off the schedule.")


def family_tools() -> list[Tool]:
    return [
        Tool(
            name="family_add",
            description="Add an event for one family member to the Voxa Calendar file.",
            parameters={"person": "who it is for, e.g. Emma", "event": "the event", "when": "the day and time"},
            risk=RiskLevel.REVERSIBLE,
            handler=_family_add_handler,
            required=("person", "event", "when"),
        ),
        Tool(
            name="family_week",
            description="Read back the week ahead grouped by family member.",
            parameters={"person": "whose week, e.g. Emma; empty means everyone"},
            risk=RiskLevel.READ_ONLY,
            handler=_family_week_handler,
        ),
        Tool(
            name="family_remove",
            description="Remove a family member's event from the Voxa Calendar file.",
            parameters={"person": "who it is for, e.g. Emma", "event": "the event to remove"},
            risk=RiskLevel.REVERSIBLE,
            handler=_family_remove_handler,
            required=("event",),
        ),
    ]
