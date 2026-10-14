from __future__ import annotations

import math
from datetime import datetime

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.reminders import Reminder, ReminderStore, parse_when, timer_phrase
from voxa.agent.result import ToolResult


def _fmt_time(due: datetime) -> str:
    return due.strftime("%I:%M %p").lstrip("0")


def _set_timer_handler(args: dict[str, str]) -> ToolResult:
    duration = args["duration"].strip()
    now = datetime.now()
    due = parse_when(f"in {duration}", now)
    if due is None:
        return ToolResult.failure("I couldn't understand that duration.")
    ReminderStore().add(Reminder(id=f"t{int(due.timestamp())}", due=due, text=duration, kind="timer"))
    return ToolResult.success(f"Timer set for {duration}.", detail=due.isoformat(timespec="seconds"))


def _set_reminder_handler(args: dict[str, str]) -> ToolResult:
    when, text = args["when"].strip(), args["text"].strip()
    now = datetime.now()
    due = parse_when(when, now)
    if due is None:
        return ToolResult.failure("I couldn't understand that time.")
    ReminderStore().add(Reminder(id=f"r{int(due.timestamp())}", due=due, text=text, kind="reminder"))
    return ToolResult.success(f"I'll remind you at {_fmt_time(due)} to {text}.", detail=due.isoformat(timespec="seconds"))


def _list_reminders_handler(args: dict[str, str]) -> ToolResult:
    items = ReminderStore().upcoming()
    if not items:
        return ToolResult.success("You have no reminders.")
    parts = [
        (timer_phrase(r.text) if r.kind == "timer" else f"{r.text} at {_fmt_time(r.due)}")
        for r in items
    ]
    word = "reminders" if len(items) > 1 else "reminder"
    return ToolResult.success(f"You have {len(items)} {word}: {', '.join(parts)}.")


def _cancel_reminders_handler(args: dict[str, str]) -> ToolResult:
    store = ReminderStore()
    items = store.upcoming()
    if not items:
        return ToolResult.success("There was nothing to cancel.")
    for item in items:
        store.remove(item.id)
    return ToolResult.success(f"Cancelled {len(items)} reminders.")


def _focus_start_handler(args: dict[str, str]) -> ToolResult:
    duration = args.get("duration", "").strip() or "25 minutes"
    what = args.get("what", "").strip()
    now = datetime.now()
    due = parse_when(f"in {duration}", now)
    if due is None:
        return ToolResult.failure("I couldn't understand that duration.")
    store = ReminderStore()
    for item in store.upcoming():
        if item.kind == "focus":
            store.remove(item.id)
    store.add(Reminder(id=f"f{int(due.timestamp())}", due=due, text=what, kind="focus"))
    suffix = f" for {what}" if what else ""
    return ToolResult.success(
        f"Focus time started: {duration}{suffix}. I'll tell you when it's over.",
        detail=due.isoformat(timespec="seconds"),
    )


def _focus_status_handler(args: dict[str, str]) -> ToolResult:
    now = datetime.now()
    focus = [r for r in ReminderStore().upcoming() if r.kind == "focus"]
    if not focus:
        return ToolResult.success("You are not in a focus session.")
    due = min(r.due for r in focus)
    minutes = max(1, math.ceil((due - now).total_seconds() / 60))
    word = "minute" if minutes == 1 else "minutes"
    return ToolResult.success(f"You have {minutes} {word} of focus time left.")


def _focus_stop_handler(args: dict[str, str]) -> ToolResult:
    store = ReminderStore()
    focus = [r for r in store.upcoming() if r.kind == "focus"]
    if not focus:
        return ToolResult.success("You are not in a focus session.")
    for item in focus:
        store.remove(item.id)
    return ToolResult.success("Focus time ended.")


def reminders_tools() -> list[Tool]:
    return [
        Tool(
            name="set_timer",
            description="Set a countdown timer, e.g. for 10 minutes.",
            parameters={"duration": "the duration, e.g. '10 minutes' or '2 hours'"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_timer_handler,
            required=("duration",),
        ),
        Tool(
            name="set_reminder",
            description="Remind the user at a spoken time, e.g. at 3pm to call the dentist.",
            parameters={"when": "the when-phrase, e.g. 'at 3pm' or 'in 10 minutes'", "text": "what to remind about"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_reminder_handler,
            required=("when", "text"),
        ),
        Tool(
            name="list_reminders",
            description="List the timers and reminders still waiting.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_list_reminders_handler,
        ),
        Tool(
            name="cancel_reminders",
            description="Cancel every timer and reminder.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_cancel_reminders_handler,
        ),
        Tool(
            name="focus_start",
            description="Start a focus session, e.g. for 25 minutes, optionally on a task.",
            parameters={"duration": "the duration, e.g. '25 minutes'; empty means 25 minutes", "what": "what to focus on, optional"},
            risk=RiskLevel.REVERSIBLE,
            handler=_focus_start_handler,
        ),
        Tool(
            name="focus_status",
            description="Report how much focus time is left.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_focus_status_handler,
        ),
        Tool(
            name="focus_stop",
            description="End the focus session.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_focus_stop_handler,
        ),
    ]
