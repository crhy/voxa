from __future__ import annotations

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
    ]
