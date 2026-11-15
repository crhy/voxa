"""Bills and subscriptions: the voice layer over the spending log and the reminder store.

Issue #60. Detection is pure (``voxa.agent.bills``); these handlers read the spending log
the expense tools already read, and ``bills_remind`` writes ordinary reminders with
``kind="bill"`` so the existing reminder loop in the window speaks them — no UI change.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import date, datetime, time

from voxa.agent import bills
from voxa.agent.expenses import entries
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.reminders import Reminder, ReminderStore
from voxa.agent.result import ToolResult
from voxa.agent.tools import expenses as expense_files

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your expenses."
BILL_KIND = "bill"
BILL_HOUR = 9  # a bill reminder lands in the morning of its due date


def _days(args: dict[str, str]) -> int:
    try:
        return max(1, int(args.get("days", "").strip()))
    except ValueError:
        return 7


def _subscriptions(args: dict[str, str]) -> list[bills.Subscription] | None:
    """The detected charges, or None when the spending log could not be read."""
    try:
        content = expense_files._read()
    except (OSError, subprocess.SubprocessError):
        return None
    return bills.recurring_charges(entries(content), date.today())


def _bills_due_handler(args: dict[str, str]) -> ToolResult:
    subs = _subscriptions(args)
    if subs is None:
        return ToolResult.failure(REACH_ERROR)
    days = _days(args)
    return ToolResult.success(bills.spoken_due(subs, date.today(), days))


def _subscriptions_list_handler(args: dict[str, str]) -> ToolResult:
    subs = _subscriptions(args)
    if subs is None:
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(bills.spoken_subscriptions(subs), detail=str(len(subs)))


def _bills_remind_handler(args: dict[str, str]) -> ToolResult:
    subs = _subscriptions(args)
    if subs is None:
        return ToolResult.failure(REACH_ERROR)
    if not subs:
        return ToolResult.failure("I have not found any recurring charges to remind you about.")
    store = ReminderStore()
    waiting = {item.id for item in store.upcoming() if item.kind == BILL_KIND}
    added = 0
    for sub in subs:
        reminder_id = f"bill {sub.label.lower()}"
        if reminder_id in waiting:
            continue  # already waiting for this bill
        due = datetime.combine(sub.next_due, time(BILL_HOUR, 0))
        store.add(
            Reminder(
                id=reminder_id,
                due=due,
                text=f"your {sub.label.lower()} bill is due",
                kind=BILL_KIND,
            )
        )
        waiting.add(reminder_id)
        added += 1
    if not added:
        return ToolResult.success("You are already set to be reminded about those bills.")
    word = "bills" if added > 1 else "bill"
    return ToolResult.success(f"I'll remind you about {added} {word} when they come due.")


def bills_tools() -> list[Tool]:
    return [
        Tool(
            name="bills_due",
            description="Say which recurring charges found in the spending log come due in the next week.",
            parameters={"days": "how far ahead to look, e.g. 7"},
            risk=RiskLevel.READ_ONLY,
            handler=_bills_due_handler,
        ),
        Tool(
            name="subscriptions_list",
            description="List the subscriptions you pay for, with their amount and how often they come.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_subscriptions_list_handler,
        ),
        Tool(
            name="bills_remind",
            description="Set a reminder for each recurring charge found in the spending log, once per charge.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_bills_remind_handler,
        ),
    ]
