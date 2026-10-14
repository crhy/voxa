from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.expenses import (
    FILE_NAME,
    add,
    budgets,
    category_for,
    entries,
    parse_amount,
    remove_last,
    set_budget,
    spoken_breakdown,
    spoken_budget,
    spoken_money,
    spoken_total,
    spoken_trip,
    tagged,
    to_csv,
)
from voxa.agent.host import host_command, spawn
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your expenses."

CSV_NAME = "Voxa Expenses.csv"


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


def _write_csv(content: str) -> None:
    folder = filemanage._folder("documents")
    if folder is None:
        raise OSError("no documents folder")
    path = os.path.join(folder, CSV_NAME)
    _run(["sh", "-c", 'cat > "$1"', "sh", path], stdin=content)


def _expense_add_handler(args: dict[str, str]) -> ToolResult:
    amount = parse_amount(args.get("amount", ""))
    if amount is None:
        return ToolResult.failure("How much was it?")
    what = args.get("what", "").strip()
    if not what:
        return ToolResult.failure("What was it for?")
    try:
        content = _read()
        _write(add(content, date.today(), amount, what, args.get("tag", "").strip()))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    money = spoken_money(amount)
    category = category_for(what)
    return ToolResult.success(f"Noted: {money} for {what.lower()}, under {category}.")


def _expense_total_handler(args: dict[str, str]) -> ToolResult:
    category = args.get("category", "").strip() or None
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_total(entries(content), date.today(), args.get("period", ""), category))


def _expense_breakdown_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_breakdown(entries(content), date.today(), args.get("period", "")))


def _expense_undo_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
        new_content, removed = remove_last(content)
        if removed is None:
            return ToolResult.failure("There is no expense to remove.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    _day, amount, _cat, what = removed
    money = spoken_money(amount)
    return ToolResult.success(f"I removed the last expense: {money} for {what.lower()}.")


def _budget_set_handler(args: dict[str, str]) -> ToolResult:
    amount = parse_amount(args.get("amount", ""))
    if amount is None:
        return ToolResult.failure("How much should the budget be?")
    category = args.get("category", "").strip()
    if not category:
        return ToolResult.failure("Which category?")
    try:
        content = _read()
        _write(set_budget(content, category, amount))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"Your {category} budget is {spoken_money(amount)} a month.")


def _budget_status_handler(args: dict[str, str]) -> ToolResult:
    category = args.get("category", "").strip() or None
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_budget(entries(content), budgets(content), date.today(), category))


def _trip_cost_handler(args: dict[str, str]) -> ToolResult:
    name = args.get("name", "").strip()
    if not name:
        return ToolResult.failure("Which trip?")
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_trip(tagged(content, name), name))


def _expense_spreadsheet_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    items = entries(content)
    if not items:
        return ToolResult.failure("There are no expenses to put in a spreadsheet.")
    try:
        _write_csv(to_csv(items))
        folder = filemanage._folder("documents")
        if folder is None:
            return ToolResult.failure(REACH_ERROR)
        spawn(["xdg-open", os.path.join(folder, CSV_NAME)])
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I made a spreadsheet of {len(items)} expenses and opened it.")


def expense_tools() -> list[Tool]:
    return [
        Tool(
            name="expense_add",
            description="Record a spent expense with its amount and what it was for, optionally tagged with a trip.",
            parameters={
                "amount": "how much, e.g. 12 dollars",
                "what": "what it was for, e.g. lunch",
                "tag": "trip name, if the expense belongs to a trip",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=_expense_add_handler,
            required=("amount", "what"),
        ),
        Tool(
            name="expense_total",
            description="Total spending over a period, optionally for one category.",
            parameters={"period": "today, this week, this month or this year", "category": "which category, if any"},
            risk=RiskLevel.READ_ONLY,
            handler=_expense_total_handler,
        ),
        Tool(
            name="expense_breakdown",
            description="Break spending down by category over a period.",
            parameters={"period": "today, this week, this month or this year"},
            risk=RiskLevel.READ_ONLY,
            handler=_expense_breakdown_handler,
        ),
        Tool(
            name="expense_undo",
            description="Remove the most recently recorded expense.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_expense_undo_handler,
        ),
        Tool(
            name="budget_set",
            description="Set a monthly budget for one spending category.",
            parameters={"category": "which category, e.g. food", "amount": "how much, e.g. 300 dollars"},
            risk=RiskLevel.REVERSIBLE,
            handler=_budget_set_handler,
            required=("category", "amount"),
        ),
        Tool(
            name="budget_status",
            description="Report spending against budgets, for one category or for all.",
            parameters={"category": "which category, if any"},
            risk=RiskLevel.READ_ONLY,
            handler=_budget_status_handler,
        ),
        Tool(
            name="trip_cost",
            description="Total the cost of expenses tagged with a trip.",
            parameters={"name": "the trip name, e.g. Denver"},
            risk=RiskLevel.READ_ONLY,
            handler=_trip_cost_handler,
            required=("name",),
        ),
        Tool(
            name="expense_spreadsheet",
            description="Write the spending log out as a CSV spreadsheet and open it.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_expense_spreadsheet_handler,
        ),
    ]
