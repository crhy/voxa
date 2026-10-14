from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.dates import FILE_NAME, add, find, parse_day, parse_monthly, remove, spoken_upcoming, spoken_when
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your dates."


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


def _date_remember_handler(args: dict[str, str]) -> ToolResult:
    kind = args["kind"].strip().lower()
    if kind not in ("birthday", "anniversary", "deadline", "bill", "subscription", "service"):
        return ToolResult.failure("Which kind of date is it?")
    name = args["name"].strip()
    if not name:
        return ToolResult.failure("Who or what is it for?")
    monthly = parse_monthly(args["when"])
    if monthly is not None:
        when: tuple[int | None, int, int] | None = (None, None, monthly)
    else:
        when = parse_day(args["when"], date.today())
    if when is None:
        return ToolResult.failure("I didn't catch that date.")
    year, month, day = when
    try:
        today = date.today()
        content = _read()
        new_content = add(content, kind, name, (year, month, day))
        _write(new_content)
        if year is None and month is None:
            date_text = f"{day:02d}"
        elif year is None:
            date_text = f"{month:02d}-{day:02d}"
        else:
            date_text = f"{year:04d}-{month:02d}-{day:02d}"
        clean = name[0].upper() + name[1:]
        return ToolResult.success(f"Got it. {spoken_when((date_text, kind, clean), today)}")
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)


def _date_when_handler(args: dict[str, str]) -> ToolResult:
    try:
        today = date.today()
        content = _read()
        entry = find(content, args["name"])
        if entry is None:
            return ToolResult.failure(f"I don't have a date for {args['name']}.")
        return ToolResult.success(spoken_when(entry, today))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)


def _date_upcoming_handler(args: dict[str, str]) -> ToolResult:
    try:
        today = date.today()
        days = int(args["days"]) if args.get("days", "").strip().isdigit() else 30
        kind = args.get("kind", "").strip().lower() or None
        content = _read()
        return ToolResult.success(spoken_upcoming(content, today, days, kind))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)


def _date_forget_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
        new_content, removed = remove(content, args["name"])
        if removed is None:
            return ToolResult.failure(f"I don't have a date for {args['name']}.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I forgot {removed}.")


def date_tools() -> list[Tool]:
    return [
        Tool(
            name="date_remember",
            description="Remember a birthday, anniversary, deadline, bill, subscription or service date; the date may be monthly.",
            parameters={"kind": "birthday, anniversary, deadline, bill, subscription or service", "name": "who or what it is for", "when": "the date"},
            risk=RiskLevel.REVERSIBLE,
            handler=_date_remember_handler,
            required=("kind", "name", "when"),
        ),
        Tool(
            name="date_when",
            description="Say when a remembered birthday, anniversary, deadline, bill, subscription or service date falls.",
            parameters={"name": "who or what to look up"},
            risk=RiskLevel.READ_ONLY,
            handler=_date_when_handler,
            required=("name",),
        ),
        Tool(
            name="date_upcoming",
            description="List the birthdays, anniversaries, deadlines, bills, subscriptions and service dates coming up soon.",
            parameters={"days": "how many days ahead to look", "kind": "optional kind to filter by"},
            risk=RiskLevel.READ_ONLY,
            handler=_date_upcoming_handler,
        ),
        Tool(
            name="date_forget",
            description="Forget a remembered birthday, anniversary, deadline, bill, subscription or service date.",
            parameters={"name": "who or what to forget"},
            risk=RiskLevel.REVERSIBLE,
            handler=_date_forget_handler,
            required=("name",),
        ),
    ]
