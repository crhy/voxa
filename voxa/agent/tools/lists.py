from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.dates import upcoming
from voxa.agent.host import host_command
from voxa.agent.lists import (
    FILE_NAME,
    add,
    clean_item,
    clear,
    complete,
    list_name,
    parse,
    remove,
    spoken,
    spoken_next,
    spoken_pending,
    spoken_progress,
)
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import dates as dates_tools
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your lists."


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


def _list_add_handler(args: dict[str, str]) -> ToolResult:
    item = args["item"].strip()
    if not item:
        return ToolResult.failure("What should I add?")
    try:
        name = list_name(args.get("list", ""))
        content = _read()
        _write(add(content, name, item))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I added {clean_item(item)} to your {name.lower()} list.")


def _list_read_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken(parse(content), list_name(args.get("list", ""))))


def _list_done_handler(args: dict[str, str]) -> ToolResult:
    try:
        name = list_name(args.get("list", ""))
        content = _read()
        new_content, text = complete(content, name, args["item"])
        if text is None and not args.get("list", "").strip():
            # "Mark milk as done" names no list: look on every list, not only the to-do list.
            for other in parse(content):
                new_content, text = complete(content, other, args["item"])
                if text is not None:
                    break
        if text is None:
            return ToolResult.failure(f"I can't find {args['item']} on your {name.lower()} list.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"Done: {text}.")


def _list_remove_handler(args: dict[str, str]) -> ToolResult:
    try:
        name = list_name(args.get("list", ""))
        content = _read()
        new_content, text = remove(content, name, args["item"])
        if text is None:
            return ToolResult.failure(f"I can't find {args['item']} on your {name.lower()} list.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I removed {text} from your {name.lower()} list.")


def _list_clear_handler(args: dict[str, str]) -> ToolResult:
    try:
        name = list_name(args.get("list", ""))
        content = _read()
        new_content, removed = clear(content, name)
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    if removed == 0:
        return ToolResult.success(f"Your {name.lower()} list was already empty.")
    return ToolResult.success(f"I cleared your {name.lower()} list.")


def _list_progress_handler(args: dict[str, str]) -> ToolResult:
    try:
        name = list_name(args.get("list", ""))
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_progress(parse(content), name))


def _list_pending_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_pending(parse(content)))


def _list_next_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    try:
        due = [name for _, _, name in upcoming(dates_tools._read(), date.today(), 7)]
    except Exception:
        due = []
    return ToolResult.success(spoken_next(parse(content), due))


def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_add",
            description="Add an item to a to-do or shopping list in the Voxa Lists file.",
            parameters={"item": "what to add", "list": "which list, e.g. to-do or shopping"},
            risk=RiskLevel.REVERSIBLE,
            handler=_list_add_handler,
            required=("item",),
        ),
        Tool(
            name="list_read",
            description="Read back the unfinished items on a list.",
            parameters={"list": "which list, e.g. to-do or shopping"},
            risk=RiskLevel.READ_ONLY,
            handler=_list_read_handler,
        ),
        Tool(
            name="list_done",
            description="Mark an item on a list as done.",
            parameters={"item": "what was finished", "list": "which list"},
            risk=RiskLevel.REVERSIBLE,
            handler=_list_done_handler,
            required=("item",),
        ),
        Tool(
            name="list_remove",
            description="Remove an item from a list.",
            parameters={"item": "what to remove", "list": "which list"},
            risk=RiskLevel.REVERSIBLE,
            handler=_list_remove_handler,
            required=("item",),
        ),
        Tool(
            name="list_clear",
            description="Remove every item from a list.",
            parameters={"list": "which list"},
            risk=RiskLevel.REVERSIBLE,
            handler=_list_clear_handler,
        ),
        Tool(
            name="list_progress",
            description="Report how many items on a list are finished and which are still open.",
            parameters={"list": "which list, e.g. to-do or shopping"},
            risk=RiskLevel.READ_ONLY,
            handler=_list_progress_handler,
        ),
        Tool(
            name="list_pending",
            description="Count the open items over every list.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_list_pending_handler,
        ),
        Tool(
            name="list_next",
            description="Say which open to-do item to start with, using deadlines from the dates file.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_list_next_handler,
        ),
    ]
