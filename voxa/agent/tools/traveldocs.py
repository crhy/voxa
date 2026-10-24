from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage
from voxa.agent.traveldocs import (
    FILE_NAME,
    add,
    forget,
    parse,
    parse_expiry,
    spoken_list,
    spoken_saved,
    spoken_when,
    when,
)

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your travel documents."


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


def _travel_doc_add_handler(args: dict[str, str]) -> ToolResult:
    document = args["document"].strip()
    if not document:
        return ToolResult.failure("Which document?")
    expires = args.get("expires", "").strip()
    today = date.today()
    day = parse_expiry(expires, today)
    if day is None:
        return ToolResult.failure("I need a full date, like June 5 2028.")
    iso = day.isoformat()
    try:
        content = _read()
        _write(add(content, document, iso))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_saved(document, iso, today))


def _travel_doc_when_handler(args: dict[str, str]) -> ToolResult:
    document = args["document"].strip()
    if not document:
        return ToolResult.failure("Which document?")
    today = date.today()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    iso = when(content, document)
    if iso is None:
        return ToolResult.failure(f"I don't have a saved expiry for your {document.lower()}.")
    return ToolResult.success(spoken_when(document, iso, today))


def _travel_doc_list_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_list(parse(content), today))


def _travel_doc_forget_handler(args: dict[str, str]) -> ToolResult:
    document = args["document"].strip()
    if not document:
        return ToolResult.failure("Which document?")
    try:
        content = _read()
        new_content, removed = forget(content, document)
        if removed is None:
            return ToolResult.failure(f"I don't have a saved {document.lower()}.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I forgot your {removed.lower()}.")


def travel_doc_tools() -> list[Tool]:
    return [
        Tool(
            name="travel_doc_add",
            description="Save a travel document and its expiry date in the Voxa Travel Documents file.",
            parameters={"document": "which document, e.g. passport", "expires": "the expiry date, e.g. June 5 2028"},
            risk=RiskLevel.REVERSIBLE,
            handler=_travel_doc_add_handler,
            required=("document", "expires"),
        ),
        Tool(
            name="travel_doc_when",
            description="Report the saved expiry date of a travel document.",
            parameters={"document": "which document"},
            risk=RiskLevel.READ_ONLY,
            handler=_travel_doc_when_handler,
            required=("document",),
        ),
        Tool(
            name="travel_doc_list",
            description="List every saved travel document and its expiry date.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_travel_doc_list_handler,
        ),
        Tool(
            name="travel_doc_forget",
            description="Remove a saved travel document.",
            parameters={"document": "which document"},
            risk=RiskLevel.REVERSIBLE,
            handler=_travel_doc_forget_handler,
            required=("document",),
        ),
    ]
