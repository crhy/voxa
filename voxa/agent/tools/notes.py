from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.notes import FILE_NAME, note_line, parse_notes, spoken_notes, without_last
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your notes."


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


def _take_note_handler(args: dict[str, str]) -> ToolResult:
    text = args["text"].strip()
    if not text:
        return ToolResult.failure("What should the note say?")
    try:
        path = _path()
        if path is None:
            return ToolResult.failure(REACH_ERROR)
        stamp = _run(["date", "+%Y-%m-%d %H:%M"]).stdout.strip()
        line = note_line(text, stamp)
        _run(["sh", "-c", 'printf "%s\\n" "$1" >> "$2"', "sh", line, path])
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success("Noted.")


def _read_notes_handler(args: dict[str, str]) -> ToolResult:
    try:
        path = _path()
        if path is None:
            return ToolResult.failure(REACH_ERROR)
        content = _run(["cat", path], check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_notes(parse_notes(content)))


def _delete_last_note_handler(args: dict[str, str]) -> ToolResult:
    try:
        path = _path()
        if path is None:
            return ToolResult.failure(REACH_ERROR)
        content = _run(["cat", path], check=False).stdout
        new_content, text = without_last(content)
        if text is None:
            return ToolResult.failure("You have no notes.")
        _run(["sh", "-c", 'printf "%s" "$1" > "$2"', "sh", new_content, path])
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"Deleted the note: {text}")


def _open_notes_handler(args: dict[str, str]) -> ToolResult:
    try:
        path = _path()
        if path is None:
            return ToolResult.failure(REACH_ERROR)
        content = _run(["cat", path], check=False).stdout
        if not content.strip():
            return ToolResult.failure("You have no notes yet.")
        _run(["gio", "open", path], check=False)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success("Opening your notes.")


def notes_tools() -> list[Tool]:
    return [
        Tool(
            name="take_note",
            description="Add a note to the Voxa Notes text file in the Documents folder.",
            parameters={"text": "what the note should say"},
            risk=RiskLevel.REVERSIBLE,
            handler=_take_note_handler,
            required=("text",),
        ),
        Tool(
            name="read_notes",
            description="Read back the notes, newest first.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_read_notes_handler,
        ),
        Tool(
            name="delete_last_note",
            description="Delete the most recent note.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_delete_last_note_handler,
        ),
        Tool(
            name="open_notes",
            description="Open the Voxa Notes text file in the Documents folder.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_open_notes_handler,
        ),
    ]
