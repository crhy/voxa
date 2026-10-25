from __future__ import annotations

import json
import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.finance_docs import (
    KINDS,
    folder_name,
    kind_of,
    normalize_kind,
    spoken_count,
    spoken_sort,
    spoken_undo,
)
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your documents."


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=20,
        check=check,
    )


def _state_path() -> str:
    state = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(state, "voxa", "finance-moves.json")


def _read_state() -> list[dict[str, str]]:
    path = _state_path()
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _write_state(moves: list[dict[str, str]]) -> None:
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(moves, f)


def _find_files(folder: str) -> list[str]:
    done = _run(["find", folder, "-maxdepth", "1", "-type", "f"], check=False)
    if done.returncode != 0:
        return []
    return [line for line in done.stdout.splitlines() if line]


def _unique_dest(dest_dir: str, filename: str) -> str:
    dest = os.path.join(dest_dir, filename)
    done = _run(["test", "-e", dest], check=False)
    if done.returncode != 0:
        return dest
    base, ext = os.path.splitext(filename)
    return os.path.join(dest_dir, f"{base} 2{ext}")


def _sort_handler(args: dict[str, str]) -> ToolResult:
    try:
        downloads = filemanage._folder("downloads")
        documents = filemanage._folder("documents")
        if downloads is None or documents is None:
            return ToolResult.failure(REACH_ERROR)
        finance_base = os.path.join(documents, "Finance")
        counts: dict[str, int] = {}
        moves: list[dict[str, str]] = []
        for folder in (downloads, documents):
            for filepath in _find_files(folder):
                filename = os.path.basename(filepath)
                kind = kind_of(filename)
                if kind is None:
                    continue
                dest_dir = os.path.join(finance_base, folder_name(kind))
                _run(["mkdir", "-p", dest_dir])
                dest = _unique_dest(dest_dir, filename)
                _run(["gio", "move", filepath, dest])
                moves.append({"from": filepath, "to": dest})
                counts[kind] = counts.get(kind, 0) + 1
        _write_state(moves)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_sort(counts))


def _undo_handler(args: dict[str, str]) -> ToolResult:
    try:
        moves = _read_state()
        if not moves:
            return ToolResult.success(spoken_undo(0))
        for move in reversed(moves):
            _run(["gio", "move", move["to"], move["from"]])
        _write_state([])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_undo(len(moves)))


def _count_handler(args: dict[str, str]) -> ToolResult:
    kind = normalize_kind(args.get("kind", ""))
    if kind not in KINDS:
        return ToolResult.failure(f"I do not know the kind {args.get('kind', '')}.")
    try:
        documents = filemanage._folder("documents")
        if documents is None:
            return ToolResult.failure(REACH_ERROR)
        folder = os.path.join(documents, "Finance", folder_name(kind))
        files = _find_files(folder)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_count(kind, len(files)))


def finance_docs_tools() -> list[Tool]:
    return [
        Tool(
            name="finance_sort",
            description="Move financial documents from Downloads and Documents into Finance subfolders.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_sort_handler,
        ),
        Tool(
            name="finance_undo",
            description="Move previously filed financial documents back to where they were.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_undo_handler,
        ),
        Tool(
            name="finance_count",
            description="Count the files in a Finance subfolder.",
            parameters={"kind": "the kind of document, e.g. invoice or receipt"},
            risk=RiskLevel.READ_ONLY,
            handler=_count_handler,
        ),
    ]
