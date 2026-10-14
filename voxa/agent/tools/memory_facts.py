from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.memory_facts import FILE_NAME, add, facts, forget, search, spoken_all, spoken_recall
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach my memory."


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


def _remember_fact_handler(args: dict[str, str]) -> ToolResult:
    fact = args["fact"].strip()
    if not fact:
        return ToolResult.failure("What should I remember?")
    try:
        content = _read()
        _write(add(content, date.today(), fact))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success("I'll remember that.")


def _recall_fact_handler(args: dict[str, str]) -> ToolResult:
    topic = args["topic"].strip()
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_recall(search(facts(content), topic), topic))


def _list_facts_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(spoken_all(facts(content)))


def _forget_fact_handler(args: dict[str, str]) -> ToolResult:
    topic = args["topic"].strip()
    try:
        content = _read()
        new_content, removed = forget(content, topic)
        if removed is None:
            return ToolResult.failure(f"I don't have anything about {topic}.")
        _write(new_content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"Forgotten: {removed}.")


def memory_facts_tools() -> list[Tool]:
    return [
        Tool(
            name="remember_fact",
            description="Remember a personal fact in the Voxa Memory file.",
            parameters={"fact": "the fact to remember"},
            risk=RiskLevel.REVERSIBLE,
            handler=_remember_fact_handler,
            required=("fact",),
        ),
        Tool(
            name="recall_fact",
            description="Recall what Voxa remembers about a topic.",
            parameters={"topic": "what to recall"},
            risk=RiskLevel.READ_ONLY,
            handler=_recall_fact_handler,
            required=("topic",),
        ),
        Tool(
            name="list_facts",
            description="List everything Voxa remembers.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_list_facts_handler,
        ),
        Tool(
            name="forget_fact",
            description="Forget a remembered fact.",
            parameters={"topic": "which fact to forget"},
            risk=RiskLevel.REVERSIBLE,
            handler=_forget_fact_handler,
            required=("topic",),
        ),
    ]
