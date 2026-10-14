from __future__ import annotations

import logging
import subprocess

from voxa import simulation
from voxa.agent.devstatus import parse_git_status, project_dir, spoken_git, spoken_model
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

_FAILURE = "I could not read that from this computer."

_STATUS_FILE = "/Voxa/.voxa-spec/tasks/agent/status.txt"
_HEALTH_URL = "http://127.0.0.1:8080/health"


_SIMULATED = {
    "ls": "/Voxa/.git\n/home/me/voxa/.git\n",
    "home": "/home/me\n",
    "git status": "## main...origin/main [ahead 3, behind 2]\nM voxa/agent/lists.py\n",
    "git log": "Fix the radio list\n",
    "curl": '{"status":"ok","model":"qwen3.8-flash-next-coder-iq1_m"}\n',
    "tail": "06:08:48 c92-headsup OK (144 paths changed; 2797 passed, 1 skipped in 77.22s (0:01:17))\n",
}


def _canned(command: list[str]) -> str:
    if command[:2] == ["sh", "-c"]:
        script = command[2]
        if "ls -d" in script:
            return _SIMULATED["ls"]
        if "echo $HOME" in script:
            return _SIMULATED["home"]
        return ""
    if command[0] == "git":
        return _SIMULATED["git status"] if "status" in command else _SIMULATED["git log"]
    if command[0] == "curl":
        return _SIMULATED["curl"]
    if command[0] == "tail":
        return _SIMULATED["tail"]
    return ""


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, _canned(command), "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def _project_status_handler(args: dict[str, str]) -> ToolResult:
    name = args["name"]
    try:
        listing = _run(["sh", "-c", 'ls -d /*/.git "$HOME"/*/.git 2>/dev/null'], check=False).stdout
        home = _run(["sh", "-c", "echo $HOME"], check=False).stdout.strip()
        candidates = [line[: -len("/.git")] if line.endswith("/.git") else line for line in listing.splitlines() if line]
        folder = project_dir(name, home, candidates)
        if folder is None:
            return ToolResult.failure(f"I can't find a project called {name}.")
        status = _run(["git", "-C", folder, "status", "--porcelain=v1", "--branch"], check=False).stdout
        last = _run(["git", "-C", folder, "log", "-1", "--format=%s"], check=False).stdout.strip()
        branch, changed, ahead, behind = parse_git_status(status)
        return ToolResult.success(spoken_git(name, branch, changed, ahead, behind, last))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(_FAILURE)


def _model_status_handler(args: dict[str, str]) -> ToolResult:
    try:
        health = _run(["curl", "-s", "-m", "2", _HEALTH_URL], check=False).stdout.strip()
        last_line = _run(["tail", "-n", "1", _STATUS_FILE], check=False).stdout.strip()
        return ToolResult.success(spoken_model(health or None, last_line or None))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(_FAILURE)


def devstatus_tools() -> list[Tool]:
    return [
        Tool(
            name="project_status",
            description="Say the git branch, changed files and last commit of a code project by name.",
            parameters={"name": "the project to check"},
            risk=RiskLevel.READ_ONLY,
            handler=_project_status_handler,
            required=("name",),
        ),
        Tool(
            name="model_status",
            description="Say whether the local model is running and what its last task was.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_model_status_handler,
        ),
    ]
