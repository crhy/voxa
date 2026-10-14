from __future__ import annotations

import logging
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

_SCHEMA = "org.mate.caja.preferences"
_KEY = "show-hidden-files"
_MISSING = "I can't change hidden files on this desktop."


def _set_command(show: bool) -> list[str]:
    return ["gsettings", "set", _SCHEMA, _KEY, "true" if show else "false"]


def _get_command() -> list[str]:
    return ["gsettings", "get", _SCHEMA, _KEY]


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        capture_output=True,
        text=True,
        timeout=10,
        check=check,
    )


def _current_show() -> bool | None:
    """True/False when the desktop reports the setting; None when it cannot."""
    if simulation.actions_simulated():
        return False
    try:
        current = _run(_get_command(), check=False).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    if current == "true":
        return True
    if current == "false":
        return False
    return None


def set_hidden_files(args: dict[str, str]) -> ToolResult:
    mode = args.get("show", "toggle")
    if mode == "on":
        show = True
    elif mode == "off":
        show = False
    elif mode == "toggle":
        current = _current_show()
        if current is None:
            return ToolResult.failure(_MISSING)
        show = not current
    else:
        return ToolResult.failure(_MISSING)
    try:
        _run(_set_command(show))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ToolResult.failure(_MISSING)
    return ToolResult.success("Showing hidden files." if show else "Hiding hidden files.")


def hidden_files_tools() -> list[Tool]:
    return [
        Tool(
            name="hidden_files",
            description="Show or hide hidden files in the file manager.",
            parameters={"show": "on, off or toggle"},
            risk=RiskLevel.REVERSIBLE,
            handler=set_hidden_files,
            required=("show",),
        ),
    ]
