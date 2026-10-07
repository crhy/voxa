from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

GRABBERS = (
    ("scrot", ["scrot", "{path}"]),
    ("gnome-screenshot", ["gnome-screenshot", "-f", "{path}"]),
    ("import", ["import", "-window", "root", "{path}"]),
    ("grim", ["grim", "{path}"]),
)

NO_GRABBER = "I could not take a screenshot on this computer."


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
        check=check,
    )


def screenshot_name(stamp: str) -> str:
    return f"Screenshot {stamp}.png"


def take_screenshot() -> ToolResult:
    try:
        folder = filemanage._folder("pictures")
        if folder is None:
            return ToolResult.failure(NO_GRABBER)
        stamp = _run(["date", "+%Y-%m-%d %H-%M-%S"]).stdout.strip()
        path = os.path.join(folder, screenshot_name(stamp))
        for program, template in GRABBERS:
            probe = _run(["sh", "-c", 'command -v "$1"', "sh", program], check=False)
            if probe.returncode != 0:
                continue
            command = [part.replace("{path}", path) for part in template]
            if _run(command, check=False).returncode != 0:
                continue
            if _run(["test", "-s", path], check=False).returncode == 0:
                return ToolResult.success("Screenshot saved to Pictures.", detail=path)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(NO_GRABBER)
    return ToolResult.failure(NO_GRABBER)


def _take_screenshot_handler(args: dict[str, str]) -> ToolResult:
    return take_screenshot()


def screenshot_tools() -> list[Tool]:
    return [
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the screen and save it to the Pictures folder.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_take_screenshot_handler,
        ),
    ]
