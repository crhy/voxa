from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable

from voxa import simulation
from voxa.agent.cleanup import CLEANUP_PROMPT, sane_result
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

ask_model: Callable | None = None


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=5,
        check=check,
    )


def cleanup_text(args: dict[str, str]) -> ToolResult:
    saved = _run(["xclip", "-selection", "clipboard", "-o"]).stdout
    _run(["xdotool", "key", "--clearmodifiers", "ctrl+a"])
    _run(["xdotool", "key", "--clearmodifiers", "ctrl+c"])
    time.sleep(0.15)
    copied = _run(["xclip", "-selection", "clipboard", "-o"]).stdout
    if not copied or (copied == saved and len(copied) > 20000):
        return ToolResult.failure("I could not read any text in that window.")
    if ask_model is None:
        return ToolResult.failure("I could not read any text in that window.")
    edited = ask_model(
        [
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": copied},
        ]
    )
    if sane_result(copied, edited):
        _run(["xclip", "-selection", "clipboard", "-i"], stdin=edited)
        _run(["xdotool", "key", "ctrl+v"])
        time.sleep(0.3)
        _run(["xclip", "-selection", "clipboard", "-i"], stdin=saved)
        return ToolResult.success("Text edited for clarity.")
    _run(["xdotool", "key", "Right"])
    return ToolResult.success("I was not sure about my edit, so I left your text as it was.")


def textedit_tools() -> list[Tool]:
    return [
        Tool(
            name="cleanup_text",
            description="Clean up the text in the focused window for clarity, spelling and grammar.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=cleanup_text,
            required=(),
        ),
    ]
