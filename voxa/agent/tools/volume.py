from __future__ import annotations

import logging
import re
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.volume")

_MIN_VOLUME = 0
_MAX_VOLUME = 150


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        capture_output=True,
        text=True,
        timeout=5,
        check=check,
    )


def _clamp(percent: str) -> int | None:
    try:
        value = int(percent)
    except ValueError:
        return None
    return max(_MIN_VOLUME, min(_MAX_VOLUME, value))


def _parse_volume(stdout: str) -> int | None:
    match = re.search(r"(\d{1,3})%", stdout)  # e.g. "Volume: mono: 45880 /  70% / -9.29 dB"
    return int(match.group(1)) if match else None


def system_volume(args: dict[str, str]) -> ToolResult:
    action = args["action"]

    if action == "louder":
        _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], check=False)
        return ToolResult.success("Volume up.")
    if action == "quieter":
        _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], check=False)
        return ToolResult.success("Volume down.")
    if action == "set":
        value = _clamp(args.get("percent", ""))
        if value is None:
            return ToolResult.failure("I need a volume percentage to set.")
        _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{value}%"], check=False)
        return ToolResult.success(f"Volume {value} percent.")
    if action == "mute":
        _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], check=False)
        return ToolResult.success("Muted.")
    if action == "unmute":
        _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], check=False)
        return ToolResult.success("Unmuted.")
    if action == "get":
        result = _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], check=False)
        value = _parse_volume(result.stdout)
        if value is None:
            return ToolResult.failure("I couldn't read the volume.")
        return ToolResult.success(f"Volume {value} percent.")

    return ToolResult.failure("I don't know that volume action.")


def volume_tools() -> list[Tool]:
    return [
        Tool(
            name="system_volume",
            description=(
                "Adjust the computer's speaker volume: louder, quieter, set to a "
                "percent, mute, unmute, or get the current volume."
            ),
            parameters={
                "action": "one of louder, quieter, set, mute, unmute, get",
                "percent": "a percentage from 0 to 150, for set",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=system_volume,
            required=("action",),
        ),
    ]
