from __future__ import annotations

import logging
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.sysinfo import (
    parse_df,
    parse_meminfo,
    parse_upower,
    spoken_battery,
    spoken_date,
    spoken_disk,
    spoken_memory,
    spoken_time,
)

log = logging.getLogger("voxa.agent.tools.system")

_FAILURE = "I could not read that from this computer."


_SIMULATED = {
    ("date", "+%H %M"): "12 00\n",
    ("date", "+%A|%-d|%B|%Y"): "Monday|1|January|2024\n",
    ("df", "-B1", "--output=avail,size"): "        Avail         1B-blocks\n128849018880 536870912000\n",
    ("cat", "/proc/meminfo"): "MemTotal: 32000000 kB\nMemAvailable: 24000000 kB\n",
    ("upower", "-e"): "",
}


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        if command[:2] == ["sh", "-c"]:
            canned = "/home/me\n" if "$HOME" in command[2] else ""
        else:
            canned = _SIMULATED.get(tuple(command), _SIMULATED.get(tuple(command[:3]), ""))
        return subprocess.CompletedProcess(command, 0, canned, "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def tell_time(args: dict[str, str]) -> ToolResult:
    try:
        hour, minute = _run(["date", "+%H %M"], check=False).stdout.split()
        return ToolResult.success(spoken_time(int(hour), int(minute)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def tell_date(args: dict[str, str]) -> ToolResult:
    try:
        weekday, day, month, year = _run(["date", "+%A|%-d|%B|%Y"], check=False).stdout.split("|")
        return ToolResult.success(spoken_date(weekday, int(day), month, int(year)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def disk_space(args: dict[str, str]) -> ToolResult:
    try:
        home = _run(["sh", "-c", "echo $HOME"], check=False).stdout.strip()
        free, total = parse_df(_run(["df", "-B1", "--output=avail,size", home], check=False).stdout)
        return ToolResult.success(spoken_disk(free, total))
    except Exception:
        return ToolResult.failure(_FAILURE)


def memory_free(args: dict[str, str]) -> ToolResult:
    try:
        available, total = parse_meminfo(_run(["cat", "/proc/meminfo"], check=False).stdout)
        return ToolResult.success(spoken_memory(available, total))
    except Exception:
        return ToolResult.failure(_FAILURE)


def battery_level(args: dict[str, str]) -> ToolResult:
    try:
        found = _run(["sh", "-c", "upower -e | grep -m1 BAT"], check=False).stdout.strip()
        if not found:
            return ToolResult.success(spoken_battery(None, ""))
        percent, state = parse_upower(_run(["upower", "-i", found], check=False).stdout)
        return ToolResult.success(spoken_battery(percent, state))
    except Exception:
        return ToolResult.failure(_FAILURE)


def system_tools() -> list[Tool]:
    return [
        Tool(
            name="tell_time",
            description="Say the current time on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=tell_time,
        ),
        Tool(
            name="tell_date",
            description="Say today's date on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=tell_date,
        ),
        Tool(
            name="disk_space",
            description="Say how much disk space is free on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=disk_space,
        ),
        Tool(
            name="memory_free",
            description="Say how much memory is free on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=memory_free,
        ),
        Tool(
            name="battery_level",
            description="Say how much battery is left on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=battery_level,
        ),
    ]
