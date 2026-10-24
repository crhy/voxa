from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

ask_model: Callable[[list[dict]], str] | None = None

REACH_ERROR = "I could not reach your trips."

FILE_NAME = "Voxa Trips.md"

_DAY_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "weekend": 2,
}

PROMPT = (
    "Plan a {days}-day trip to {place}. Reply with {days} numbered lines, one per day, "
    "each under 20 words, like 'Day 1: ...'. No other text."
)


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


def _day_count(spoken: str) -> int:
    word = (spoken or "").strip().lower()
    if word in _DAY_WORDS:
        return _DAY_WORDS[word]
    return 3


def _place_title(spoken: str) -> str:
    words = (spoken or "").strip()
    if not words:
        return ""
    return " ".join(word.capitalize() for word in words.split())


def _section_header(place: str, days: int) -> str:
    return f"## {place} ({days} days)"


def _parse_day_lines(reply: str, days: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in reply.splitlines():
        text = raw.strip()
        if not text:
            continue
        day_match = re.match(r"(?i)^day\s*\d+\s*[:.\-)]?\s*", text)
        if day_match:
            text = text[day_match.end() :]
        elif text[0].isdigit():
            text = text.lstrip("0123456789").strip()
            text = text.lstrip(".:-) ").strip()
        elif text[0] in "-*":
            text = text.lstrip("-* ").strip()
        else:
            continue
        if not text:
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        lines.append(text)
        if len(lines) >= days:
            break
    return lines


def _find_section(content: str, place: str) -> tuple[int, int]:
    header = f"## {place} ("
    start = -1
    for index, line in enumerate(content.splitlines()):
        if line.startswith(header):
            start = index
            break
    if start < 0:
        return (-1, -1)
    end = len(content.splitlines())
    for index in range(start + 1, len(content.splitlines())):
        if content.splitlines()[index].startswith("## "):
            end = index
            break
    return (start, end)


def _trip_plan_handler(args: dict[str, str]) -> ToolResult:
    place = _place_title(args.get("place", ""))
    if not place:
        return ToolResult.failure("Where should I plan a trip?")
    days = _day_count(args.get("days", ""))
    if ask_model is None:
        return ToolResult.failure("I can't reach my brain right now.")
    try:
        reply = ask_model(
            [
                {"role": "system", "content": PROMPT.format(days=days, place=place)},
                {"role": "user", "content": f"Plan {days} days in {place}."},
            ]
        )
    except Exception:
        return ToolResult.failure("I couldn't plan the trip.")
    lines = _parse_day_lines(reply or "", days)
    if not lines:
        return ToolResult.failure("My brain didn't give me a usable plan.")
    try:
        content = _read()
        block = _section_header(place, days) + "\n" + "\n".join(lines) + "\n"
        _write((content.rstrip("\n") + "\n\n" + block) if content.strip() else block)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I planned {days} days in {place} and saved it in Voxa Trips. Day 1: {lines[0]}")


def _trip_read_handler(args: dict[str, str]) -> ToolResult:
    place = _place_title(args.get("place", ""))
    if not place:
        return ToolResult.failure("Which trip should I read?")
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    lines = content.splitlines()
    start, end = _find_section(content, place)
    if start < 0:
        return ToolResult.failure(f"I don't have a saved trip to {place}.")
    day_lines = [line for line in lines[start + 1 : end] if line.strip()]
    if not day_lines:
        return ToolResult.failure(f"Your trip to {place} has no days yet.")
    first = day_lines[0].strip()
    return ToolResult.success(f"Your trip to {place}: {first}.")


def _trip_forget_handler(args: dict[str, str]) -> ToolResult:
    place = _place_title(args.get("place", ""))
    if not place:
        return ToolResult.failure("Which trip should I forget?")
    try:
        content = _read()
        lines = content.splitlines()
        start, end = _find_section(content, place)
        if start < 0:
            return ToolResult.failure(f"I don't have a saved trip to {place}.")
        remaining = lines[:start] + lines[end:]
        _write("\n".join(remaining).strip("\n") + ("\n" if any(line.strip() for line in remaining) else ""))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I forgot your trip to {place}.")


def trip_tools() -> list[Tool]:
    return [
        Tool(
            name="trip_plan",
            description="Plan a short trip to a place and save one line per day in the Voxa Trips file.",
            parameters={"place": "where to go", "days": "how many days, e.g. three or weekend"},
            risk=RiskLevel.REVERSIBLE,
            handler=_trip_plan_handler,
            required=("place",),
        ),
        Tool(
            name="trip_read",
            description="Read back the saved day lines for a trip.",
            parameters={"place": "which trip"},
            risk=RiskLevel.READ_ONLY,
            handler=_trip_read_handler,
            required=("place",),
        ),
        Tool(
            name="trip_forget",
            description="Remove a saved trip from the Voxa Trips file.",
            parameters={"place": "which trip"},
            risk=RiskLevel.REVERSIBLE,
            handler=_trip_forget_handler,
            required=("place",),
        ),
    ]
