from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable

from voxa.agent.lists import add
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.lists import REACH_ERROR, _read, _write

log = logging.getLogger(__name__)

ask_model: Callable[[list[dict]], str] | None = None

PROMPT = (
    "Break the goal into 5 to 8 short steps. Reply with 5 to 8 numbered lines, "
    "one step per line, under 12 words each. No other text."
)


def _short_name(goal: str) -> str:
    words = goal.split()
    return " ".join(words[:4]).title()


def _parse_steps(reply: str) -> list[str]:
    steps: list[str] = []
    seen: set[str] = set()
    for line in (reply or "").splitlines():
        line = line.strip()
        found = re.match(r"^(?:\d+[.)]?|[-*])\s*(.*)$", line)
        if not found:
            continue
        text = found.group(1).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        steps.append(text)
        if len(steps) >= 8:
            break
    return steps


def _project_steps_handler(args: dict[str, str]) -> ToolResult:
    goal = (args.get("goal") or "").strip()
    if not goal:
        return ToolResult.failure("What goal should I break down?")
    if ask_model is None:
        return ToolResult.failure("I can't reach my brain right now.")
    try:
        reply = ask_model([{"role": "system", "content": PROMPT}, {"role": "user", "content": goal}])
    except Exception as exc:  # noqa: BLE001 - the model server is outside our control
        return ToolResult.failure("I can't reach my brain right now.", detail=str(exc))
    steps = _parse_steps(reply)
    if not steps:
        return ToolResult.failure("My brain did not give me usable steps.")
    name = _short_name(goal)
    try:
        content = _read()
        for step in steps:
            content = add(content, name, step)
        _write(content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(f"I broke {goal} into {len(steps)} steps and saved them as the {name} list. First: {steps[0]}.")


def project_steps_tools() -> list[Tool]:
    return [
        Tool(
            name="project_steps",
            description="Break a big goal into 5 to 8 short steps and save them as a named list.",
            parameters={"goal": "the goal to break into steps"},
            risk=RiskLevel.REVERSIBLE,
            handler=_project_steps_handler,
            required=("goal",),
        ),
    ]
