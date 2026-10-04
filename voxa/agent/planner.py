"""Turn an unrouted spoken request into a short sequence of tool calls.

Pure Python: no GTK, no host access. The local model is asked for a JSON plan
which is validated against the tool registry before anything runs.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from voxa.agent.intents import ToolCall
from voxa.agent.registry import ToolError, ToolRegistry
from voxa.agent.result import ToolResult
from voxa.agent.tools.typing import KEYS

__all__ = [
    "COMMAND_VERBS",
    "looks_like_command",
    "system_prompt",
    "Plan",
    "PlanError",
    "parse_plan",
    "run_plan",
]

COMMAND_VERBS: frozenset[str] = frozenset(
    {
        "open", "close", "quit", "launch", "start", "stop", "play", "pause",
        "resume", "skip", "mute", "unmute", "turn", "set", "put", "switch",
        "go", "show", "bring", "focus", "type", "write", "press", "send",
        "search", "find", "look", "copy", "paste", "save", "select", "scroll",
        "minimize", "maximize", "move", "check", "compose", "reply", "read",
        "dictate", "enter", "undo",
    }
)

_LEADING = re.compile(
    r"^(?:please|can you|could you|hey|okay|ok)\s*[,:]?\s*",
    re.IGNORECASE,
)


def looks_like_command(text: str) -> bool:
    """True when the request starts with an action verb, not a question."""
    text = text.strip()
    while True:
        match = _LEADING.match(text)
        if not match:
            break
        text = text[match.end():]
    text = text.strip().strip("\"'.,!?")
    first = re.match(r"[a-z]+", text.casefold())
    if first is not None and first.group(0) in COMMAND_VERBS:
        return True
    from voxa.agent.intents import is_compound

    return is_compound(text)


def system_prompt(registry: ToolRegistry) -> str:
    """The instruction the model gets when it must plan a sequence of tools."""
    return (
        "You are the planner for a desktop voice assistant. You turn one spoken "
        "request into a short sequence of tool calls.\n"
        f"Available tools:\n{registry.describe()}\n"
        f"press_key keys: {', '.join(sorted(KEYS))}\n"
        "Reply with ONLY one JSON object, no prose and no code fence:\n"
        '  {"steps": [{"tool": "<name>", "args": {"<arg>": "<value>"}}], '
        '"say": "<one short sentence to speak>"}\n'
        "Use 1 to 6 steps, in order. Use only the listed tools and argument "
        "names. If no tool fits, reply "
        '{"steps": [], "say": "<a short honest answer or what you would need>"}.\n'
        "Examples:\n"
        '  "open gmail and start a new message" -> {"steps": [{"tool": "open_site", '
        '"args": {"name": "gmail"}}, {"tool": "compose_gmail", "args": {}}], '
        '"say": "Opening Gmail and starting a message."}\n'
        '  "put on some jazz and turn it up" -> {"steps": [{"tool": "play_youtube", '
        '"args": {"query": "jazz"}}, {"tool": "press_key", "args": {"key": "volume up"}}], '
        '"say": "Playing jazz and turning it up."}'
    )


@dataclass(frozen=True, slots=True)
class Plan:
    steps: tuple[ToolCall, ...]
    say: str


class PlanError(Exception):
    """Raised when the model's reply cannot be turned into a valid plan."""


def parse_plan(reply: str, registry: ToolRegistry, max_steps: int = 6) -> Plan:
    """Extract and validate the JSON plan from a model reply."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end < start:
        raise PlanError("the reply does not contain a JSON object")
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError as exc:
        raise PlanError(f"invalid JSON in the reply: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanError("the reply is not a JSON object")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise PlanError('"steps" is not a list')
    if len(raw_steps) > max_steps:
        raise PlanError(f"too many steps: {len(raw_steps)}")
    steps: list[ToolCall] = []
    for step in raw_steps:
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            raise PlanError('a step must be an object with a string "tool"')
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise PlanError('"args" must be an object')
        try:
            cleaned = registry.validate(step["tool"], args)
        except ToolError as exc:
            raise PlanError(str(exc)) from exc
        steps.append(ToolCall(step["tool"], cleaned))
    say = data.get("say", "")
    if not isinstance(say, str):
        say = str(say)
    return Plan(tuple(steps), say)


def run_plan(
    plan: Plan,
    registry: ToolRegistry,
    on_step: Callable[[ToolCall, ToolResult, int], None] | None = None,
    pause_seconds: float = 0.8,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run the steps in order, stopping at the first failure; return the sentence to speak."""
    last_speech = ""
    for index, step in enumerate(plan.steps):
        if index:
            sleep(pause_seconds)
        started = time.perf_counter()
        result = registry.call(step.tool, step.args)
        ms = int((time.perf_counter() - started) * 1000)
        if on_step is not None:
            on_step(step, result, ms)
        if not result.ok:
            return result.speech
        if result.speech:
            last_speech = result.speech
    return plan.say or last_speech or "Done."
