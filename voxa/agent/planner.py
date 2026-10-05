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
    "looks_like_browser_task",
    "system_prompt",
    "Plan",
    "PlanError",
    "parse_plan",
    "run_plan",
    "plan_browser_step",
    "parse_browser_step",
    "run_browser_task",
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


BROWSER_SITE_HINTS: tuple[str, ...] = (
    "on the website", "on the site", "on this page", "on that page",
    "on wikipedia", "on github", "on amazon", "online", "the web page", "website",
)

BROWSER_TASK_VERBS: tuple[str, ...] = (
    "find", "look up", "check", "search", "get", "read", "tell me", "show me",
    "download", "sign in", "log in", "fill", "book", "order", "buy", "compare",
)


def looks_like_browser_task(text: str) -> bool:
    """True when the request names a site or the web AND an action beyond opening it."""
    folded = text.casefold()
    has_site = any(hint in folded for hint in BROWSER_SITE_HINTS)
    if not has_site and re.search(r"\b[a-z0-9]+\.[a-z]{2,}\b", folded) is None:
        return False
    return any(re.search(rf"\b{re.escape(verb)}\b", folded) for verb in BROWSER_TASK_VERBS)


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


BROWSER_TOOLS = (
    "browse",
    "search_web",
    "click_on",
    "fill_field",
    "fill_and_submit",
    "search_site",
    "scroll",
    "go_back",
    "read_page",
    "wait_for",
)


def plan_browser_step(
    request: str,
    page_title: str,
    page_url: str,
    outline_lines: list[str],
    history: list[tuple[ToolCall, ToolResult]],
    registry: ToolRegistry,
) -> list[dict]:
    """Chat messages for ONE next step of a browsing task."""
    system = (
        "You are the browser planner for a desktop voice assistant. You decide ONE "
        "next action for a browsing task.\n"
        f"Available tools: {', '.join(BROWSER_TOOLS)}\n"
        f"Tool details:\n{registry.describe()}\n"
        "Reply with ONLY one JSON object, no prose and no code fence:\n"
        '  {"tool": "<name>", "args": {"<arg>": "<value>"}} for the next action, or\n'
        '  {"done": true, "say": "<one sentence result for the user>"} when finished.\n'
        "Use only the listed tools and argument names."
    )
    parts = [
        f"Request: {request}",
        f"Current page title: {page_title}",
        f"Current page URL: {page_url}",
    ]
    if outline_lines:
        parts.append("Page outline (one element per line):\n" + "\n".join(outline_lines))
    if history:
        hist_lines = [
            f"{step.tool} {step.args} -> {result.speech}" for step, result in history
        ]
        parts.append("Previous steps:\n" + "\n".join(hist_lines))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_browser_step(reply: str, registry: ToolRegistry) -> ToolCall | str:
    """Extract one browser action from a model reply; a sentence when done."""
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
    if data.get("done") is True:
        say = data.get("say", "")
        if not isinstance(say, str):
            say = str(say)
        return say
    tool = data.get("tool")
    if not isinstance(tool, str):
        raise PlanError('the reply must have a string "tool" or "done": true')
    if tool not in BROWSER_TOOLS and tool in registry.names():
        raise PlanError(f"browser steps may only use browser tools: {tool}")
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise PlanError('"args" must be an object')
    try:
        cleaned = registry.validate(tool, args)
    except ToolError as exc:
        raise PlanError(str(exc)) from exc
    return ToolCall(tool, cleaned)


def run_browser_task(
    request: str,
    ask_model: Callable[[list[dict]], str],
    session,
    registry: ToolRegistry,
    max_steps: int = 8,
    on_step: Callable[[ToolCall, ToolResult, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_caption: Callable[[ToolCall], None] | None = None,
) -> str:
    """Drive a browsing task: outline, ask, parse, run, repeat."""
    history: list[tuple[ToolCall, ToolResult]] = []
    last_title = ""
    # The task may be the first thing that touches the browser: start it if it is not running.
    ensure = getattr(session, "ensure", None)
    if callable(ensure):
        ensure()
    for _ in range(max_steps):
        if should_stop is not None and should_stop():
            return last_title
        outline = session.outline()
        outline_lines = [json.dumps(item, separators=(",", ":")) for item in outline]
        page_title = session.title()
        page_url = session.url()
        last_title = page_title
        messages = plan_browser_step(request, page_title, page_url, outline_lines, history, registry)
        reply = ask_model(messages)
        try:
            action = parse_browser_step(reply, registry)
        except PlanError:
            return "I got stuck on that page."
        if isinstance(action, str):
            return action
        if on_caption is not None:
            on_caption(action)
        started = time.perf_counter()
        result = registry.call(action.tool, action.args)
        ms = int((time.perf_counter() - started) * 1000)
        if on_step is not None:
            on_step(action, result, ms)
        history.append((action, result))
    return last_title
