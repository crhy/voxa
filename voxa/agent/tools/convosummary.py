from __future__ import annotations

import logging
import re
from collections.abc import Callable

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

REACH_ERROR = "I can't reach my brain right now."

get_history: Callable[[], list[dict]] | None = None
ask_model: Callable[[list[dict]], str] | None = None

PROMPT = (
    "Read the conversation. Reply with at most 3 lines that start with 'Decision:' "
    "and at most 3 lines that start with 'To do:', each under 12 words. No other text."
)

_MARKER = r"^\s*(?:[-*]|\d+[.)])?\s*"
_DECISION = re.compile(_MARKER + r"decision\s*[:.\-]\s*(.+)$", re.IGNORECASE)
_TODO = re.compile(_MARKER + r"to[- ]?do\s*[:.\-]\s*(.+)$", re.IGNORECASE)


def _clean(text: str) -> str:
    t = text.strip().strip(".-")
    if t:
        t = t[0].upper() + t[1:]
    return t


def _parse(reply: str) -> tuple[list[str], list[str]]:
    decisions: list[str] = []
    todos: list[str] = []
    for line in reply.splitlines():
        d = _DECISION.match(line)
        if d:
            text = _clean(d.group(1))
            if text and text.lower() not in {x.lower() for x in decisions}:
                decisions.append(text)
            continue
        t = _TODO.match(line)
        if t:
            text = _clean(t.group(1))
            if text and text.lower() not in {x.lower() for x in todos}:
                todos.append(text)
    return decisions[:3], todos[:3]


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _spoken(decisions: list[str], todos: list[str]) -> str:
    parts: list[str] = []
    if decisions:
        parts.append(f"We decided {_join(decisions)}.")
    if todos:
        parts.append(f"To do: {_join(todos)}.")
    return " ".join(parts)


def _conversation_summary_handler(args: dict[str, str]) -> ToolResult:
    history = get_history() if get_history is not None else []
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-30:]
    user_turns = [m for m in turns if m.get("role") == "user"]
    if len(user_turns) < 2:
        return ToolResult.success("We haven't talked about much yet.")
    if ask_model is None:
        return ToolResult.failure(REACH_ERROR)
    transcript = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in turns)
    try:
        reply = ask_model([{"role": "system", "content": PROMPT}, {"role": "user", "content": transcript}])
    except Exception:
        return ToolResult.failure(REACH_ERROR)
    decisions, todos = _parse(reply or "")
    if not decisions and not todos:
        return ToolResult.failure("I couldn't pull out any clear decisions from our chat.")
    return ToolResult.success(_spoken(decisions, todos))


def convosummary_tools() -> list[Tool]:
    return [
        Tool(
            name="conversation_summary",
            description="Summarise the current conversation as decisions and action items.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_conversation_summary_handler,
        ),
    ]
