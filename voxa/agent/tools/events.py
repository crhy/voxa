from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import lists as list_tools
from voxa.agent.tools.lists import add

log = logging.getLogger(__name__)

ask_model: Callable[[list[dict]], str] | None = None

PROMPT = (
    "Plan a {event}. Reply with 6 numbered lines, one checklist item per line, "
    "under 12 words each. No other text."
)


def _parse_items(reply: str, limit: int = 6) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in reply.splitlines():
        text = raw.strip()
        if not text:
            continue
        if re.match(r"(?i)^\s*\d+\s*[.):\-]?\s*", text):
            text = re.sub(r"(?i)^\s*\d+\s*[.):\-]?\s*", "", text)
        elif text[0] in "-*":
            text = text.lstrip("-* ").strip()
        else:
            continue
        text = text.strip()
        if not text:
            continue
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _event_plan_handler(args: dict[str, str]) -> ToolResult:
    event = (args.get("event") or "").strip()
    if not event:
        return ToolResult.failure("What event should I plan?")
    when = (args.get("when") or "").strip()
    if ask_model is None:
        return ToolResult.failure("I can't reach my brain right now.")
    try:
        reply = ask_model(
            [
                {"role": "system", "content": PROMPT.format(event=event)},
                {"role": "user", "content": f"Plan a {event}."},
            ]
        )
    except Exception:
        return ToolResult.failure("I couldn't plan the event.")
    items = _parse_items(reply or "", 6)
    if not items:
        return ToolResult.failure("My brain didn't give me a usable plan.")
    if when:
        items.insert(0, f"Date: {when}")
    name = f"{event.title()} Plan"
    try:
        content = list_tools._read()
        for item in items:
            content = add(content, name, item)
        list_tools._write(content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure("I could not reach your lists.")
    return ToolResult.success(
        f"I made a {event} plan with {len(items)} things to do and saved it as the {name} list. First: {items[0]}."
    )


def event_tools() -> list[Tool]:
    return [
        Tool(
            name="event_plan",
            description="Plan an event or gathering: ask the brain for a short checklist and save it as a named list.",
            parameters={"event": "the event, e.g. birthday party or barbecue", "when": "when it is, may be empty"},
            risk=RiskLevel.REVERSIBLE,
            handler=_event_plan_handler,
            required=("event",),
        ),
    ]
