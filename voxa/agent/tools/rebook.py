from __future__ import annotations

import logging
import re
from collections.abc import Callable

from voxa import websearch
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

ask_model: Callable[[list[dict]], str] | None = None

PROMPT = (
    "A traveller's {what} to {place} was {problem}. "
    "Based on the search results below, give exactly 3 numbered rebooking options, "
    "one per line, under 15 words each. No other text."
)

FALLBACK_OPTIONS = [
    "Call the carrier and ask them to help you rebook.",
    "Rebook in the carrier's app or website.",
    "Ask about a refund or a hotel voucher.",
]


def _parse_options(reply: str) -> list[str]:
    options: list[str] = []
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
        options.append(text)
        if len(options) >= 3:
            break
    return options


def _travel_options_handler(args: dict[str, str]) -> ToolResult:
    what = (args.get("what") or "").strip()
    place = (args.get("place") or "").strip()
    problem = (args.get("problem") or "").strip()
    if not what or not place or not problem:
        return ToolResult.failure("What happened and where are you going?")
    query = f"{what} to {place} {problem} rebooking options"
    try:
        results = websearch.search(query)
    except Exception:  # noqa: BLE001 - network is outside our control
        results = []
    options: list[str] = []
    if results and ask_model is not None:
        snippets = "\n".join(f"{r.title}: {r.snippet}" for r in results[:5])
        prompt = PROMPT.format(what=what, place=place, problem=problem)
        try:
            reply = ask_model([
                {"role": "system", "content": prompt},
                {"role": "user", "content": snippets},
            ])
        except Exception:  # noqa: BLE001 - the model server is outside our control
            reply = ""
        options = _parse_options(reply)
    if not options:
        options = list(FALLBACK_OPTIONS)
    o1, o2, o3 = options[0], options[1], options[2].rstrip(".")
    return ToolResult.success(f"Sorry about your {what}. Your options: {o1}; {o2}; {o3}.")


def rebook_tools() -> list[Tool]:
    return [
        Tool(
            name="travel_options",
            description="When a flight, train, bus or connection is cancelled, delayed or missed, search the web and give 3 rebooking options.",
            parameters={"what": "flight/train/bus/connection", "place": "destination", "problem": "cancelled/delayed/missed"},
            risk=RiskLevel.READ_ONLY,
            handler=_travel_options_handler,
            required=("what", "place", "problem"),
        ),
    ]
