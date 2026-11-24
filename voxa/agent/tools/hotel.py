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
    "A traveller wants a hotel in {place}{when}. "
    "Based on the search results below, give exactly 3 numbered hotel options, "
    "one per line, each a hotel name, a price band (budget/mid-range/luxury) and "
    "where to book it, under 15 words each. No other text."
)

FALLBACK_OPTIONS = [
    "Compare rooms on a booking site like Booking.com or Expedia.",
    "Try the hotel's own website, which can be cheaper.",
    "Ask the front desk about a discount or a walk-in rate.",
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


def _hotel_options_handler(args: dict[str, str]) -> ToolResult:
    city = (args.get("city") or "").strip()
    check_in = (args.get("check_in") or "").strip()
    check_out = (args.get("check_out") or "").strip()
    guests = (args.get("guests") or "").strip()
    if not city:
        return ToolResult.failure("Which city or area should I find a hotel in?")
    query = f"hotel {city} {check_in} {check_out}"
    try:
        results = websearch.search(query)
    except Exception:  # noqa: BLE001 - network is outside our control
        results = []
    options: list[str] = []
    if results and ask_model is not None:
        snippets = "\n".join(f"{r.title}: {r.snippet}" for r in results[:5])
        when = " ".join(
            part
            for part in (
                f"from {check_in} to {check_out}" if check_in and check_out else check_in,
                f"{guests} guests" if guests else "",
            )
            if part
        )
        prompt = PROMPT.format(place=city, when=f" ({when})" if when else "")
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
    return ToolResult.success(
        f"Here are hotel options for {city}. I have not booked anything. Your options: {o1}; {o2}; {o3}."
    )


def hotel_tools() -> list[Tool]:
    return [
        Tool(
            name="hotel_options",
            description="When someone wants to find or book a hotel room, search the web and give 3 hotel options with a name, a price band and where to book, without reserving anything.",
            parameters={"city": "city or area", "check_in": "check-in date", "check_out": "check-out date", "guests": "number of guests"},
            risk=RiskLevel.READ_ONLY,
            handler=_hotel_options_handler,
            required=("city",),
        ),
    ]
