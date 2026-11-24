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
    "A shopper wants these items {items}"
    "{store_clause}. Based on the search results below, list up to 3 numbered "
    "local grocery or food delivery services that could take this order, one per "
    "line, under 12 words each. No other text."
)

FALLBACK_SERVICES = [
    "A local grocery-delivery app that carries the store's items.",
    "The store's own delivery or pickup, if it offers one.",
    "A food-delivery app that stocks the same items.",
]

_ITEM_SPLIT = re.compile(r"\s*(?:,|&|\band\b|\bplus\b)\s*", re.IGNORECASE)


def _split_items(items: str) -> list[str]:
    """Normalize a spoken item list into individual items."""
    parts: list[str] = []
    for chunk in _ITEM_SPLIT.split(items):
        part = re.sub(r"\s+", " ", chunk).strip(" .,;")
        part = re.sub(r"^(?:some|a|an|the)\s+", "", part, flags=re.IGNORECASE)
        if part:
            parts.append(part)
    return parts


def _estimate_total(count: int) -> str:
    """A rough estimated-total band for the read-back, never a bill."""
    low, high = max(count, 1) * 5, max(count, 1) * 8
    return f"about ${low} to ${high}"


def _parse_services(reply: str) -> list[str]:
    services: list[str] = []
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
        services.append(text)
        if len(services) >= 3:
            break
    return services


def _grocery_order_handler(args: dict[str, str]) -> ToolResult:
    items = (args.get("items") or "").strip()
    store = (args.get("store") or "").strip()
    item_list = _split_items(items)
    if not item_list:
        return ToolResult.failure("What should I put on the order?")
    query = f"{store + ' ' if store else ''}grocery delivery"
    try:
        results = websearch.search(query)
    except Exception:  # noqa: BLE001 - network is outside our control
        results = []
    services: list[str] = []
    if results and ask_model is not None:
        snippets = "\n".join(f"{r.title}: {r.snippet}" for r in results[:5])
        prompt = PROMPT.format(
            items=", ".join(item_list),
            store_clause=f" from {store}" if store else "",
        )
        try:
            reply = ask_model(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": snippets},
                ]
            )
        except Exception:  # noqa: BLE001 - the model server is outside our control
            reply = ""
        services = _parse_services(reply)
    if not services:
        services = list(FALLBACK_SERVICES)
    order_for = f" from {store}" if store else ""
    listed = "; ".join(services[:3])
    return ToolResult.success(
        f"Your order read-back{order_for}: {', '.join(item_list)}. "
        f"Estimated total {_estimate_total(len(item_list))}. "
        f"Delivery services that could take it: {listed}."
    )


def groceries_tools() -> list[Tool]:
    return [
        Tool(
            name="grocery_order",
            description=(
                "When asked to order or deliver groceries, build a read-back of the items, "
                "a rough estimated total and up to 3 delivery services, without placing an order."
            ),
            parameters={"items": "the items to order", "store": "the store or delivery service, optional"},
            risk=RiskLevel.READ_ONLY,
            handler=_grocery_order_handler,
            required=("items",),
        ),
    ]
