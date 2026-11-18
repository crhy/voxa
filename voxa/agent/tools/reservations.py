"""Restaurant reservation tools: say the booking back, then open its page and pre-fill it.

crhy/voxa#47's first step. `reservation_fill` types the party, day, time and restaurant
into the fields the booking page offers and stops there: submitting the booking is the
spend rule's call (crhy/voxa#43), so this module never presses Enter and never clicks
a Book/Confirm button.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from voxa.agent.cdp import CdpError
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.reservations import (
    booking_url,
    describe_reservation,
    form_values,
    missing_details,
    parse_reservation,
    spoken_site,
)
from voxa.agent.result import ToolResult
from voxa.agent.tools.web import get_session
from voxa.config import ConfigStore

log = logging.getLogger("voxa.agent.tools.reservations")

NOT_A_BOOKING = "That doesn't look like a table booking. Say \u201cbook a table for four at Luigi's on Friday at seven\u201d."
ASK_FOR = "book a table for four at Luigi's on Friday at seven"

# What a booking page calls the field a diner would call it.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Party size": ("party size", "party", "number of guests", "guests", "people", "covers", "size"),
    "Date": ("date", "booking date", "reservation date", "day", "when"),
    "Time": ("time", "time of day", "reservation time", "hour"),
    "Restaurant": ("restaurant", "restaurant name", "name", "venue", "establishment"),
}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def _spoken_list(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _offered_field(label: str, outline: list) -> str:
    """The page's own label for the field the diner would call `label`, or "" when it has none."""
    wanted = _norm(label)
    keys = [wanted] + [_norm(word) for word in _SYNONYMS.get(label, ())]
    for item in outline or []:
        if not isinstance(item, dict) or item.get("kind") != "field":
            continue
        shown = str(item.get("text") or item.get("placeholder") or "").strip()
        hay = _norm(shown)
        if not hay:
            continue
        for key in keys:
            if len(key) < 3:
                continue
            if hay == key or key in hay or hay in key:
                return shown
    return ""


def _location() -> str:
    """The city the user set with "I'm in X"; read only, never written."""
    try:
        return ConfigStore().load(migrate=False).home_location.strip()
    except Exception:
        return ""


def reservation_plan(args: dict[str, str]) -> ToolResult:
    today = date.today()
    res = parse_reservation(args["request"], today)
    if res is None:
        return ToolResult.failure(NOT_A_BOOKING)
    speech = describe_reservation(res, today)
    gaps = missing_details(res)
    if gaps:
        speech += f" I don't know {_spoken_list(gaps)} yet. Try \u201c{ASK_FOR}\u201d."
    detail = "\n".join(
        [
            f"Booking page: {booking_url(res, _location())}",
            "Values: " + ", ".join(f"{label}={value}" for label, value in form_values(res)),
            "Not said: " + ", ".join(gaps) if gaps else "",
        ]
    )
    return ToolResult.success(speech, detail=detail)


def reservation_fill(args: dict[str, str]) -> ToolResult:
    today = date.today()
    res = parse_reservation(args["request"], today)
    if res is None:
        return ToolResult.failure(NOT_A_BOOKING)
    url = booking_url(res, _location())
    if not url:
        return ToolResult.failure(NOT_A_BOOKING)
    session = get_session()
    try:
        session.ensure()
        session.goto(url)
        outline = session.outline()
    except CdpError as exc:
        return ToolResult.failure("The browser did not open that page.", detail=str(exc))

    filled: list[str] = []
    filled_detail: list[str] = []
    skipped: list[str] = []
    for label, value in form_values(res):
        field = _offered_field(label, outline)
        if not field:
            skipped.append(label)
            continue
        try:
            ok = session.type_into(field, value)
        except CdpError as exc:
            return ToolResult.failure(
                f"I filled {_spoken_list(filled)} before the page stopped answering.",
                detail=str(exc),
            )
        if not ok:
            skipped.append(label)
            continue
        filled.append(label)
        filled_detail.append(f"{label} -> {field}={value}")

    site = spoken_site(res)
    if filled:
        speech = f"Filled {_spoken_list(filled)} on {site}."
    else:
        speech = f"I opened the booking page, but it offered none of the fields I had: {_spoken_list(skipped)}."
    if skipped and filled:
        speech += f" It did not offer {_spoken_list(skipped)}."
    speech += " Nothing was submitted."
    detail = "\n".join(
        [
            describe_reservation(res, today),
            f"Page: {url}",
            "Filled: " + ", ".join(filled_detail) if filled_detail else "",
            "Not offered: " + ", ".join(skipped) if skipped else "",
        ]
    )
    return ToolResult.success(speech, detail=detail)


def reservation_tools() -> list[Tool]:
    return [
        Tool(
            name="reservation_plan",
            description="Say back a restaurant table booking: party size, day, time and restaurant.",
            parameters={"request": "the whole reservation request, as spoken"},
            risk=RiskLevel.READ_ONLY,
            handler=reservation_plan,
            required=("request",),
        ),
        Tool(
            name="reservation_fill",
            description=(
                "Open the restaurant's booking page and type the party size, date, time and restaurant "
                "into the fields it offers, without submitting the booking."
            ),
            parameters={"request": "the whole reservation request, as spoken"},
            risk=RiskLevel.REVERSIBLE,
            handler=reservation_fill,
            required=("request",),
        ),
    ]
