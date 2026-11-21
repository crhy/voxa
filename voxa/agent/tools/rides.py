"""The ride_quote tool: open a ride provider with the destination pre-filled.

crhy/voxa#50 first step. It reads the price estimate and arrival time back from
the provider's own page and speaks them; it never requests a ride (the ordering,
scheduling and cancelling halves stay behind the undecided #43 spend rule), so
the tool is :data:`RiskLevel.READ_ONLY` and never clicks anything.
"""

from __future__ import annotations

from datetime import date

from voxa.agent import host
from voxa.agent.deals import detect_location
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.rides import describe_ride, parse_ride_request, provider_url, quote_from_text
from voxa.agent.tools.web import current_session
from voxa.config import ConfigStore

NO_LOCATION = "I couldn't tell where you are. Say \u201cI'm in <city>\u201d."


def ride_quote(args: dict[str, str]) -> ToolResult:
    settings = ConfigStore().load()
    home = settings.home_location or detect_location()
    req = parse_ride_request(args["request"], date.today(), home)
    if req is None:
        return ToolResult.failure("That doesn't look like a ride request.")
    pickup = req.pickup or home
    if not pickup:
        return ToolResult.failure(NO_LOCATION)
    url = provider_url(req, pickup)

    session = current_session()
    if session is None:
        # No controllable browser open: just open the provider so the rider can
        # read the quote themselves; we cannot capture its text here.
        host.spawn(["xdg-open", url])
        return ToolResult.success(f"{describe_ride(req)} I opened it so you can see the price.", detail=url)

    try:
        session.goto(url)
        page = session.text(4000)
    except Exception as exc:  # the provider page did not load
        return ToolResult.failure("The ride page did not open.", detail=str(exc))

    price, eta = quote_from_text(page)
    if price is None and eta is None:
        return ToolResult.success(f"{describe_ride(req)} The page showed no price or time yet.", detail=url)
    parts = [describe_ride(req)]
    if price:
        parts.append(f"The estimate is {price}")
    if eta:
        parts.append(f"It's {eta}")
    return ToolResult.success(". ".join(parts) + ".", detail=url)


def ride_tools() -> list[Tool]:
    return [
        Tool(
            name="ride_quote",
            description=(
                "Open a Lyft, Uber or taxi page with the destination pre-filled and read back "
                "the price estimate and arrival time, without ordering the ride."
            ),
            parameters={"request": "the whole ride request, as spoken"},
            risk=RiskLevel.READ_ONLY,
            handler=ride_quote,
            required=("request",),
        ),
    ]
