from __future__ import annotations

from datetime import date

from voxa.agent import host
from voxa.agent.deals import build_url, describe, detect_location, parse_request
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.config import ConfigStore

NO_LOCATION = "I couldn't tell where you are. Say \u201cI'm in <city>\u201d."


def find_deal(args: dict[str, str]) -> ToolResult:
    req = parse_request(args["request"], date.today())
    if req is None:
        return ToolResult.failure("That doesn't look like a price or booking search.")
    settings = ConfigStore().load()
    location = settings.home_location or detect_location()
    if not location:
        return ToolResult.failure(NO_LOCATION)
    url, _site = build_url(req, location)
    host.spawn(["xdg-open", url])
    return ToolResult.success(describe(req, location), detail=url)


def set_location(args: dict[str, str]) -> ToolResult:
    settings = ConfigStore().load()
    city = args["city"].strip()
    settings.home_location = city
    ConfigStore().save(settings)
    if city:
        return ToolResult.success(f"Got it. I'll use {city} as my location.")
    return ToolResult.success("Okay, I'll figure out where I am from the web.")


def deal_tools() -> list[Tool]:
    return [
        Tool(
            name="find_deal",
            description="Find the cheapest ticket, room, car or product and open the search page.",
            parameters={"request": "the whole price or booking request, as spoken"},
            risk=RiskLevel.REVERSIBLE,
            handler=find_deal,
            required=("request",),
        ),
        Tool(
            name="set_location",
            description="Set or clear the user's home location override.",
            parameters={"city": "the city to use, empty to clear the override"},
            risk=RiskLevel.REVERSIBLE,
            handler=set_location,
            required=("city",),
        ),
    ]
