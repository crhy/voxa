from __future__ import annotations

from datetime import date

from voxa.agent import host
from voxa.agent.cdp import CdpError
from voxa.agent.deal_results import parse_products, product_line, spoken_products
from voxa.agent.deals import build_url, describe, detect_location, parse_request
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.web import get_session
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


def read_products(args: dict[str, str]) -> ToolResult:
    """Open a product search and say the top three products aloud, cheapest first.

    The page is read through the browser session the web tools own, so nothing here
    spends money or touches checkout: it is READ_ONLY, and the #43 spend rule stays
    out of it.
    """
    req = parse_request(args["request"], date.today())
    if req is None or req.kind != "product":
        return ToolResult.failure("That doesn't look like a product or price search.")
    settings = ConfigStore().load()
    location = settings.home_location or detect_location()
    if not location:
        return ToolResult.failure(NO_LOCATION)
    url, _site = build_url(req, location)
    session = get_session()
    try:
        session.ensure()
        session.goto(url)
        body = session.text(6000)
        outline_text = " ".join(
            str(item.get("text", "")) for item in session.outline() if isinstance(item, dict)
        )
    except CdpError as exc:
        return ToolResult.failure("The browser did not open that page.", detail=str(exc))
    items = parse_products(body) or parse_products(outline_text)
    if not items:
        return ToolResult.success(
            f"{describe(req, location)} The page itself still needs your eyes.",
            detail=url,
        )
    lines = [product_line(entry, req) for entry in items]
    return ToolResult.success(spoken_products(items, req), detail="\n".join(lines))


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
            name="product_results",
            description="Open a product search and read the top three products aloud with price and rating.",
            parameters={"request": "the whole price or product request, as spoken"},
            risk=RiskLevel.READ_ONLY,
            handler=read_products,
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
