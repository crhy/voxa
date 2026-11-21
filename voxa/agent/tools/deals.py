from __future__ import annotations

from datetime import date

from voxa.agent import host
from voxa.agent.cdp import CdpError
from voxa.agent.deal_results import parse_products, product_line, spoken_products
from voxa.agent.deal_results import compact, parse_results, spoken
from voxa.agent.deals import build_url, describe, detect_location, parse_request
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.web import get_session
from voxa.config import ConfigStore

NO_LOCATION = "I couldn't tell where you are. Say \u201cI'm in <city>\u201d."

_ORDINALS = ("Cheapest", "Second", "Third", "Fourth", "Fifth")


def set_session(session) -> None:
    """Swap the browser session these tools read; the web tools own it."""
    from voxa.agent.tools import web

    web.set_session(session)


def _kind_of_open_page(url: str) -> str:
    """Guess what kind of search an already open results page holds."""
    low = url.lower()
    if "flight" in low or "/travel/flights" in low:
        return "flight"
    if "hotel" in low:
        return "hotel"
    if "carsearch" in low or "car rental" in low or "car-rental" in low:
        return "rental_car"
    if "cars.com" in low or "shopping/results" in low:
        return "car_for_sale"
    return "product"


def _read_page(session) -> tuple[str, str]:
    """The page's text and its outline, as one string each."""
    body = session.text(6000)
    items = session.outline()
    outline_text = " ".join(
        str(item.get("text", "")) for item in items if isinstance(item, dict)
    )
    return body, outline_text


def read_deal_results(args: dict[str, str]) -> ToolResult:
    """Open a price or booking search and read the top three results aloud.

    A request that parses gets its own results page; a bare "read them back"
    reads the page the controlled browser already has open.
    """
    req = parse_request(args["request"], date.today())
    url = ""
    if req is not None:
        settings = ConfigStore().load()
        location = settings.home_location or detect_location()
        if not location:
            return ToolResult.failure(NO_LOCATION)
        url, _site = build_url(req, location)
    session = get_session()
    try:
        if url:
            session.ensure()
            session.goto(url)
        body, outline_text = _read_page(session)
        kind = req.kind if req is not None else _kind_of_open_page(session.url())
    except CdpError as exc:
        if url:
            return ToolResult.failure("The browser did not open that page.", detail=str(exc))
        return ToolResult.failure(
            "Nothing is open to read back. Ask for a search first.", detail=str(exc)
        )
    results = parse_results(body, kind) or parse_results(outline_text, kind)
    if not results:
        if req is None:
            return ToolResult.failure("Nothing on the page could be read back.")
        return ToolResult.success(
            f"{describe(req, location)} The page itself still needs your eyes.",
            detail=url,
        )
    speech = " ".join(
        f"{_ORDINALS[index]}: {spoken(entry)}." for index, entry in enumerate(results)
    )
    lines = [compact(entry) for entry in results]
    return ToolResult.success(speech, detail="\n".join([url, *lines]) if url else "\n".join(lines))


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
            name="deal_results",
            description="Open a price or booking search and read the top three results aloud.",
            parameters={"request": "the whole price or booking request, as spoken"},
            risk=RiskLevel.READ_ONLY,
            handler=read_deal_results,
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
