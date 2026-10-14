"""Things near the user: the local weather, local news and what is on tonight (issue #101)."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable

from voxa.agent import briefing, host
from voxa.agent.deals import detect_location
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.config import ConfigStore

NO_LOCATION = "I don't know where you are yet. Tell me, for example: I'm in Portland, Oregon."

# Replaced in tests; the window does not need to set these.
locate: Callable[[], str] | None = None
fetch: Callable[[str], str] | None = None


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Voxa"})
    with urllib.request.urlopen(request, timeout=6) as response:  # noqa: S310 - fixed https address
        return response.read().decode("utf-8", "replace")


def where() -> str:
    """The place the user saved ("I'm in …"), else the one worked out from the network."""
    if locate is not None:
        return locate()
    return ConfigStore().load().home_location or detect_location()


def _city(place: str) -> str:
    return place.split(",")[0].strip()


def _open(url: str) -> bool:
    return host.spawn(["xdg-open", url]) is not False


def where_am_i(args: dict[str, str]) -> ToolResult:
    place = where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    return ToolResult.success(f"You are in {place}.", detail=place)


def local_weather(args: dict[str, str]) -> ToolResult:
    place = (args.get("place") or "").strip() or where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    url = "https://www.google.com/search?q=" + urllib.parse.quote(f"weather {place}")
    spoken = ""
    try:
        text = (fetch or _fetch)(f"https://wttr.in/{urllib.parse.quote(place)}?format=j1")
        spoken = briefing.parse_weather(text, briefing.uses_fahrenheit(text))
        if spoken and (args.get("place") or "").strip():
            # The forecast service names its nearest weather station ("Shikinajima" for Tokyo):
            # answer with the place that was asked for.
            spoken = re.sub(r"^In .+? it is ", f"In {_city(place)} it is ", spoken, count=1)
    except Exception:  # noqa: BLE001 - no forecast service: the page still opens
        spoken = ""
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(spoken or f"Here is the weather for {_city(place)}.", detail=url)


def local_news(args: dict[str, str]) -> ToolResult:
    place = where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    # YouTube's "live now" filter, so a station that is on air comes first.
    query = urllib.parse.quote(f"{_city(place)} local news live")
    url = f"https://www.youtube.com/results?search_query={query}&sp=EgJAAQ%253D%253D"
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here is the local news for {_city(place)}.", detail=url)


def local_events(args: dict[str, str]) -> ToolResult:
    place = where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    what = (args.get("what") or "events").strip()
    when = (args.get("when") or "").strip()
    wanted = " ".join(part for part in (what, when) if part)
    url = "https://www.google.com/search?q=" + urllib.parse.quote(f"{wanted} near {place}")
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here is {wanted} near {_city(place)}.", detail=url)


def local_tools() -> list[Tool]:
    return [
        Tool(name="local_weather", description="Say and show the weather where the user is, or in a named place.",
             parameters={"place": "optional: a town or city; empty for the user's own"},
             risk=RiskLevel.REVERSIBLE, handler=local_weather),
        Tool(name="where_am_i", description="Say which town or city the user is in.", parameters={},
             risk=RiskLevel.READ_ONLY, handler=where_am_i),
        Tool(name="local_news", description="Open live local news for where the user is.", parameters={},
             risk=RiskLevel.REVERSIBLE, handler=local_news),
        Tool(name="local_events", description="Show what is on near the user: live music, concerts, events.",
             parameters={"what": "what to look for, e.g. live music", "when": "optional: tonight, this weekend"},
             risk=RiskLevel.REVERSIBLE, handler=local_events),
    ]
