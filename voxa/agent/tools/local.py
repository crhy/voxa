"""Things near the user: the local weather, local news and what is on tonight (issue #101)."""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date
from xml.etree import ElementTree

from voxa.agent import host, weather
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
    """The place the user saved ("I'm in …"), else the one worked out from the network
    — at most once a day, kept in the settings so Voxa keeps answering offline after."""
    if locate is not None:
        return locate()
    settings = ConfigStore().load()
    if settings.home_location:
        return settings.home_location
    if settings.location_detected and settings.location_detected_on == date.today().isoformat():
        return settings.location_detected
    found = detect_location()
    if found:
        settings.location_detected = found
        settings.location_detected_on = date.today().isoformat()
        ConfigStore().save(settings)
    return found


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
    asked = (args.get("place") or "").strip()
    place = asked or where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    city = _city(place)
    spoken = ""
    try:
        # Open-Meteo needs no key and no account: geocode the name, then one forecast call.
        coords = weather.latlon(place)
        if coords is not None:
            lat, lon, fahrenheit = coords[0], coords[1], weather.in_us(*coords)
        else:
            geo = weather.parse_geocode((fetch or _fetch)(weather.geocode_url(place)))
            if geo is None:
                raise ValueError(f"Open-Meteo has no place called {place}")
            lat, lon, fahrenheit = geo
        spoken = weather.parse_forecast((fetch or _fetch)(weather.forecast_url(lat, lon, fahrenheit)), city)
        if not spoken:
            raise ValueError("unexpected forecast answer")
    except Exception:  # noqa: BLE001 - no forecast service: the page still opens
        spoken = ""
    if spoken:
        # The forecast answers on the exchange card; nothing to open.
        return ToolResult.success(spoken)
    url = "https://www.google.com/search?q=" + urllib.parse.quote(f"weather {place}")
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here is the weather for {city}.", detail=url)


def _news_page_url(city: str) -> str:
    return "https://news.google.com/search?q=" + urllib.parse.quote(f"{city} news")


def _headlines(xml_text: str) -> list[str]:
    """The item titles of a Google News RSS search, newest first; the feed's own title first."""
    root = ElementTree.fromstring(xml_text)
    titles = []
    for title in root.iter("title"):
        text = (title.text or "").strip()
        if text:
            titles.append(text)
    return titles[1:]


def local_news(args: dict[str, str]) -> ToolResult:
    place = where()
    if not place:
        return ToolResult.failure(NO_LOCATION)
    city = _city(place)
    page = _news_page_url(city)
    if (args.get("page") or "").strip():
        # "Open the news": the page, without reading the headlines again.
        if not _open(page):
            return ToolResult.failure("I couldn't open the browser.", detail=page)
        return ToolResult.success(f"The local news for {city} is open.", detail=page)
    try:
        rss_url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(f"{city} news")
        rss = (fetch or _fetch)(rss_url)
        top = _headlines(rss)[:3]
        if not top:
            raise ValueError("no headlines in the feed")
    except Exception:  # noqa: BLE001 - no RSS feed: the news page still opens
        if not _open(page):
            return ToolResult.failure("I couldn't open the browser.", detail=page)
        return ToolResult.success(f"Here is the local news for {city}.", detail=page)
    return ToolResult.success(f"Here are the top local news headlines for {city}: {'; '.join(top)}.", detail=page)


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
             parameters={"place": "optional: a town or city, or a latitude, longitude; empty for the user's own"},
             risk=RiskLevel.REVERSIBLE, handler=local_weather),
        Tool(name="where_am_i", description="Say which town or city the user is in.", parameters={},
             risk=RiskLevel.READ_ONLY, handler=where_am_i),
        Tool(name="local_news", description="Read the top three local news headlines for where the user is.",
             parameters={"page": "set to open the news page instead of reading the headlines"},
             risk=RiskLevel.REVERSIBLE, handler=local_news),
        Tool(name="local_events", description="Show what is on near the user: live music, concerts, events.",
             parameters={"what": "what to look for, e.g. live music", "when": "optional: tonight, this weekend"},
             risk=RiskLevel.REVERSIBLE, handler=local_events),
    ]
