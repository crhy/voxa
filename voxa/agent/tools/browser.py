from __future__ import annotations

import logging
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable

import voxa.simulation
from voxa.agent import browsercheck, host
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.spoken_text import site_label
from voxa.agent.youtube import play

log = logging.getLogger("voxa.agent.tools.browser")

SITES: dict[str, str] = {
    "gmail": "https://mail.google.com/mail/u/0/#inbox",
    "google mail": "https://mail.google.com/mail/u/0/#inbox",
    "youtube": "https://www.youtube.com/",
    "youtube music": "https://music.youtube.com/",
    "google calendar": "https://calendar.google.com/",
    "google drive": "https://drive.google.com/",
    "google docs": "https://docs.google.com/",
    "google maps": "https://www.google.com/maps",
    "github": "https://github.com/",
    "wikipedia": "https://en.wikipedia.org/",
}

GMAIL_COMPOSE = "https://mail.google.com/mail/u/0/?view=cm&fs=1&tf=1"

_VIDEO_ID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')


def site_url(name: str) -> str | None:
    """Look up a spoken site name, tolerating 'the ...' and '... website'."""
    key = name.casefold().strip()
    if key.startswith("the "):
        key = key[4:].strip()
    for suffix in (" website", " dot com"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return SITES.get(key)


def normalize_url(text: str) -> str | None:
    """Return an http(s) URL, or None for anything unsafe or malformed."""
    url = text.strip()
    if not url or " " in url or "." not in url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    return f"https://{url}"


def youtube_search_url(query: str, music: bool = False) -> str:
    quoted = urllib.parse.quote(query)
    if music:
        return f"https://music.youtube.com/search?q={quoted}"
    return f"https://www.youtube.com/results?search_query={quoted}"


def first_video_id(page: str) -> str | None:
    match = _VIDEO_ID_RE.search(page)
    return match.group(1) if match else None


def _default_fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:  # noqa: S310 - https URL
        return response.read().decode("utf-8", errors="replace")


def youtube_play_url(query: str, fetch: Callable[[str], str] | None = None) -> str:
    """Resolve a query to a single watch URL, falling back to the search page."""
    search = youtube_search_url(query)
    fetcher = fetch or _default_fetch
    try:
        video_id = first_video_id(fetcher(search))
    except Exception:
        log.debug("could not resolve YouTube video for %r", query)
        return search
    if not video_id:
        return search
    return f"https://www.youtube.com/watch?v={video_id}"


NO_BROWSER_MESSAGE = (
    'There\'s no web browser on this computer yet. Say "install Firefox" and I\'ll set one up.'
)


def _host_run(command: list[str]) -> tuple[int, str]:
    """Run a host query command and return (returncode, stdout)."""
    if voxa.simulation.actions_simulated():
        return (0, "")
    try:
        completed = subprocess.run(  # noqa: S603 - argv is fixed, host-wrapped queries
            host.host_command(command),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return (1, "")
    return (completed.returncode, completed.stdout)


def _open(url: str, speech: str) -> ToolResult:
    if not voxa.simulation.actions_simulated() and not browsercheck.has_web_browser(_host_run):
        return ToolResult.failure(NO_BROWSER_MESSAGE, detail=url)
    if host.spawn(["xdg-open", url]) is False:
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(speech, detail=url)


def open_site(args: dict[str, str]) -> ToolResult:
    name = args["name"]
    url = site_url(name)
    if url is None:
        key = name.casefold().strip()
        if re.fullmatch(r"[a-z0-9]+\.[a-z0-9.]+", key):
            url = f"https://{key}"
        elif re.fullmatch(r"[a-z0-9]+", key):
            url = f"https://www.{key}.com"
        else:
            return ToolResult.failure(f"I don't know a site called {name}.")
    return _open(url, f"Opening {site_label(name)}.")


def open_url(args: dict[str, str]) -> ToolResult:
    url = normalize_url(args["url"])
    if url is None:
        return ToolResult.failure("That doesn't look like a web address.")
    return _open(url, f"Opening {site_label(url)}.")


def play_youtube(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    from voxa.agent.cdp import CdpError
    from voxa.agent.tools.web import get_session

    session = get_session()
    try:
        session.ensure()
    except CdpError:
        return _open(youtube_play_url(query), f"Playing {query} on YouTube.")
    ok, how, url = play(query, session)
    if not ok:
        return ToolResult.failure("YouTube wouldn't play that. I've left the page open.", detail=url)
    if how == "embed":
        speech = f"Playing {query}."
    else:
        speech = f"Playing {query} on YouTube."
    return ToolResult.success(speech, detail=f"{how} - {url}")


def search_youtube(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    return _open(youtube_search_url(query), f"Searching YouTube for {query}.")


def compose_gmail(args: dict[str, str]) -> ToolResult:
    return _open(GMAIL_COMPOSE, "Opening a new Gmail message.")


def web_search(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    return _open(f"https://duckduckgo.com/?q={urllib.parse.quote(query)}", f"Searching the web for {query}.")


def image_search(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    return _open(f"https://search.brave.com/images?q={urllib.parse.quote(query)}", f"Here are images of {query}.")


def directions(args: dict[str, str]) -> ToolResult:
    """Google Maps directions from where the user is (Maps works that out itself) to the destination."""
    destination = args["destination"]
    origin = (args.get("origin") or "").strip()
    url = f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(destination)}"
    if origin:
        url += f"&origin={urllib.parse.quote(origin)}"
    spoken = (args.get("mode") or "").casefold()
    travel = {"walking": "walking", "walk": "walking", "on foot": "walking", "biking": "bicycling", "bike": "bicycling",
              "cycling": "bicycling", "transit": "transit", "bus": "transit", "train": "transit"}.get(spoken)
    if travel:
        url += f"&travelmode={travel}"
        return _open(url, f"Here are {travel} directions to {destination}.")
    if origin:
        return _open(url, f"Here are directions from {origin} to {destination}.")
    return _open(url, f"Here are directions to {destination}.")


def find_nearby(args: dict[str, str]) -> ToolResult:
    """Google Maps search for the nearest places of a kind."""
    what = args["what"]
    url = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(what + ' near me')}"
    return _open(url, f"Here is the nearest {what}.")


def browser_tools() -> list[Tool]:
    return [
        Tool(
            name="directions",
            description="Open Google Maps directions to a place.",
            parameters={"destination": "where to go", "origin": "optional: where to start from",
                        "mode": "optional: walking, biking or transit"},
            risk=RiskLevel.REVERSIBLE,
            handler=directions,
            required=("destination",),
        ),
        Tool(
            name="find_nearby",
            description="Open Google Maps showing the nearest places of a kind (gas station, pharmacy, ...).",
            parameters={"what": "the kind of place"},
            risk=RiskLevel.REVERSIBLE,
            handler=find_nearby,
            required=("what",),
        ),
        Tool(
            name="open_site",
            description="Open a well-known website by spoken name.",
            parameters={"name": "site name, e.g. gmail"},
            risk=RiskLevel.REVERSIBLE,
            handler=open_site,
            required=("name",),
        ),
        Tool(
            name="open_url",
            description="Open any web address.",
            parameters={"url": "the address to open"},
            risk=RiskLevel.REVERSIBLE,
            handler=open_url,
            required=("url",),
        ),
        Tool(
            name="play_youtube",
            description="Play the first YouTube result for a query.",
            parameters={"query": "what to search for"},
            risk=RiskLevel.REVERSIBLE,
            handler=play_youtube,
            required=("query",),
        ),
        Tool(
            name="search_youtube",
            description="Search YouTube and open the results page.",
            parameters={"query": "what to search for"},
            risk=RiskLevel.REVERSIBLE,
            handler=search_youtube,
            required=("query",),
        ),
        Tool(
            name="compose_gmail",
            description="Open a new Gmail compose window.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=compose_gmail,
        ),
        Tool(
            name="web_search",
            description="Search the web with DuckDuckGo.",
            parameters={"query": "the search terms"},
            risk=RiskLevel.REVERSIBLE,
            handler=web_search,
            required=("query",),
        ),
        Tool(
            name="image_search",
            description="Open a Brave image search for a query.",
            parameters={"query": "what to search for"},
            risk=RiskLevel.REVERSIBLE,
            handler=image_search,
            required=("query",),
        ),
    ]
