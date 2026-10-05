from __future__ import annotations

import json
import logging
import urllib.parse

from voxa.agent.browser_session import BrowserSession
from voxa.agent.cdp import CdpError
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.browser import normalize_url

log = logging.getLogger("voxa.agent.tools.web")

_session: BrowserSession | None = None


def get_session() -> BrowserSession:
    global _session
    if _session is None:
        _session = BrowserSession()
    return _session


def set_session(session: BrowserSession | None) -> None:
    global _session
    _session = session


def browse(args: dict[str, str]) -> ToolResult:
    url = normalize_url(args["url"])
    if url is None:
        return ToolResult.failure("That doesn't look like a web address.")
    session = get_session()
    try:
        session.ensure()
        session.goto(url)
        title = session.title()
        detail = session.text(1500)
    except CdpError as exc:
        return ToolResult.failure("The browser did not open that page.", detail=str(exc))
    return ToolResult.success(f"Opened {title}.", detail=detail)


def read_page(args: dict[str, str]) -> ToolResult:
    session = get_session()
    try:
        title = session.title()
        url = session.url()
        body = session.text(6000)
        detail = f"{title} - {url}\n{body}"
    except CdpError as exc:
        return ToolResult.failure("I could not read the page.", detail=str(exc))
    return ToolResult.success("Here's what's on the page.", detail=detail)


def page_outline(args: dict[str, str]) -> ToolResult:
    session = get_session()
    try:
        items = session.outline()
    except CdpError as exc:
        return ToolResult.failure("I could not read the page.", detail=str(exc))
    detail = "\n".join(json.dumps(item, separators=(",", ":")) for item in items)
    return ToolResult.success("Here's what you can click.", detail=detail)


def click_on(args: dict[str, str]) -> ToolResult:
    text = args["text"]
    session = get_session()
    try:
        clicked = session.click(text)
        if clicked is None:
            return ToolResult.failure(f"I couldn't find {text} on the page.")
        detail = f"{session.title()} - {session.url()}\nClicked: {clicked}"
    except CdpError as exc:
        return ToolResult.failure(f"I couldn't find {text} on the page.", detail=str(exc))
    return ToolResult.success(f"Clicked {text}.", detail=detail)


def fill_field(args: dict[str, str]) -> ToolResult:
    field, value = args["field"], args["value"]
    session = get_session()
    try:
        ok = session.type_into(field, value)
    except CdpError as exc:
        return ToolResult.failure(f"I couldn't find a field called {field}.", detail=str(exc))
    if not ok:
        return ToolResult.failure(f"I couldn't find a field called {field}.")
    return ToolResult.success(f"Put {value} in {field}.")


def fill_and_submit(args: dict[str, str]) -> ToolResult:
    field, value = args["field"], args["value"]
    session = get_session()
    try:
        ok = session.type_into(field, value, submit=True)
    except CdpError as exc:
        return ToolResult.failure(f"I couldn't find a field called {field}.", detail=str(exc))
    if not ok:
        return ToolResult.failure(f"I couldn't find a field called {field}.")
    return ToolResult.success(f"Put {value} in {field} and sent it.")


def wait_for(args: dict[str, str]) -> ToolResult:
    text = args["text"]
    session = get_session()
    try:
        seen = session.wait_for_text(text)
    except CdpError as exc:
        return ToolResult.failure(f"I never saw {text}.", detail=str(exc))
    if not seen:
        return ToolResult.failure(f"I never saw {text}.")
    return ToolResult.success(f"Saw {text}.")


def go_back(args: dict[str, str]) -> ToolResult:
    session = get_session()
    try:
        went = session.back()
        detail = session.url()
    except CdpError as exc:
        return ToolResult.failure("There was nothing to go back to.", detail=str(exc))
    if not went:
        return ToolResult.failure("There was nothing to go back to.")
    return ToolResult.success("Went back.", detail=detail)


def search_site(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    session = get_session()
    try:
        ok = session.search_site(query)
    except CdpError as exc:
        return ToolResult.failure("I couldn't find a search box on this page.", detail=str(exc))
    if not ok:
        return ToolResult.failure("I couldn't find a search box on this page.")
    return ToolResult.success(f"Searching for {query}.")


def search_web(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    session = get_session()
    try:
        session.ensure()
        session.goto(f"https://duckduckgo.com/?q={urllib.parse.quote(query)}")
        detail = session.text(1500)
    except CdpError as exc:
        return ToolResult.failure("The browser did not open the search page.", detail=str(exc))
    return ToolResult.success(f"Searching the web for {query}.", detail=detail)


def scroll(args: dict[str, str]) -> ToolResult:
    direction = args["direction"]
    session = get_session()
    try:
        if direction == "down":
            session.evaluate("window.scrollBy(0, 500)")
        elif direction == "up":
            session.evaluate("window.scrollBy(0, -500)")
        elif direction == "top":
            session.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            session.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            return ToolResult.failure(f"I don't know how to scroll {direction}.")
    except CdpError as exc:
        return ToolResult.failure("I couldn't scroll the page.", detail=str(exc))
    return ToolResult.success(f"Scrolled {direction}.")


def web_tools() -> list[Tool]:
    return [
        Tool(
            name="browse",
            description="Open a web address in the browser Voxa controls.",
            parameters={"url": "the address to open"},
            risk=RiskLevel.REVERSIBLE,
            handler=browse,
            required=("url",),
        ),
        Tool(
            name="read_page",
            description="Read the text of the page currently open.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=read_page,
        ),
        Tool(
            name="page_outline",
            description="List the links, buttons and fields on the open page.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=page_outline,
        ),
        Tool(
            name="click_on",
            description="Click the element on the page whose text matches.",
            parameters={"text": "text of the link or button to click"},
            risk=RiskLevel.REVERSIBLE,
            handler=click_on,
            required=("text",),
        ),
        Tool(
            name="fill_field",
            description="Type a value into a named field on the page.",
            parameters={"field": "label or name of the field", "value": "text to type"},
            risk=RiskLevel.REVERSIBLE,
            handler=fill_field,
            required=("field", "value"),
        ),
        Tool(
            name="fill_and_submit",
            description="Type a value into a named field and press Enter.",
            parameters={"field": "label or name of the field", "value": "text to type"},
            risk=RiskLevel.REVERSIBLE,
            handler=fill_and_submit,
            required=("field", "value"),
        ),
        Tool(
            name="wait_for",
            description="Wait until the page shows the given text.",
            parameters={"text": "text that should appear"},
            risk=RiskLevel.REVERSIBLE,
            handler=wait_for,
            required=("text",),
        ),
        Tool(
            name="go_back",
            description="Go back to the previous page.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=go_back,
        ),
        Tool(
            name="search_site",
            description="Search the current site for the given query.",
            parameters={"query": "text to search for"},
            risk=RiskLevel.REVERSIBLE,
            handler=search_site,
            required=("query",),
        ),
        Tool(
            name="search_web",
            description="Search the web on DuckDuckGo in the browser Voxa controls.",
            parameters={"query": "text to search for"},
            risk=RiskLevel.REVERSIBLE,
            handler=search_web,
            required=("query",),
        ),
        Tool(
            name="scroll",
            description="Scroll the page down, up, to the top, or to the bottom.",
            parameters={"direction": "down, up, top, or bottom"},
            risk=RiskLevel.REVERSIBLE,
            handler=scroll,
            required=("direction",),
        ),
    ]
