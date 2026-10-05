from __future__ import annotations

import json

import pytest

from voxa.agent.browser_session import (
    BrowserSession,
    best_element_text,
    click_candidates,
    launch_command,
    simplify_click_text,
)
from voxa.agent.cdp import CdpError
from voxa.agent.tools import default_registry
from voxa.agent.tools.web import (
    browse,
    click_on,
    fill_and_submit,
    fill_field,
    go_back,
    page_outline,
    read_page,
    scroll,
    search_site,
    search_web,
    set_session,
    wait_for,
    web_tools,
)


class FakeSession:
    def __init__(self):
        self.calls = []
        self._title = "Test page"
        self._url = "https://example.com/"
        self._text = "Voxa test page body text"
        self._outline = [
            {"n": 1, "kind": "link", "text": "Next page", "placeholder": "", "href": "/next"},
            {"n": 2, "kind": "button", "text": "Greet", "placeholder": "", "href": ""},
            {"n": 3, "kind": "field", "text": "Your name", "placeholder": "Your name", "href": ""},
        ]
        self._click_ok = True
        self._type_ok = True
        self._wait_ok = True
        self._goto_raises = False
        self._back_ok = True
        self._search_ok = True
        self._click_returns = None

    def ensure(self):
        self.calls.append(("ensure", {}))

    def goto(self, url, wait=15.0):
        self.calls.append(("goto", {"url": url}))
        if self._goto_raises:
            raise CdpError("navigation failed")
        self._url = url

    def title(self):
        self.calls.append(("title", {}))
        return self._title

    def url(self):
        self.calls.append(("url", {}))
        return self._url

    def text(self, limit=6000):
        self.calls.append(("text", {"limit": limit}))
        return self._text[:limit]

    def outline(self, limit=60):
        self.calls.append(("outline", {"limit": limit}))
        return self._outline

    def click(self, text):
        self.calls.append(("click", {"text": text}))
        if not self._click_ok:
            return None
        return self._click_returns or text

    def type_into(self, field, value, submit=False):
        self.calls.append(("type_into", {"field": field, "value": value, "submit": submit}))
        return self._type_ok

    def wait_for_text(self, text, timeout=15.0):
        self.calls.append(("wait_for_text", {"text": text}))
        return self._wait_ok

    def evaluate(self, expression, timeout=10.0):
        self.calls.append(("evaluate", {"expression": expression}))
        return None

    def back(self):
        self.calls.append(("back", {}))
        return self._back_ok

    def search_site(self, query):
        self.calls.append(("search_site", {"query": query}))
        return self._search_ok


def _fresh_session():
    session = FakeSession()
    set_session(session)
    return session


def test_browse_success():
    session = _fresh_session()
    result = browse({"url": "example.com"})
    assert result.ok
    assert result.speech == "Opened Test page."
    assert result.detail == "Voxa test page body text"
    assert ("goto", {"url": "https://example.com"}) in session.calls


def test_browse_bad_url():
    _fresh_session()
    result = browse({"url": "not a url"})
    assert not result.ok
    assert result.speech == "That doesn't look like a web address."


def test_browse_cdp_error():
    session = _fresh_session()
    session._goto_raises = True
    result = browse({"url": "example.com"})
    assert not result.ok
    assert result.speech == "The browser did not open that page."
    assert result.detail == "navigation failed"


def test_read_page():
    _fresh_session()
    result = read_page({})
    assert result.ok
    assert result.speech == "Here's what's on the page."
    assert result.detail == "Test page - https://example.com/\nVoxa test page body text"


def test_page_outline():
    session = _fresh_session()
    result = page_outline({})
    assert result.ok
    expected = "\n".join(json.dumps(item, separators=(",", ":")) for item in session._outline)
    assert result.detail == expected


def test_click_on_success():
    _fresh_session()
    result = click_on({"text": "Next page"})
    assert result.ok
    assert result.speech == "Clicked Next page."
    assert result.detail == "Test page - https://example.com/\nClicked: Next page"


def test_click_on_failure():
    session = _fresh_session()
    session._click_ok = False
    result = click_on({"text": "Missing"})
    assert not result.ok
    assert result.speech == "I couldn't find Missing on the page."


def test_fill_field_success():
    _fresh_session()
    result = fill_field({"field": "Your name", "value": "rhY"})
    assert result.ok
    assert result.speech == "Put rhY in Your name."


def test_fill_field_failure():
    session = _fresh_session()
    session._type_ok = False
    result = fill_field({"field": "Nope", "value": "x"})
    assert not result.ok
    assert result.speech == "I couldn't find a field called Nope."


def test_fill_and_submit_success():
    session = _fresh_session()
    result = fill_and_submit({"field": "Your name", "value": "rhY"})
    assert result.ok
    assert result.speech == "Put rhY in Your name and sent it."
    assert ("type_into", {"field": "Your name", "value": "rhY", "submit": True}) in session.calls


def test_wait_for_success():
    _fresh_session()
    result = wait_for({"text": "Hello"})
    assert result.ok
    assert result.speech == "Saw Hello."


def test_wait_for_failure():
    session = _fresh_session()
    session._wait_ok = False
    result = wait_for({"text": "Nope"})
    assert not result.ok
    assert result.speech == "I never saw Nope."


def test_go_back():
    session = _fresh_session()
    result = go_back({})
    assert result.ok
    assert result.speech == "Went back."
    assert result.detail == "https://example.com/"
    assert ("back", {}) in session.calls


def test_go_back_no_history():
    session = _fresh_session()
    session._back_ok = False
    result = go_back({})
    assert not result.ok
    assert result.speech == "There was nothing to go back to."


def test_search_site_success():
    _fresh_session()
    result = search_site({"query": "Devuan"})
    assert result.ok
    assert result.speech == "Searching for Devuan."


def test_search_site_failure():
    session = _fresh_session()
    session._search_ok = False
    result = search_site({"query": "Devuan"})
    assert not result.ok
    assert result.speech == "I couldn't find a search box on this page."


def test_scroll_down():
    session = _fresh_session()
    result = scroll({"direction": "down"})
    assert result.ok
    assert result.speech == "Scrolled down."
    assert ("evaluate", {"expression": "window.scrollBy(0, 500)"}) in session.calls


def test_scroll_top():
    session = _fresh_session()
    result = scroll({"direction": "top"})
    assert result.ok
    assert result.speech == "Scrolled top."
    assert ("evaluate", {"expression": "window.scrollTo(0, 0)"}) in session.calls


def test_scroll_unknown_direction():
    _fresh_session()
    result = scroll({"direction": "sideways"})
    assert not result.ok
    assert result.speech == "I don't know how to scroll sideways."


def test_launch_command_headless_false():
    cmd = launch_command(9777)
    assert cmd[:3] == ["flatpak", "run", "com.brave.Browser"]
    assert "--headless=new" not in cmd
    assert "--remote-debugging-port=9777" in cmd
    assert cmd[-1] == "about:blank"


def test_launch_command_headless_true():
    cmd = launch_command(9777, headless=True)
    assert "--headless=new" in cmd
    assert cmd[-1] == "about:blank"


def test_launch_command_override(monkeypatch):
    monkeypatch.setenv("VOXA_BROWSER_COMMAND", "chromium")
    cmd = launch_command(9777)
    assert cmd[0] == "chromium"
    assert "--headless=new" not in cmd
    assert "--remote-debugging-port=9777" in cmd


def test_web_tools_in_registry():
    names = default_registry().names()
    for tool in web_tools():
        assert tool.name in names


def _search_fake(fields_first, fields_after=None, toggle_ok=False, submit_ok=True, button_ok=False):
    session = BrowserSession(port=1, headless=True, profile="fake")
    session.calls = []
    state = {
        "fields_first": fields_first,
        "fields_after": fields_after if fields_after is not None else fields_first,
        "toggle_ok": toggle_ok,
        "submit_ok": submit_ok,
        "button_ok": button_ok,
        "fields_count": 0,
        "nav_count": 0,
    }

    def evaluate(expression, timeout=10.0):
        session.calls.append(("evaluate", {"expression": expression}))
        if "voxa-search-fields" in expression:
            if state["fields_count"] == 0:
                state["fields_count"] += 1
                return state["fields_first"]
            return state["fields_after"]
        if "voxa-search-toggle" in expression:
            return state["toggle_ok"]
        if "voxa-search-fill" in expression:
            return {"inForm": True, "value": "Devuan"}
        if "voxa-search-request-submit" in expression:
            return state["submit_ok"]
        if "voxa-search-button" in expression:
            return state["button_ok"]
        if "timeOrigin" in expression:
            href = "https://example.com/" if state["nav_count"] < 2 else "https://example.com/results"
            state["nav_count"] += 1
            return {"href": href, "origin": 1.0, "ready": "complete"}
        return None

    def _call(method, params=None, timeout=None):
        session.calls.append(("_call", {"method": method, "params": params}))
        return {}

    def _exprs():
        return [str(p.get("expression", "")) for (_, p) in session.calls if isinstance(p, dict)]

    def _enter_calls():
        return [p for (name, p) in session.calls if name == "_call"]

    session.evaluate = evaluate
    session._call = _call
    session._exprs = _exprs
    session._enter_calls = _enter_calls
    set_session(session)
    return session


def test_launch_command_window_size():
    assert "--window-size=1366,900" in launch_command(9777)
    assert "--window-size=1366,900" in launch_command(9777, headless=True)


def test_search_site_prefers_visible_field():
    fake = _search_fake(
        [
            {"visible": False, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True},
            {"visible": True, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True},
        ]
    )
    assert fake.search_site("Devuan")
    exprs = fake._exprs()
    assert not any("voxa-search-toggle" in e for e in exprs)
    assert any("voxa-search-fill" in e for e in exprs)


def test_search_site_clicks_toggle_when_only_field_hidden():
    fake = _search_fake(
        [{"visible": False, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True}],
        fields_after=[{"visible": True, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True}],
        toggle_ok=True,
    )
    assert fake.search_site("Devuan")
    exprs = fake._exprs()
    assert any("voxa-search-toggle" in e for e in exprs)
    assert any("voxa-search-fill" in e for e in exprs)


def test_search_site_no_toggle_no_visible_field():
    fake = _search_fake(
        [{"visible": False, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True}],
        toggle_ok=False,
    )
    assert not fake.search_site("Devuan")


def test_search_site_submit_request_first():
    fake = _search_fake(
        [{"visible": True, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True}],
        submit_ok=True,
    )
    assert fake.search_site("Devuan")
    exprs = fake._exprs()
    assert any("voxa-search-request-submit" in e for e in exprs)
    assert not any("voxa-search-button" in e for e in exprs)
    assert not fake._enter_calls()


def test_search_site_submit_button_fallback():
    fake = _search_fake(
        [{"visible": True, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": True}],
        submit_ok=False,
        button_ok=True,
    )
    assert fake.search_site("Devuan")
    exprs = fake._exprs()
    assert any("voxa-search-request-submit" in e for e in exprs)
    assert any("voxa-search-button" in e for e in exprs)
    assert not fake._enter_calls()


def test_search_site_submit_enter_fallback():
    fake = _search_fake(
        [{"visible": True, "tag": "input", "name": "search", "type": "search", "id": "", "inForm": False}],
        submit_ok=False,
        button_ok=False,
    )
    assert fake.search_site("Devuan")
    exprs = fake._exprs()
    request_at = next(i for i, e in enumerate(exprs) if "voxa-search-request-submit" in e)
    button_at = next(i for i, e in enumerate(exprs) if "voxa-search-button" in e)
    enter_at = next(i for i, (name, _) in enumerate(fake.calls) if name == "_call")
    assert request_at < button_at < enter_at
    enter_calls = fake._enter_calls()
    assert all(p["params"]["type"] in ("rawKeyDown", "char", "keyUp") for p in enter_calls)


def test_search_web_success():
    session = _fresh_session()
    result = search_web({"query": "voxa assistant"})
    assert result.ok
    assert result.speech == "Searching the web for voxa assistant."
    assert ("goto", {"url": "https://duckduckgo.com/?q=voxa%20assistant"}) in session.calls
    assert result.detail == session._text[:1500]


def test_search_web_cdp_error():
    session = _fresh_session()
    session._goto_raises = True
    result = search_web({"query": "voxa"})
    assert not result.ok
    assert result.speech == "The browser did not open the search page."


def test_click_on_detail_reports_clicked_text():
    session = _fresh_session()
    session._click_returns = "Next page"
    result = click_on({"text": "Next page - Wikipedia"})
    assert result.ok
    assert "Clicked: Next page" in result.detail


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Next page - Wikipedia", "Next page"),
        ("Sign in!", "Sign in"),
        ("Save changes | Discard", "Save changes"),
        ("read the full article", "read the full article"),
    ],
)
def test_simplify_click_text(query: str, expected: str) -> None:
    assert simplify_click_text(query) == expected


@pytest.mark.parametrize(
    "query, elements, expected",
    [
        ("next page", ["Next page", "Greet", "Page two"], "Next page"),
        (
            "read the full article",
            ["Read", "Full article", "Read the full article now"],
            "Read the full article now",
        ),
        ("click here", ["Home", "Contact"], None),
    ],
)
def test_best_element_text(query: str, elements: list[str], expected: str | None) -> None:
    assert best_element_text(query, elements) == expected


@pytest.mark.parametrize(
    "query, elements, expected",
    [
        ("Next page - Wikipedia", ["Next page", "Greet"], ["Next page - Wikipedia", "Next page"]),
        ("Sign in!", ["Sign in", "Register"], ["Sign in!", "Sign in"]),
        (
            "read the full article",
            ["Read", "Full article", "Read the full article now"],
            ["read the full article", "Read the full article now"],
        ),
    ],
)
def test_click_candidates(query: str, elements: list[str], expected: list[str]) -> None:
    assert click_candidates(query, elements) == expected


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _click_fake(navigates: bool):
    session = BrowserSession(port=1, headless=True, profile="fake")
    session.calls = []
    state = {"nav_evals": 0}

    def evaluate(expression, timeout=10.0):
        session.calls.append(("evaluate", {"expression": expression}))
        if "timeOrigin" in expression:
            state["nav_evals"] += 1
            if navigates and state["nav_evals"] == 2:
                raise CdpError("context destroyed")
            if navigates and state["nav_evals"] >= 3:
                return {"href": "https://example.com/history", "origin": 2.0, "ready": "complete"}
            return {"href": "https://example.com/", "origin": 1.0, "ready": "complete"}
        if "el.click()" in expression:
            return "link"
        if "out.push" in expression:
            return ["History"]
        return None

    def _call(method, params=None, timeout=None):
        return {}

    session.evaluate = evaluate
    session._call = _call
    return session


def test_click_navigating_returns_early(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("voxa.agent.browser_session.time", clock)
    session = _click_fake(navigates=True)
    set_session(session)
    assert session.click("History") == "History"
    assert clock.now <= 0.1


def test_click_non_navigating_returns_after_settle_ms(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("voxa.agent.browser_session.time", clock)
    session = _click_fake(navigates=False)
    set_session(session)
    assert session.click("History", settle_ms=600) == "History"
    assert clock.now == pytest.approx(0.6, abs=0.05)
