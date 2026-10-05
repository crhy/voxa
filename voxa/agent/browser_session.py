"""A controllable Brave (or any Chrome-family) browser over CDP."""

from __future__ import annotations

import json
import logging
import os
import shlex
import time

from voxa.agent import host
from voxa.agent.cdp import CdpConnection, CdpError, browser_alive, browser_socket, page_socket

log = logging.getLogger("voxa.agent.browser_session")

DEFAULT_PORT = 9777

OUTLINE_JS = """
(() => {
  const limit = __LIMIT__;
  const out = [];
  const nodes = document.querySelectorAll(
    'a[href], button, input, textarea, select, [role=button], [role=link]');
  for (const el of nodes) {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const role = el.getAttribute('role');
    const tag = el.tagName.toLowerCase();
    let kind;
    if (tag === 'a' || role === 'link') kind = 'link';
    else if (tag === 'button' || role === 'button') kind = 'button';
    else if (tag === 'input' || tag === 'textarea') kind = 'field';
    else kind = 'select';
    const text = (el.innerText || el.getAttribute('aria-label') || el.value
      || el.getAttribute('placeholder') || '').trim();
    out.push({
      n: out.length + 1,
      kind,
      text: text.slice(0, 80),
      placeholder: (el.getAttribute('placeholder') || '').trim(),
      href: el.getAttribute('href') || ''
    });
    if (out.length >= limit) break;
  }
  return out;
})()
"""

CLICK_JS = """
(() => {
  const want = (__TEXT__ || '').toLowerCase();
  if (!want) return false;
  const nodes = document.querySelectorAll(
    'a, button, input, textarea, select, [role=button], [role=link]');
  let exact = null;
  let first = null;
  for (const el of nodes) {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (el.tagName.toLowerCase() === 'a' && (el.getAttribute('href') || '').startsWith('#')) continue;
    const label = ((el.innerText || '') + ' ' + (el.getAttribute('aria-label') || '')
      + ' ' + (el.value || '')).trim().toLowerCase();
    if (label === want) { exact = el; break; }
    if (label.includes(want) && first === null) first = el;
  }
  const el = exact || first;
  if (!el) return false;
  el.scrollIntoView({block: 'center'});
  el.click();
  return (el.tagName.toLowerCase() === 'a' && el.getAttribute('href')) ? 'link' : true;
})()
"""


def simplify_click_text(text: str) -> str:
    """Drop " - " and " | " suffixes, then remove punctuation characters."""
    for sep in (" - ", " | "):
        text = text.split(sep, 1)[0]
    kept = "".join(ch for ch in text if ch.isspace() or ch.isalnum())
    return " ".join(kept.split())


def best_element_text(query: str, element_texts: list[str]) -> str | None:
    """Longest element text containing at least two of the query's words."""
    words = {w for w in query.lower().split() if w}
    best: str | None = None
    for element in element_texts:
        shared = sum(1 for w in words if w in element.lower().split())
        if shared >= 2 and (best is None or len(element) > len(best)):
            best = element
    return best


def click_candidates(text: str, element_texts: list[str]) -> list[str]:
    """Candidate click texts in order: full text, simplified text, best element text."""
    candidates = [text]
    simplified = simplify_click_text(text)
    if simplified and simplified not in candidates:
        candidates.append(simplified)
    best = best_element_text(text, element_texts)
    if best and best not in candidates:
        candidates.append(best)
    return candidates

TYPE_JS = """
(() => {
  const want = (__FIELD__ || '').toLowerCase();
  const value = __VALUE__;
  const nodes = document.querySelectorAll('input, textarea');
  for (const el of nodes) {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    let labelFor = '';
    if (el.id) {
      for (const lab of document.querySelectorAll('label')) {
        if (lab.htmlFor === el.id) { labelFor = lab.innerText || ''; break; }
      }
    }
    const hay = ((el.getAttribute('placeholder') || '') + ' '
      + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('name') || '') + ' '
      + (el.getAttribute('title') || '') + ' ' + (el.id || '') + ' ' + labelFor).trim().toLowerCase();
    let match = hay.includes(want);
    if (!match && want.includes('search') && el.getAttribute('type') === 'search') match = true;
    if (!match) continue;
    el.focus();
    el.value = value;
    el.dispatchEvent(new Event('input'));
    el.dispatchEvent(new Event('change'));
    const form = el.closest('form');
    if (form) window.__voxaForm = form;
    return { inForm: form ? true : false };
  }
  return false;
})()
"""

TEXT_JS = """
(() => {
  const body = document.body;
  return body ? body.innerText : '';
})()
"""

NAV_STATE_JS = "(() => ({href: location.href, origin: performance.timeOrigin, ready: document.readyState}))()"

SEARCH_FIELDS_JS = """
(() => {
  // voxa-search-fields
  const out = [];
  const nodes = document.querySelectorAll('input, textarea, [role=searchbox], [role=search]');
  for (const el of nodes) {
    const tag = el.tagName.toLowerCase();
    if (tag !== 'input' && tag !== 'textarea') continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    const name = (el.getAttribute('name') || '').toLowerCase();
    const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
    const isSearch = type === 'search' || role === 'searchbox' || role === 'search'
      || name.includes('search') || name === 'q'
      || placeholder.includes('search') || placeholder === 'q'
      || ariaLabel.includes('search') || ariaLabel === 'q';
    if (!isSearch) continue;
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    const visible = el.offsetParent !== null
      && cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0'
      && r.width > 0 && r.height > 0;
    out.push({
      visible,
      tag,
      name: el.getAttribute('name') || '',
      type,
      id: el.id || '',
      inForm: el.closest('form') ? true : false
    });
  }
  return out;
})()
"""

SEARCH_TOGGLE_JS = """
(() => {
  // voxa-search-toggle
  const nodes = document.querySelectorAll('a, button, [role=button], [role=link]');
  for (const el of nodes) {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const label = ((el.getAttribute('aria-label') || '') + ' '
      + (el.getAttribute('title') || '') + ' ' + (el.innerText || '')).trim().toLowerCase();
    if (!label.includes('search')) continue;
    el.click();
    return true;
  }
  return false;
})()
"""

SEARCH_FILL_JS = """
(() => {
  // voxa-search-fill
  const query = __QUERY__;
  const nodes = document.querySelectorAll('input, textarea, [role=searchbox], [role=search]');
  for (const el of nodes) {
    const tag = el.tagName.toLowerCase();
    if (tag !== 'input' && tag !== 'textarea') continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    const name = (el.getAttribute('name') || '').toLowerCase();
    const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
    const isSearch = type === 'search' || role === 'searchbox' || role === 'search'
      || name.includes('search') || name === 'q'
      || placeholder.includes('search') || placeholder === 'q'
      || ariaLabel.includes('search') || ariaLabel === 'q';
    if (!isSearch) continue;
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    const visible = el.offsetParent !== null
      && cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0'
      && r.width > 0 && r.height > 0;
    if (!visible) continue;
    el.focus();
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    if (el.tagName === 'INPUT') {
      const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      if (desc && desc.set) desc.set.call(el, query); else el.value = query;
    } else {
      el.value = query;
      el.textContent = query;
    }
    const form = el.closest('form');
    if (form) window.__voxaForm = form;
    let url = false;
    if (form) {
      const action = form.action || location.href;
      const parts = [];
      for (const f of form.querySelectorAll('input, textarea')) {
        const fname = f.getAttribute('name') || '';
        if (!fname) continue;
        let v = '';
        if (f.tagName.toLowerCase() === 'textarea') v = f.textContent || f.value || '';
        else v = f.value || f.getAttribute('value') || '';
        parts.push(fname + '=' + encodeURIComponent(v));
      }
      const sep = action.includes('?') ? '&' : '?';
      url = action + sep + parts.join('&');
    }
    return { inForm: form ? true : false, value: el.value, url: url };
  }
  return false;
})()
"""

SEARCH_SUBMIT_JS = """
(() => {
  // voxa-search-request-submit
  const nodes = document.querySelectorAll('input, textarea, [role=searchbox], [role=search]');
  for (const el of nodes) {
    const tag = el.tagName.toLowerCase();
    if (tag !== 'input' && tag !== 'textarea') continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    const name = (el.getAttribute('name') || '').toLowerCase();
    const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
    const isSearch = type === 'search' || role === 'searchbox' || role === 'search'
      || name.includes('search') || name === 'q'
      || placeholder.includes('search') || placeholder === 'q'
      || ariaLabel.includes('search') || ariaLabel === 'q';
    if (!isSearch) continue;
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    const visible = el.offsetParent !== null
      && cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0'
      && r.width > 0 && r.height > 0;
    if (!visible) continue;
    const form = el.form || el.closest('form');
    if (!form) return false;
    try {
      form.requestSubmit();
      return true;
    } catch (e) {
      return false;
    }
  }
  return false;
})()
"""

SEARCH_BUTTON_JS = """
(() => {
  // voxa-search-button
  const visible = (el) => {
    if (el.hasAttribute('hidden') || el.hasAttribute('disabled')) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const fields = document.querySelectorAll('input, textarea, [role=searchbox], [role=search]');
  let form = null;
  for (const el of fields) {
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    const name = (el.getAttribute('name') || '').toLowerCase();
    const isSearch = type === 'search' || role === 'searchbox' || role === 'search'
      || name.includes('search') || name === 'q';
    if (!isSearch) continue;
    form = (el.form || el.closest('form'));
    if (form) break;
  }
  const scope = form ? form.querySelectorAll('button, input')
    : document.querySelectorAll('button[type=submit], input[type=submit], button');
  for (const btn of scope) {
    const tag = btn.tagName.toLowerCase();
    const btype = (btn.getAttribute('type') || '').toLowerCase();
    if (tag === 'button' && (btype === 'submit' || btype === '')) {
      if (visible(btn)) { btn.click(); return true; }
    } else if (tag === 'input' && btype === 'submit') {
      if (visible(btn)) { btn.click(); return true; }
    }
  }
  for (const btn of scope) {
    if (btn.tagName.toLowerCase() !== 'button') continue;
    if (!visible(btn)) continue;
    if ((btn.innerText || '').trim().toLowerCase() === 'search') { btn.click(); return true; }
  }
  return false;
})()
"""

SEARCH_SERIALIZE_JS = """
(() => {
  // voxa-search-serialize
  const nodes = document.querySelectorAll('input, textarea, [role=searchbox], [role=search]');
  for (const el of nodes) {
    const tag = el.tagName.toLowerCase();
    if (tag !== 'input' && tag !== 'textarea') continue;
    const type = (el.getAttribute('type') || '').toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    const name = (el.getAttribute('name') || '').toLowerCase();
    const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
    const isSearch = type === 'search' || role === 'searchbox' || role === 'search'
      || name.includes('search') || name === 'q'
      || placeholder.includes('search') || placeholder === 'q'
      || ariaLabel.includes('search') || ariaLabel === 'q';
    if (!isSearch) continue;
    const form = el.form || el.closest('form');
    if (!form) return false;
    const action = form.action || location.href;
    const parts = [];
    for (const f of form.querySelectorAll('input, textarea')) {
      const fname = f.getAttribute('name') || '';
      if (!fname) continue;
      let v = '';
      if (f.tagName.toLowerCase() === 'textarea') v = f.textContent || f.value || '';
      else v = f.value || f.getAttribute('value') || '';
      parts.push(fname + '=' + encodeURIComponent(v));
    }
    const sep = action.includes('?') ? '&' : '?';
    return action + sep + parts.join('&');
  }
  return false;
})()
"""


def profile_dir() -> str:
    """A profile directory the Brave Flatpak can always write to."""
    return os.path.expanduser("~/.var/app/com.brave.Browser/config/voxa-profile")


def launch_command(
    port: int,
    headless: bool = False,
    url: str = "about:blank",
    profile: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """The command that starts a controllable browser instance."""
    override = os.environ.get("VOXA_BROWSER_COMMAND")
    if override:
        base = shlex.split(override)
    else:
        base = ["flatpak", "run", "com.brave.Browser"]
    command = [
        *base,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile or profile_dir()}",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1366,900",
    ]
    if headless:
        command.append("--headless=new")
    else:
        command.append("--autoplay-policy=no-user-gesture-required")
    command.extend(extra_args)
    command.append(url)
    return command


class BrowserSession:
    def __init__(self, port: int = DEFAULT_PORT, headless: bool = False, profile: str | None = None,
                 extra_args: tuple[str, ...] = ()) -> None:
        self._port = port
        self._headless = headless
        self._profile = profile
        self._extra_args = tuple(extra_args)
        self._conn: CdpConnection | None = None

    def ensure(self) -> None:
        """Start the browser if it is not already answering on the port."""
        if browser_alive(self._port):
            return
        host.spawn(self.launch())
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if browser_alive(self._port):
                return
            time.sleep(0.25)
        raise CdpError("the browser did not start in time")

    def launch(self) -> list[str]:
        return launch_command(self._port, self._headless, "about:blank", self._profile, self._extra_args)

    def page(self) -> CdpConnection:
        """Connection to the active page; reconnects when the socket died."""
        if self._conn is not None and self._conn.is_closed():
            self._conn = None
        if self._conn is None:
            socket_url = page_socket(self._port)
            if socket_url is None:
                raise CdpError("the browser has no debuggable page")
            conn = CdpConnection(socket_url)
            for domain in ("Page", "Runtime", "DOM"):
                conn.call(f"{domain}.enable")
            self._conn = conn
        return self._conn

    def _call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        conn = self.page()
        try:
            return conn.call(method, params, timeout)
        except CdpError as exc:
            if "socket" not in str(exc):
                raise
            self._conn = None
            return self.page().call(method, params, timeout)

    def goto(self, url: str, wait: float = 15.0) -> None:
        """Navigate and wait for the load event."""
        conn = self.page()
        conn.call("Page.navigate", {"url": url})
        conn.wait_event("Page.loadEventFired", wait)

    def evaluate(self, expression: str, timeout: float = 10.0):
        """Evaluate a page expression by value; CdpError on page exceptions."""
        result = self._call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            timeout,
        )
        details = result.get("exceptionDetails")
        if details:
            raise CdpError(str(details.get("text", details)) if isinstance(details, dict) else str(details))
        return result.get("result", {}).get("value")

    def title(self) -> str:
        return str(self.evaluate("document.title") or "")

    def url(self) -> str:
        return str(self.evaluate("document.location.href") or "")

    def text(self, limit: int = 6000) -> str:
        """Visible page text, whitespace collapsed and cut to limit."""
        raw = self.evaluate(TEXT_JS) or ""
        return " ".join(str(raw).split())[:limit]

    def outline(self, limit: int = 60) -> list[dict]:
        """Visible interactive elements, for the planner."""
        js = OUTLINE_JS.replace("__LIMIT__", json.dumps(limit))
        value = self.evaluate(js)
        return value if isinstance(value, list) else []

    def click(self, text: str, settle_ms: int = 600) -> str | None:
        """Click the first visible element whose text matches; return as soon as the page settles.

        Tries the full text, then the text with punctuation and " - "/" | "
        suffixes removed, then the longest element text that contains at
        least two of the query's words. Returns the text actually clicked.
        """
        texts = [e.get("text", "") if isinstance(e, dict) else str(e) for e in self.outline()]
        for candidate in click_candidates(text, texts):
            js = CLICK_JS.replace("__TEXT__", json.dumps(candidate))
            before = self.evaluate(NAV_STATE_JS)
            if self.evaluate(js):
                self._wait_after_click(before, settle_ms)
                return candidate
        return None

    def _wait_after_click(self, before, settle_ms: int = 600) -> None:
        """Poll page state after a click: return once a navigation has finished
        (href or performance.timeOrigin changed and readyState is interactive
        or complete), or after settle_ms with no change (non-navigating click); a
        navigation that never reports readyState gives up 15 s later. Tolerates
        "context destroyed" errors while the page swaps. Polls every 50 ms."""
        before = before if isinstance(before, dict) else {}
        deadline = time.monotonic() + settle_ms / 1000.0
        navigating = False
        while True:
            time.sleep(0.05)
            try:
                after = self.evaluate(NAV_STATE_JS)
            except CdpError as exc:
                message = str(exc).lower()
                if "context destroyed" in message or "cannot find context" in message or "navigated or closed" in message:
                    navigating = True
                    continue
                raise
            if not isinstance(after, dict):
                continue
            if after.get("href") != before.get("href") or after.get("origin") != before.get("origin"):
                navigating = True
            if navigating:
                if after.get("ready") in ("interactive", "complete"):
                    return
                if time.monotonic() >= deadline + 15.0:
                    return
            elif time.monotonic() >= deadline:
                return

    def type_into(self, field: str, value: str, submit: bool = False) -> bool:
        """Fill a visible field by label; optionally press Enter."""
        js = TYPE_JS.replace("__FIELD__", json.dumps(field)).replace("__VALUE__", json.dumps(value))
        result = self.evaluate(js)
        if not result:
            return False
        if not submit:
            return True
        in_form = isinstance(result, dict) and result.get("inForm")
        if in_form:
            return self.wait_for_navigation(
                lambda: self.evaluate("window.__voxaForm.requestSubmit()"), timeout=15.0
            )
        return self.wait_for_navigation(self._press_enter, timeout=15.0)

    def _press_enter(self) -> None:
        """Send a real Enter key through the input domain; needs focus on the field."""
        self._call(
            "Input.dispatchKeyEvent",
            {"type": "rawKeyDown", "key": "Enter", "code": "Enter",
             "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13},
        )
        self._call("Input.dispatchKeyEvent", {"type": "char", "text": "\r"})
        self._call(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": "Enter", "code": "Enter",
             "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13},
        )

    def wait_for_text(self, text: str, timeout: float = 15.0) -> bool:
        """Poll the page text until it contains text."""
        needle = text.lower()
        deadline = time.monotonic() + timeout
        while True:
            if needle in self.text(20000).lower():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def wait_for_navigation(self, action, timeout: float = 15.0) -> bool:
        """Record page state, run action, poll until navigation completes or timeout."""
        before = self.evaluate(NAV_STATE_JS)
        action()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.1)
            after = self.evaluate(NAV_STATE_JS)
            if (after["href"] != before["href"] or after["origin"] != before["origin"]) and after["ready"] == "complete":
                return True
        return False

    def back(self) -> bool:
        """Go back to the previous page; wait for the navigation to complete."""
        return self.wait_for_navigation(lambda: self.evaluate("history.back()"), timeout=10.0)

    def search_site(self, query: str) -> bool:
        """Search the current site: prefer visible fields, click a toggle if needed,
        fill with the native setter, then submit (requestSubmit, submit button, Enter)."""
        fields = self.evaluate(SEARCH_FIELDS_JS)
        if not isinstance(fields, list) or not fields:
            return False
        visible = [f for f in fields if isinstance(f, dict) and f.get("visible")]
        if not visible:
            if not self.evaluate(SEARCH_TOGGLE_JS):
                return False
            time.sleep(0.3)
            fields = self.evaluate(SEARCH_FIELDS_JS)
            visible = [f for f in fields if isinstance(f, dict) and f.get("visible")] if isinstance(fields, list) else []
        if not visible:
            return False
        js = SEARCH_FILL_JS.replace("__QUERY__", json.dumps(query))
        filled = self.evaluate(js)
        if not filled:
            return False
        before = self.evaluate(NAV_STATE_JS)
        before_href = before.get("href", "") if isinstance(before, dict) else ""
        log.info("search_site before: %s", before_href)
        built = filled.get("url") if isinstance(filled, dict) else False
        attempts = [
            ("requestSubmit", lambda: self.evaluate(SEARCH_SUBMIT_JS)),
            ("submit button", lambda: self.evaluate(SEARCH_BUTTON_JS)),
            ("enter key", lambda: self._press_enter() or True),
        ]
        for label, action in attempts:
            ran = action()
            if not ran:
                continue
            if self.wait_for_navigation(lambda: None, timeout=5.0):
                after = self.evaluate(NAV_STATE_JS)
                after_href = after.get("href", "") if isinstance(after, dict) else ""
                log.info("search_site after (%s): %s", label, after_href)
                return True
        after = self.evaluate(NAV_STATE_JS)
        after_href = after.get("href", "") if isinstance(after, dict) else ""
        log.info("search_site after (none): %s", after_href)
        if isinstance(built, str) and built:
            log.info("search_site serialize: %s", built)
            self.goto(built)
            after = self.evaluate(NAV_STATE_JS)
            after_href = after.get("href", "") if isinstance(after, dict) else ""
            log.info("search_site after (serialize): %s", after_href)
            return True
        return False

    def close_browser(self) -> None:
        """Close the whole browser through the browser-level socket."""
        socket_url = browser_socket(self._port)
        if socket_url is None:
            return
        conn = CdpConnection(socket_url)
        try:
            conn.call("Browser.close")
        except CdpError:
            pass
        finally:
            conn.close()
