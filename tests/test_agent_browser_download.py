"""Downloads into Voxa's own folder (issue #40 step 2).

Offline: no browser is launched, nothing is fetched, and the only files written
are inside pytest's tmp_path.
"""
from __future__ import annotations

import os

import pytest

from voxa.agent import browser_session as bs
from voxa.agent.browser_session import BrowserSession, download_dir
from voxa.agent.cdp import CdpError
from voxa.agent.policy import RiskLevel, needs_confirmation
from voxa.agent.tools import default_registry
from voxa.agent.tools.web import download_file, set_session, web_tools


class FakeSession:
    """Same shape as the fake in tests/test_agent_tools_web.py, plus downloads."""

    def __init__(self, folder: str = ""):
        self.calls = []
        self._folder = folder
        self._title = "Test page"
        self._url = "https://example.com/"
        self._text = "Voxa test page body text"
        self._outline = [
            {"n": 1, "kind": "link", "text": "Annual report", "placeholder": "", "href": "/report.pdf"},
        ]
        self._click_ok = True
        self._download_path = None
        self._download_raises = False
        self._allow_raises = False

    def ensure(self):
        self.calls.append(("ensure", {}))

    def goto(self, url, wait=15.0):
        self.calls.append(("goto", {"url": url}))
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
        return text

    def allow_downloads(self, folder=None):
        self.calls.append(("allow_downloads", {"folder": folder}))
        if self._allow_raises:
            raise CdpError("'Page.setDownloadBehavior' is not a valid method")
        return self._folder

    def await_download(self, before, folder=None, timeout=30.0):
        self.calls.append(("await_download", {"before": before, "folder": folder, "timeout": timeout}))
        if self._download_raises:
            raise CdpError("the browser socket is closed")
        return self._download_path

    def type_into(self, field, value, submit=False):
        self.calls.append(("type_into", {"field": field, "value": value, "submit": submit}))
        return True

    def wait_for_text(self, text, timeout=15.0):
        self.calls.append(("wait_for_text", {"text": text}))
        return True

    def evaluate(self, expression, timeout=10.0):
        self.calls.append(("evaluate", {"expression": expression}))
        return None

    def back(self):
        self.calls.append(("back", {}))
        return True

    def search_site(self, query):
        self.calls.append(("search_site", {"query": query}))
        return True


def _fresh(tmp_path):
    session = FakeSession(str(tmp_path))
    set_session(session)
    return session


def test_download_file_says_where_the_file_went(tmp_path):
    session = _fresh(tmp_path)
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4")
    session._download_path = str(tmp_path / "report.pdf")
    result = download_file({"link": "Annual report"})
    assert result.ok
    assert result.speech == "Saved report.pdf in your Voxa downloads folder."
    assert result.detail == str(tmp_path / "report.pdf")
    assert ("click", {"text": "Annual report"}) in session.calls


def test_download_file_snapshots_the_folder_before_clicking(tmp_path):
    session = _fresh(tmp_path)
    (tmp_path / "old.pdf").write_bytes(b"old")
    landed = tmp_path / "old (1).pdf"

    def await_download(before, folder=None, timeout=30.0):
        session.calls.append(("await_download", {"before": before, "folder": folder, "timeout": timeout}))
        landed.write_bytes(b"new")  # the browser's de-collided name appears
        return str(landed)

    session.await_download = await_download
    result = download_file({"link": "Annual report"})
    assert result.ok
    assert result.speech == "Saved old (1).pdf in your Voxa downloads folder."
    (_, params) = next(call for call in session.calls if call[0] == "await_download")
    assert params["before"] == {"old.pdf"}
    assert params["folder"] == str(tmp_path)


def test_download_file_missing_link(tmp_path):
    session = _fresh(tmp_path)
    session._click_ok = False
    result = download_file({"link": "Annual report"})
    assert not result.ok
    assert result.speech == "I couldn't find a link called Annual report on this page."
    assert not any(call[0] == "await_download" for call in session.calls)


def test_download_file_timeout_says_nothing_was_saved(tmp_path):
    session = _fresh(tmp_path)
    session._download_path = None
    result = download_file({"link": "Annual report"})
    assert not result.ok
    assert result.speech == "Nothing was saved to your Voxa downloads folder."


def test_download_file_never_claims_a_file_before_it_exists(tmp_path):
    session = _fresh(tmp_path)
    session._download_path = str(tmp_path / "report.pdf")
    result = download_file({"link": "Annual report"})
    assert not result.ok
    assert result.speech == "Nothing was saved to your Voxa downloads folder."


def test_download_file_wraps_cdp_errors(tmp_path):
    session = _fresh(tmp_path)
    session._allow_raises = True
    result = download_file({"link": "Annual report"})
    assert not result.ok
    assert result.speech == "Nothing was saved to your Voxa downloads folder."
    assert result.detail == "'Page.setDownloadBehavior' is not a valid method"
    session._allow_raises = False
    session._download_raises = True
    result = download_file({"link": "Annual report"})
    assert not result.ok
    assert result.detail == "the browser socket is closed"


def _session_with_recorded_calls(tmp_path, monkeypatch):
    """A real BrowserSession whose socket calls are recorded, not sent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    session = BrowserSession(port=1, headless=True, profile="fake")
    calls = []

    def _call(method, params=None, timeout=None):
        calls.append((method, params))
        return {}

    session._call = _call
    session.calls = calls
    return session


def test_allow_downloads_sends_the_cdp_call(tmp_path, monkeypatch):
    session = _session_with_recorded_calls(tmp_path, monkeypatch)
    folder = session.allow_downloads(str(tmp_path / "Voxa"))
    assert folder == str(tmp_path / "Voxa")
    assert session.calls == [
        ("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(tmp_path / "Voxa")})
    ]


def test_allow_downloads_defaults_to_the_voxa_folder(tmp_path, monkeypatch):
    session = _session_with_recorded_calls(tmp_path, monkeypatch)
    folder = session.allow_downloads()
    assert folder == str(tmp_path / "Downloads" / "Voxa")
    assert os.path.isdir(folder)
    assert session.calls == [
        ("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": folder})
    ]


def test_download_dir_lives_under_the_users_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = download_dir()
    assert folder == str(tmp_path / "Downloads" / "Voxa")
    assert os.path.isdir(folder)
    assert download_dir() == folder


class _OldChromium:
    """A CdpConnection stand-in for a Chromium that has no download behaviour."""

    def __init__(self, websocket_url, timeout: float = 15.0) -> None:
        self.calls = []

    def call(self, method, params=None, timeout=None):
        self.calls.append((method, params))
        if method == "Page.setDownloadBehavior":
            raise CdpError("'Page.setDownloadBehavior' is not a valid method")
        return {}

    def is_closed(self) -> bool:
        return False


def test_page_still_works_when_the_browser_has_no_download_behaviour(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(bs, "page_socket", lambda port, host="127.0.0.1": "ws://127.0.0.1:1/fake")
    monkeypatch.setattr(bs, "CdpConnection", _OldChromium)
    session = BrowserSession(port=1, headless=True, profile="fake")
    conn = session.page()
    assert [name for (name, _) in conn.calls] == [
        "Page.enable",
        "Runtime.enable",
        "DOM.enable",
        "Page.setDownloadBehavior",
    ]
    assert conn.calls[-1][1]["downloadPath"] == str(tmp_path / "Downloads" / "Voxa")
    assert session.page() is conn


def _watch() -> BrowserSession:
    """A BrowserSession used only to watch a real folder."""
    return BrowserSession(port=1, headless=True, profile="fake")


def test_await_download_reports_the_renamed_file(tmp_path):
    folder = tmp_path / "Voxa"
    folder.mkdir()
    session = _watch()
    before = set(os.listdir(folder))
    partial = folder / "report.pdf.crdownload"
    partial.write_bytes(b"%PDF-1.4")
    assert session.await_download(before, str(folder), timeout=0.5) is None
    final = folder / "report.pdf"
    os.rename(partial, final)
    found = session.await_download(before, str(folder), timeout=2.0)
    assert found == str(final)
    assert os.path.isfile(found)
    assert os.path.basename(found) not in before


def test_await_download_ignores_files_that_were_already_there(tmp_path):
    folder = tmp_path / "Voxa"
    folder.mkdir()
    (folder / "report.pdf").write_bytes(b"old")
    session = _watch()
    before = {"report.pdf"}
    assert session.await_download(before, str(folder), timeout=0.3) is None
    (folder / "report (1).pdf").write_bytes(b"new")
    assert session.await_download(before, str(folder), timeout=2.0) == str(folder / "report (1).pdf")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def test_await_download_ignores_a_file_that_is_still_growing(tmp_path, monkeypatch):
    folder = tmp_path / "Voxa"
    folder.mkdir()
    target = folder / "big.zip"
    target.write_bytes(b"x" * 10)
    clock = FakeClock()
    left = {"grows": 6}

    def grow():
        if left["grows"] > 0:
            left["grows"] -= 1
            target.write_bytes(target.read_bytes() + b"x")

    clock.on_sleep = grow
    monkeypatch.setattr("voxa.agent.browser_session.time", clock)
    session = _watch()
    found = session.await_download(set(), str(folder), timeout=5.0)
    assert found == str(target)
    assert clock.now == pytest.approx(1.75, abs=0.01)


def test_await_download_returns_none_when_the_folder_is_missing(tmp_path):
    session = _watch()
    assert session.await_download(set(), str(tmp_path / "nope"), timeout=0.3) is None


def test_download_file_is_registered_as_an_external_write():
    tool = next(t for t in web_tools() if t.name == "download_file")
    assert tool.risk == RiskLevel.EXTERNAL_WRITE
    assert tool.required == ("link",)
    assert tool.timeout_seconds == 45.0
    assert "link" in tool.parameters
    assert "download_file" in default_registry().names()
    assert needs_confirmation(RiskLevel.EXTERNAL_WRITE, frozenset(), "download_file")
