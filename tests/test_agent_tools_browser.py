from __future__ import annotations

import pytest

from voxa.agent import host
from voxa.agent.tools import default_registry
from voxa.agent.tools.browser import (
    first_video_id,
    normalize_url,
    site_url,
    youtube_play_url,
    youtube_search_url,
)

INBOX = "https://mail.google.com/mail/u/0/#inbox"


@pytest.fixture
def opened(monkeypatch):
    commands: list[list[str]] = []

    def fake_spawn(command):
        commands.append(list(command))

    monkeypatch.setattr(host, "spawn", fake_spawn)
    return commands


def test_site_url_variants():
    assert site_url("Gmail") == INBOX
    assert site_url("the YouTube website") == "https://www.youtube.com/"
    assert site_url("Github dot com") == "https://github.com/"
    assert site_url("not a real site") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("example.com", "https://example.com"),
        ("  https://example.com/path?a=1  ", "https://example.com/path?a=1"),
        ("http://example.com", "http://example.com"),
        ("javascript:alert(1)", None),
        ("file:///etc/passwd", None),
        ("has a space.com", None),
        ("nodot", None),
        ("", None),
    ],
)
def test_normalize_url(text, expected):
    assert normalize_url(text) == expected


def test_youtube_search_url_quotes_query():
    assert youtube_search_url("lo-fi beats & rain") == (
        "https://www.youtube.com/results?search_query=lo-fi%20beats%20%26%20rain"
    )
    assert youtube_search_url("lo-fi beats & rain", music=True) == (
        "https://music.youtube.com/search?q=lo-fi%20beats%20%26%20rain"
    )


def test_first_video_id():
    page = 'blah "videoId":"dQw4w9WgXcQ" more "videoId":"short"'
    assert first_video_id(page) == "dQw4w9WgXcQ"
    assert first_video_id("no videos here") is None


def test_youtube_play_url_resolves_first_video():
    page = '{"videoId":"dQw4w9WgXcQ"},{"videoId":"abcdefghijk"}'
    assert youtube_play_url("never gonna", fetch=lambda url: page) == (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )


def test_youtube_play_url_falls_back_when_no_id():
    assert youtube_play_url("nothing", fetch=lambda url: "no ids here") == youtube_search_url("nothing")


def test_youtube_play_url_falls_back_when_fetch_raises():
    def boom(url):
        raise OSError("network down")

    assert youtube_play_url("nothing", fetch=boom) == youtube_search_url("nothing")


def test_open_site_tool(opened):
    result = default_registry().call("open_site", {"name": "the Gmail website"})
    assert result.ok
    assert result.speech == "Opening the Gmail website."
    assert opened == [["xdg-open", INBOX]]


def test_open_site_unknown_tool(opened):
    result = default_registry().call("open_site", {"name": "myspace"})
    assert result.ok
    assert result.speech == "Opening myspace."
    assert opened == [["xdg-open", "https://www.myspace.com"]]


def test_open_url_tool(opened):
    result = default_registry().call("open_url", {"url": "example.com"})
    assert result.ok
    assert opened == [["xdg-open", "https://example.com"]]


def test_open_url_rejects_bad_input(opened):
    result = default_registry().call("open_url", {"url": "javascript:alert(1)"})
    assert not result.ok
    assert result.speech == "That doesn't look like a web address."
    assert opened == []


def test_play_youtube_tool(opened, monkeypatch):
    class FakeSession:
        def ensure(self):
            return None

        def goto(self, url, wait=15.0):
            return None

        def _call(self, method, params=None, timeout=None):
            return {}

        def evaluate(self, expression, timeout=10.0):
            if "has_video" in expression:
                return {"playing": True, "wall": False, "has_video": True, "time": 1.0, "error": ""}
            return True

    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: FakeSession())
    result = default_registry().call("play_youtube", {"query": "lo-fi beats"})
    assert result.ok
    assert result.speech == "Playing lo-fi beats on YouTube."
    assert opened == []


def test_play_youtube_embed_speech(opened, monkeypatch):
    states = [{"playing": False, "wall": True}, {"playing": False, "wall": True}, {"playing": True}]

    class FakeSession:
        def ensure(self):
            return None

        def goto(self, url, wait=15.0):
            return None

        def _call(self, method, params=None, timeout=None):
            return {}

        def evaluate(self, expression, timeout=10.0):
            if "has_video" in expression:
                return states.pop(0)
            return True

    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: FakeSession())
    result = default_registry().call("play_youtube", {"query": "lo-fi beats"})
    assert result.ok
    assert result.speech == "Playing lo-fi beats."


def test_play_youtube_falls_back_on_cdp_error(opened, monkeypatch):
    from voxa.agent.cdp import CdpError

    class BrokenSession:
        def ensure(self):
            raise CdpError("the browser did not start in time")

    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: BrokenSession())
    monkeypatch.setattr(
        "voxa.agent.tools.browser.youtube_play_url",
        lambda query: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    result = default_registry().call("play_youtube", {"query": "lo-fi beats"})
    assert result.ok
    assert result.speech == "Playing lo-fi beats on YouTube."
    assert opened == [["xdg-open", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]]


def test_search_youtube_tool(opened):
    result = default_registry().call("search_youtube", {"query": "lo-fi beats & rain"})
    assert result.ok
    assert result.speech == "Searching YouTube for lo-fi beats & rain."
    assert opened == [
        ["xdg-open", "https://www.youtube.com/results?search_query=lo-fi%20beats%20%26%20rain"]
    ]


def test_compose_gmail_tool(opened):
    result = default_registry().call("compose_gmail", {})
    assert result.ok
    assert result.speech == "Opening a new Gmail message."
    assert opened == [["xdg-open", "https://mail.google.com/mail/u/0/?view=cm&fs=1&tf=1"]]


def test_web_search_tool(opened):
    result = default_registry().call("web_search", {"query": "lo-fi beats & rain"})
    assert result.ok
    assert result.speech == "Searching the web for lo-fi beats & rain."
    assert opened == [["xdg-open", "https://duckduckgo.com/?q=lo-fi%20beats%20%26%20rain"]]


def test_image_search_tool(opened):
    result = default_registry().call("image_search", {"query": "lo-fi beats & rain"})
    assert result.ok
    assert result.speech == "Here are images of lo-fi beats & rain."
    assert opened == [["xdg-open", "https://search.brave.com/images?q=lo-fi%20beats%20%26%20rain"]]


def test_registry_lists_all_tools():
    names = default_registry().names()
    assert names == [
        "browse",
        "cancel_reminders",
        "click_on",
        "close_app",
        "close_window",
        "compiz_control",
        "compose_gmail",
        "copy_file",
        "directions",
        "empty_trash",
        "file_dialog",
        "fill_and_submit",
        "fill_field",
        "find_deal",
        "find_file",
        "find_large_files",
        "find_nearby",
        "go_back",
        "home_scene",
        "home_set",
        "home_status",
        "home_turn",
        "image_search",
        "list_folder",
        "list_reminders",
        "lock_screen",
        "make_folder",
        "maximize_app",
        "media_control",
        "minimize_all",
        "minimize_app",
        "move_file",
        "open_app",
        "open_folder",
        "open_site",
        "open_url",
        "page_outline",
        "play_latest",
        "play_music",
        "play_video",
        "play_youtube",
        "press_key",
        "read_page",
        "rename_file",
        "repeat_last",
        "restore_app",
        "say_text",
        "scroll",
        "search_site",
        "search_web",
        "search_youtube",
        "send_gmail",
        "set_location",
        "set_reminder",
        "set_timer",
        "set_youtube_channel",
        "switch_to",
        "system_volume",
        "trash_file",
        "type_text",
        "undo_file_action",
        "wait_for",
        "web_search",
    ]


def test_simulate_actions_skips_popen(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    launched = []

    class FakeSubprocess:
        DEVNULL = 0

        @staticmethod
        def Popen(*args, **kwargs):
            launched.append(args)
            raise AssertionError("Popen must not be called when simulating")

    monkeypatch.setattr(host, "subprocess", FakeSubprocess)
    host.spawn(["xdg-open", "https://example.com"])
    assert launched == []
