from __future__ import annotations

import pytest

from voxa.agent import browsercheck, host
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
    monkeypatch.setattr(browsercheck, "has_web_browser", lambda run: True)
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
            "active_window_name",
            "backup_last",
            "backup_now",
            "battery_level",
            "browse",
            "budget_set",
            "budget_status",
            "calculate",
            "calendar_add",
            "calendar_add_weekly",
            "calendar_busy",
            "calendar_cancel",
            "calendar_conflicts",
            "calendar_day",
            "calendar_next",
            "calendar_week",
            "cancel_reminders",
            "change_character",
            "change_theme",
            "check_system_version",
            "cleanup_text",
            "click_on",
            "close_app",
            "close_window",
            "close_yourself",
            "compare_prices",
            "compiz_control",
            "compose_gmail",
            "conversation_summary",
            "convert_file",
            "copy_file",
            "cpu_status",
            "daily_briefing",
            "date_forget",
            "date_remember",
            "date_upcoming",
            "date_when",
            "delete_last_note",
            "directions",
            "disk_space",
            "doc_extract",
            "empty_trash",
            "event_plan",
            "exchange_rate",
            "expense_add",
            "expense_breakdown",
            "expense_spreadsheet",
            "expense_total",
            "expense_undo",
            "fact_check",
            "family_add",
            "family_remove",
            "family_week",
            "file_dialog",
            "fill_and_submit",
            "fill_field",
            "finance_count",
            "finance_sort",
            "finance_undo",
            "find_deal",
            "find_file",
            "find_large_files",
            "find_nearby",
            "flight_status",
            "focus_start",
            "focus_status",
            "focus_stop",
            "forget_contact_email",
            "forget_fact",
            "get_contact_email",
            "get_noise_cancellation",
            "go_back",
            "gpu_status",
            "heads_up",
            "hidden_files",
            "home_scene",
            "home_set",
            "home_status",
            "home_turn",
            "image_search",
            "install_app",
            "list_add",
            "list_buttons",
            "list_characters",
            "list_clear",
            "list_done",
            "list_facts",
            "list_folder",
            "list_next",
            "list_open_windows",
            "list_pending",
            "list_progress",
            "list_radio",
            "list_read",
            "list_reminders",
            "list_remove",
            "list_tv",
            "local_events",
            "local_news",
            "local_weather",
            "lock_screen",
            "look_up",
            "make_folder",
            "market",
            "maximize_app",
            "maximize_self",
            "meal_idea",
            "meal_plan",
            "meal_shop",
            "media_control",
            "meeting_brief",
            "memory_free",
            "minimize_all",
            "minimize_app",
            "minimize_self",
            "model_status",
            "move_file",
            "now_playing",
            "open_app",
            "open_file",
            "open_folder",
            "open_notes",
            "open_site",
            "open_url",
            "page_outline",
            "play_latest",
            "play_music",
            "play_on_app",
            "play_radio",
            "play_tv",
            "play_video",
            "play_youtube",
            "presentation_make",
            "press_button",
            "press_key",
            "project_status",
            "project_steps",
            "radio_genre",
            "read_clipboard_aloud",
            "read_notes",
            "read_page",
            "read_selection",
            "read_text_file",
            "read_window",
            "recall_fact",
            "recite_poem",
            "remember_fact",
            "rename_file",
            "repeat_last",
            "restore_all",
            "restore_app",
            "resume_app",
            "say_in_language",
            "say_text",
            "scroll",
            "search_site",
            "search_web",
            "search_youtube",
            "send_gmail",
            "set_contact_email",
            "set_github_owner",
            "set_location",
            "set_noise_cancellation",
            "set_reminder",
            "set_timer",
            "set_youtube_channel",
            "show_picture",
            "sort_folder_preview",
            "suggest_next_action",
            "summarize_selection",
            "switch_to",
            "system_health",
            "system_problems",
            "system_volume",
            "take_note",
            "take_picture",
            "take_screenshot",
            "tell_date",
            "tell_time",
            "time_difference",
            "top_cpu",
            "top_memory",
            "track_package",
            "transcribe_file",
            "trash_file",
            "travel_doc_add",
            "travel_doc_forget",
            "travel_doc_list",
            "travel_doc_when",
            "travel_options",
            "trip_cost",
            "trip_forget",
            "trip_plan",
            "trip_read",
            "type_text",
            "undo_file_action",
            "uninstall_app",
            "update_system",
            "voxa_help",
            "wait_for",
            "web_search",
            "what_did_i_forget",
            "where_am_i",
            "world_time",
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
