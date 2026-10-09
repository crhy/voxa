from __future__ import annotations

import subprocess

import voxa.agent.intents as intents
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import windows
from voxa.agent.tools.windows import scroll, scroll_keys

LISTING = (
    "0x04600007  0 brutalchess.BrutalChess  rhy-pc Brutal Chess\n"
    "0x03a00002  0 firefox.Firefox  rhy-pc Mozilla Firefox\n"
    "0x02b00001  0 voxa.Voxa  rhy-pc \n"
    "0x05c00009  0 gedit.Gedit  rhy-pc Notes on the meeting\n"
)


def _fake_run(calls: list[list[str]], active_id: str, active_title: str, listing: str):
    def run(argv, **kwargs):
        argv = list(argv)
        calls.append(argv)
        if argv[:2] == ["xdotool", "getactivewindow"] and len(argv) == 2:
            out = active_id
        elif argv[:3] == ["xdotool", "getactivewindow", "getwindowname"]:
            out = active_title
        elif argv[:2] == ["wmctrl", "-lx"]:
            out = listing
        else:
            out = ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    return run


def test_scroll_keys_every_combination() -> None:
    assert scroll_keys("up", "page") == ["Page_Up"]
    assert scroll_keys("down", "page") == ["Page_Down"]
    assert scroll_keys("up", "little") == ["Up", "Up", "Up"]
    assert scroll_keys("down", "little") == ["Down", "Down", "Down"]
    assert scroll_keys("up", "top") == ["ctrl+Home"]
    assert scroll_keys("down", "top") == ["ctrl+Home"]
    assert scroll_keys("up", "bottom") == ["ctrl+End"]
    assert scroll_keys("down", "bottom") == ["ctrl+End"]
    assert scroll_keys("up", "sideways") == []
    assert scroll_keys("down", "sideways") == []


def test_routing_each_phrase() -> None:
    assert route("scroll down") == ToolCall("scroll", {"direction": "down", "amount": "page"})
    assert route("scroll up") == ToolCall("scroll", {"direction": "up", "amount": "page"})
    assert route("page down") == ToolCall("scroll", {"direction": "down", "amount": "page"})
    assert route("page up") == ToolCall("scroll", {"direction": "up", "amount": "page"})
    assert route("scroll down a little") == ToolCall("scroll", {"direction": "down", "amount": "little"})
    assert route("scroll up a little") == ToolCall("scroll", {"direction": "up", "amount": "little"})
    assert route("scroll to the top") == ToolCall("scroll", {"direction": "up", "amount": "top"})
    assert route("scroll to the bottom") == ToolCall("scroll", {"direction": "down", "amount": "bottom"})
    assert route("go to the top of the page") == ToolCall("scroll", {"direction": "up", "amount": "top"})
    assert route("go to the bottom of the page") == ToolCall("scroll", {"direction": "down", "amount": "bottom"})


def test_keep_scrolling_uses_last_direction() -> None:
    intents._LAST_SCROLL_DIRECTION = "down"
    assert route("keep scrolling") == ToolCall("scroll", {"direction": "down", "amount": "page"})
    route("scroll up")
    assert route("keep scrolling") == ToolCall("scroll", {"direction": "up", "amount": "page"})
    route("scroll down")
    assert route("keep scrolling") == ToolCall("scroll", {"direction": "down", "amount": "page"})


def test_scroll_sends_keys_to_active_window(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x03a00002", "Mozilla Firefox", LISTING))
    result = scroll({"direction": "down", "amount": "page"})
    assert result.ok
    assert result.speech == "Scrolled down."
    assert ["wmctrl", "-i", "-a", "0x03a00002"] in calls
    assert ["xdotool", "key", "--window", "0x03a00002", "Page_Down"] in calls


def test_scroll_little_sends_three_presses(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x03a00002", "Mozilla Firefox", LISTING))
    result = scroll({"direction": "up", "amount": "little"})
    assert result.ok
    assert result.speech == "Scrolled up."
    assert ["xdotool", "key", "--window", "0x03a00002", "Up", "Up", "Up"] in calls


def test_scroll_to_top_uses_ctrl_home(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x03a00002", "Mozilla Firefox", LISTING))
    result = scroll({"direction": "up", "amount": "top"})
    assert result.ok
    assert ["xdotool", "key", "--window", "0x03a00002", "ctrl+Home"] in calls


def test_scroll_to_bottom_uses_ctrl_end(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x03a00002", "Mozilla Firefox", LISTING))
    result = scroll({"direction": "down", "amount": "bottom"})
    assert result.ok
    assert ["xdotool", "key", "--window", "0x03a00002", "ctrl+End"] in calls


def test_scroll_skips_voxas_own_window(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x02b00001", "Voxa", LISTING))
    result = scroll({"direction": "down", "amount": "page"})
    assert result.ok
    assert result.speech == "Scrolled down."
    key_calls = [c for c in calls if c[:3] == ["xdotool", "key", "--window"]]
    assert key_calls == [["xdotool", "key", "--window", "0x04600007", "Page_Down"]]
    assert "0x02b00001" not in key_calls[0]


def test_scroll_unknown_amount_sends_nothing(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(windows, "_run", _fake_run(calls, "0x03a00002", "Mozilla Firefox", LISTING))
    result = scroll({"direction": "down", "amount": "sideways"})
    assert not result.ok
    assert result.speech == "I don't know how to scroll down sideways."
    assert calls == []
