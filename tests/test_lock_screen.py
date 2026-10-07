from __future__ import annotations

import subprocess

from voxa.agent import intents
from voxa.agent.tools import windows as wt


def _fake_run(calls: list[list[str]], codes: list[int]):
    def fake(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, codes.pop(0), "", "")

    return fake


def test_lock_screen_mate_running():
    calls: list[list[str]] = []
    codes = [0, 0, 0]
    wt._run = _fake_run(calls, codes)
    wt.ON_LOCK = None
    result = wt.lock_screen({})
    assert result.ok
    assert result.speech == "Locking the screen."
    assert calls == [
        ["sh", "-c", "command -v mate-screensaver-command"],
        ["sh", "-c", "pgrep -x mate-screensaver"],
        ["mate-screensaver-command", "--lock"],
    ]


def test_lock_screen_mate_daemon_stopped():
    calls: list[list[str]] = []
    codes = [0, 1, 0, 0]
    wt._run = _fake_run(calls, codes)
    wt.ON_LOCK = None
    result = wt.lock_screen({})
    assert result.ok
    assert calls == [
        ["sh", "-c", "command -v mate-screensaver-command"],
        ["sh", "-c", "pgrep -x mate-screensaver"],
        ["sh", "-c", "mate-screensaver &"],
        ["mate-screensaver-command", "--lock"],
    ]


def test_lock_screen_falls_through():
    calls: list[list[str]] = []
    codes = [1, 1, 0, 0]
    wt._run = _fake_run(calls, codes)
    wt.ON_LOCK = None
    result = wt.lock_screen({})
    assert result.ok
    assert calls == [
        ["sh", "-c", "command -v mate-screensaver-command"],
        ["sh", "-c", "command -v xdg-screensaver"],
        ["sh", "-c", "command -v dm-tool"],
        ["dm-tool", "lock"],
    ]


def test_lock_screen_no_locker():
    calls: list[list[str]] = []
    codes = [1, 1, 1, 1, 1]
    wt._run = _fake_run(calls, codes)
    wt.ON_LOCK = None
    result = wt.lock_screen({})
    assert not result.ok
    assert result.speech == "I could not find a screen locker on this computer."
    assert calls == [
        ["sh", "-c", "command -v mate-screensaver-command"],
        ["sh", "-c", "command -v xdg-screensaver"],
        ["sh", "-c", "command -v dm-tool"],
        ["sh", "-c", "command -v xscreensaver-command"],
        ["sh", "-c", "command -v gnome-screensaver-command"],
    ]


def test_lock_screen_pause_hook_runs_first():
    calls: list[list[str]] = []
    paused: list[str] = []
    codes = [0, 0, 0]
    wt._run = _fake_run(calls, codes)
    wt.ON_LOCK = lambda: paused.append("paused")
    result = wt.lock_screen({})
    assert result.ok
    assert paused == ["paused"]
    assert calls[-1] == ["mate-screensaver-command", "--lock"]


def test_router_lock_phrases():
    for phrase in ("lock the screen", "lock my screen", "lock screen", "lock the computer",
                   "lock pc", "lock desktop", "lock it", "lock up"):
        call = intents.route(phrase)
        assert call is not None
        assert call.tool == "lock_screen"
