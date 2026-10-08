from __future__ import annotations

import subprocess

from voxa.agent.tools import windows as wt


def _fake_run(calls: list[list[str]]):
    def fake(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    return fake


def _make_windows(*specs: tuple[str, str, str]) -> list[wt.Window]:
    return [wt.Window(wid, wm, title) for wid, wm, title in specs]


def test_minimize_all_skips_desktop_and_panels(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(wt, "_run", _fake_run(calls))
    monkeypatch.setattr(wt, "list_windows", lambda: _make_windows(
        ("1", "Brave.brave", "Brave"),
        ("2", "mate-panel", "Panel"),
        ("3", "Voxa", "Voxa"),
        ("4", "Gnome-clocks", "Clocks"),
    ))
    wt._MINIMIZED_IDS.clear()
    result = wt.minimize_all({})
    assert result.ok
    assert result.speech == "Minimized everything."
    assert calls == [
        ["xdotool", "windowminimize", "1"],
        ["xdotool", "windowminimize", "4"],
        ["xdotool", "windowminimize", "3"],
    ]
    assert wt._MINIMIZED_IDS == ["1", "4", "3"]


def test_minimize_all_own_window_last(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(wt, "_run", _fake_run(calls))
    monkeypatch.setattr(wt, "list_windows", lambda: _make_windows(
        ("7", "voxa.voxa", "Voxa"),
        ("8", "Gnome-text-editor.Gnome-text-editor", "Text Editor"),
    ))
    wt._MINIMIZED_IDS.clear()
    result = wt.minimize_all({})
    assert result.ok
    assert calls == [
        ["xdotool", "windowminimize", "8"],
        ["xdotool", "windowminimize", "7"],
    ]
    assert wt._MINIMIZED_IDS == ["8", "7"]


def test_restore_all_reverse_order(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(wt, "_run", _fake_run(calls))
    wt._MINIMIZED_IDS[:] = ["1", "4", "3"]
    result = wt.restore_all({})
    assert result.ok
    assert result.speech == "Restored everything."
    assert calls == [
        ["wmctrl", "-i", "-a", "3"],
        ["wmctrl", "-i", "-a", "4"],
        ["wmctrl", "-i", "-a", "1"],
    ]
    assert wt._MINIMIZED_IDS == []


def test_minimize_all_nothing_open(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(wt, "_run", _fake_run(calls))
    monkeypatch.setattr(wt, "list_windows", lambda: _make_windows(
        ("1", "mate-panel", "Panel"),
        ("2", "Gnome-desktop-window.Gnome-desktop-window", "Desktop"),
    ))
    wt._MINIMIZED_IDS.clear()
    result = wt.minimize_all({})
    assert result.ok
    assert result.speech == "Minimized everything."
    assert calls == []
    assert wt._MINIMIZED_IDS == []
