from __future__ import annotations

import subprocess

import pytest

from voxa.agent.intents import route
from voxa.agent.tools.windows import Window, maximize_app, minimize_all, minimize_app, restore_app


def _fake_runner(runs, active_id="42"):
    def fake_run(command, check=True):
        runs.append(list(command))
        if command[:2] == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(command, 0, active_id + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return fake_run


def _patch(monkeypatch, windows, runs, active_id="42"):
    monkeypatch.setattr(
        "voxa.agent.tools.windows._run",
        _fake_runner(runs, active_id),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.windows.list_windows",
        lambda: list(windows),
    )


def test_minimize_named():
    runs: list[list[str]] = []
    windows = [Window("0x1", "pluma.Pluma", "Notes - Pluma")]
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, windows, runs)
    result = minimize_app({"name": "pluma"})
    assert runs == [["xdotool", "windowminimize", "0x1"]]
    assert result.ok
    assert result.speech == "Minimized pluma."


def test_minimize_active_window():
    runs: list[list[str]] = []
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, [], runs)
    result = minimize_app({"name": ""})
    assert runs == [["xdotool", "getactivewindow"], ["xdotool", "windowminimize", "42"]]
    assert result.ok
    assert result.speech == "Minimized."


def test_minimize_not_open():
    runs: list[list[str]] = []
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, [], runs)
    result = minimize_app({"name": "ghost"})
    assert runs == []
    assert not result.ok
    assert result.speech == "ghost does not seem to be open."


def test_maximize_named():
    runs: list[list[str]] = []
    windows = [Window("0x2", "gimp.Gimp", "Gimp")]
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, windows, runs)
    result = maximize_app({"name": "gimp"})
    assert runs == [["wmctrl", "-i", "-r", "0x2", "-b", "add,maximized_vert,maximized_horz"]]
    assert result.ok
    assert result.speech == "Maximized gimp."


def test_restore_named():
    runs: list[list[str]] = []
    windows = [Window("0x3", "pluma.Pluma", "Notes")]
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, windows, runs)
    result = restore_app({"name": "pluma"})
    assert runs == [
        ["wmctrl", "-i", "-r", "0x3", "-b", "remove,maximized_vert,maximized_horz"],
        ["wmctrl", "-i", "-a", "0x3"],
    ]
    assert result.ok
    assert result.speech == "Restored pluma."


def test_minimize_all():
    runs: list[list[str]] = []
    monkeypatch = pytest.MonkeyPatch()
    _patch(monkeypatch, [], runs)
    result = minimize_all({})
    assert runs == [["wmctrl", "-k", "on"]]
    assert result.ok
    assert result.speech == "Minimized everything."


@pytest.mark.parametrize(
    "phrase,tool,args",
    [
        ("minimize pluma", "minimize_app", {"name": "pluma"}),
        ("minimise pluma", "minimize_app", {"name": "pluma"}),
        ("minimize this", "minimize_app", {"name": ""}),
        ("minimize the window", "minimize_app", {"name": ""}),
        ("maximize pluma", "maximize_app", {"name": "pluma"}),
        ("maximise pluma", "maximize_app", {"name": "pluma"}),
        ("make pluma bigger", "maximize_app", {"name": "pluma"}),
        ("make pluma full size", "maximize_app", {"name": "pluma"}),
        ("restore pluma", "restore_app", {"name": "pluma"}),
        ("bring back pluma", "restore_app", {"name": "pluma"}),
        ("unminimize pluma", "restore_app", {"name": "pluma"}),
        ("minimize everything", "minimize_all", {}),
        ("minimize all windows", "minimize_all", {}),
        ("show the desktop", "minimize_all", {}),
    ],
)
def test_router(phrase, tool, args):
    call = route(phrase)
    assert call is not None
    assert call.tool == tool
    assert call.args == args


def test_hide_regex():
    from voxa.agent.intents import _MINIMIZE_APP

    assert _MINIMIZE_APP.fullmatch("hide pluma")
