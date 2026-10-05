from __future__ import annotations

import subprocess

from voxa import apps
from voxa.agent.tools import windows
from voxa.agent.tools.windows import (
    Window,
    close_app,
    close_window,
    flatpak_app_id,
    match_windows,
    parse_wmctrl,
    switch_to,
)

LISTING = (
    "0x04600007  0 brutalchess.BrutalChess  rhy-pc Brutal Chess\n"
    "0x03a00002  0 firefox.Firefox  rhy-pc Mozilla Firefox\n"
    "0x02b00001  0 voxa.Voxa  rhy-pc \n"
    "0x05c00009  0 gedit.Gedit  rhy-pc Notes on the meeting\n"
)

FLATPAK_PATH = "/home/rhy/.flatpak/exports/share/applications/io.github.crhy.BrutalChess.desktop"


def _argvs(calls: list[list[str]]) -> list[list[str]]:
    return [call[3:] if call[:3] == ["flatpak-spawn", "--host", "--directory=/"] else call for call in calls]


def _recorder(calls: list[list[str]], listing: str):
    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, listing, "")

    return fake_run


def test_parse_wmctrl_reads_every_line() -> None:
    parsed = parse_wmctrl(LISTING)
    assert len(parsed) == 4
    assert parsed[0] == Window("0x04600007", "brutalchess.BrutalChess", "Brutal Chess")
    assert parsed[2].title == ""
    assert parsed[3].title == "Notes on the meeting"
    assert parse_wmctrl("") == []
    assert parse_wmctrl("0x1  0 only-three-fields\n") == []


def test_match_windows_matches_title_and_class() -> None:
    parsed = parse_wmctrl(LISTING)
    assert match_windows("brutal chess", parsed) == [parsed[0]]
    assert match_windows("BrutalChess", parsed) == [parsed[0]]
    assert match_windows("meeting", parsed) == [parsed[3]]
    assert match_windows("", parsed) == []
    assert match_windows("chess", [Window("0x1", "voxa.Voxa", "Chess notes")]) == []
    assert match_windows("voxa", parsed) == [parsed[2]]


def test_flatpak_app_id() -> None:
    assert flatpak_app_id(FLATPAK_PATH) == "io.github.crhy.BrutalChess"
    assert flatpak_app_id("/usr/share/applications/org.gnome.gedit.desktop") is None


def test_close_app_closes_the_matching_window(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, LISTING))
    result = close_app({"name": "Brutal Chess"})
    assert result.ok
    assert result.speech == "Closing Brutal Chess."
    assert ["wmctrl", "-i", "-c", "0x04600007"] in _argvs(calls)


def test_close_app_falls_back_to_flatpak_kill(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, ""))
    monkeypatch.setattr(
        apps,
        "list_apps",
        lambda: [apps.DesktopApp("Brutal Chess", FLATPAK_PATH)],
    )
    result = close_app({"name": "Brutal Chess"})
    assert result.ok
    assert result.speech == "Closing Brutal Chess."
    assert ["flatpak", "kill", "io.github.crhy.BrutalChess"] in _argvs(calls)


def test_close_app_reports_when_nothing_is_open(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, ""))
    monkeypatch.setattr(apps, "list_apps", lambda: [])
    result = close_app({"name": "Brutal Chess"})
    assert not result.ok
    assert result.speech == "Brutal Chess doesn't seem to be open."


def test_close_window_and_switch_to(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, LISTING))
    monkeypatch.setattr(apps, "list_apps", lambda: [])
    assert close_window({}).speech == "Closed."
    assert ["wmctrl", "-c", ":ACTIVE:"] in _argvs(calls)
    assert switch_to({"name": "firefox"}).speech == "Switching to firefox."
    assert switch_to({"name": "the firefox"}).speech == "Switching to firefox."
    assert ["wmctrl", "-i", "-a", "0x03a00002"] in _argvs(calls)
    assert not switch_to({"name": "nowhere"}).ok


def test_close_app_strips_a_leading_the(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, LISTING))
    result = close_app({"name": "the Brutal Chess"})
    assert result.ok
    assert result.speech == "Closing Brutal Chess."
    assert ["wmctrl", "-i", "-c", "0x04600007"] in _argvs(calls)


def test_switch_to_opens_the_app_when_no_window_matches(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, ""))
    monkeypatch.setattr(
        apps,
        "list_apps",
        lambda: [apps.DesktopApp("Calculator", "/usr/share/applications/gnome-calculator.desktop")],
    )
    monkeypatch.setattr(apps, "launch", lambda app: None)
    result = switch_to({"name": "the calculator"})
    assert result.ok
    assert result.speech == "Opening Calculator."


def test_simulation_runs_nothing(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _recorder(calls, LISTING))
    monkeypatch.setattr(windows.simulation, "actions_simulated", lambda: True)
    monkeypatch.setattr(windows, "list_windows", lambda: parse_wmctrl(LISTING))
    monkeypatch.setattr(
        apps,
        "list_apps",
        lambda: [apps.DesktopApp("Brutal Chess", FLATPAK_PATH)],
    )
    assert close_app({"name": "Brutal Chess"}).ok
    assert close_window({}).ok
    assert switch_to({"name": "firefox"}).ok
    assert calls == []
