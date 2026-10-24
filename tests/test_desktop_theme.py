from __future__ import annotations

import json
import subprocess

from voxa.agent.desktoptheme import find_theme, metatheme_keys, theme_words
from voxa.agent.intents import route
from voxa.agent.tools.desktoptheme import change_theme

THEMES = [
    {"id": "linux-dark", "name": "Spaced Linux Dark", "gtk_theme": "Spaced-Linux-Dark"},
    {"id": "linux-light", "name": "Spaced Linux Light", "gtk_theme": "Spaced-Linux-Light"},
    {"id": "macos", "name": "Mac OS X", "gtk_theme": "MacOSX"},
    {"id": "win311", "name": "Windows 3.11", "gtk_theme": "Win311"},
    {"id": "winxp", "name": "Windows XP", "gtk_theme": "WinXP"},
    {"id": "win7", "name": "Windows 7 Aero", "gtk_theme": "Win7Aero"},
    {"id": "win7-aero-dark", "name": "Spaced Win7Aero Dark", "gtk_theme": "Spaced-Win7Aero-Dark"},
    {"id": "ubuntu", "name": "Ubuntu", "gtk_theme": "Ubuntu"},
    {"id": "mint", "name": "Linux Mint", "gtk_theme": "Mint"},
    {"id": "win11-light", "name": "Windows 11 Light", "gtk_theme": "Win11-Light"},
    {"id": "win11-dark", "name": "Windows 11 Dark", "gtk_theme": "Win11-Dark"},
    {"id": "geoworks", "name": "GeoWorks", "gtk_theme": "GeoWorks"},
    {"id": "android", "name": "Android", "gtk_theme": "Android"},
]

_INDEX = """[X-GNOME-Metatheme]
Name=Spaced Win7Aero Dark
GtkTheme=Spaced-Win7Aero-Dark
MetacityTheme=Spaced-Win7Aero-Dark
IconTheme=Spaced-Icons-Win7Aero-Dark
CursorTheme=Spaced-Cursors-Win7Aero-Dark
"""


def test_theme_words():
    assert theme_words("Spaced Win7Aero Dark") == ["windows", "7", "aero", "dark"]
    assert theme_words("Windows 3.11") == ["windows", "3", "11"]
    assert theme_words("Mac OS X") == ["mac", "os", "x"]


def test_find_theme():
    assert find_theme("windows 7 dark", THEMES)["id"] == "win7-aero-dark"
    assert find_theme("windows 7", THEMES)["id"] == "win7"
    assert find_theme("windows seven", THEMES)["id"] == "win7"
    assert find_theme("windows 11 dark", THEMES)["id"] == "win11-dark"
    assert find_theme("mac", THEMES)["id"] == "macos"
    assert find_theme("dark", THEMES)["id"] == "linux-dark"
    assert find_theme("windows 3.11", THEMES)["id"] == "win311"
    assert find_theme("purple", THEMES) is None
    assert find_theme("", THEMES) is None


def test_metatheme_keys():
    keys = metatheme_keys(_INDEX)
    assert keys == {
        "GtkTheme": "Spaced-Win7Aero-Dark",
        "MetacityTheme": "Spaced-Win7Aero-Dark",
        "IconTheme": "Spaced-Icons-Win7Aero-Dark",
        "CursorTheme": "Spaced-Cursors-Win7Aero-Dark",
    }


def test_route_change_theme():
    result = route("change to Windows 7 dark theme")
    assert result is not None and result.tool == "change_theme"
    assert result.args["name"] == "Windows 7 dark"

    result = route("switch the desktop theme to mac")
    assert result is not None and result.tool == "change_theme"
    assert result.args["name"] == "mac"

    other = route("switch to firefox")
    assert other is None or other.tool != "change_theme"


def _fake_run(files: dict[str, str], calls: list[list[str]]):
    def fake_run(command, check=True):
        calls.append(list(command))
        if command[0] == "cat":
            text = files.get(command[1])
            if text is None:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            return subprocess.CompletedProcess(command, 0, text, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return fake_run


def test_change_theme_order(monkeypatch):
    files = {
        "/usr/share/spaced-themes/themes.json": json.dumps({"themes": THEMES}),
        "/usr/share/themes/Spaced-Win7Aero-Dark/index.theme": _INDEX,
    }
    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.tools.windows._run", _fake_run(files, calls))

    result = change_theme({"name": "windows 7 dark"})
    assert result.ok and result.speech == "Switching to Spaced Win7Aero Dark."

    gsettings = [c for c in calls if c[0] == "gsettings"]
    assert [c[3] for c in gsettings] == ["icon-theme", "theme", "cursor-theme", "gtk-theme"]
    assert gsettings[-1][:4] == ["gsettings", "set", "org.mate.interface", "gtk-theme"]
    assert gsettings[-1][4] == "Spaced-Win7Aero-Dark"
    assert ["/usr/local/bin/spaced-switch-theme", "win7-aero-dark"] in calls


def test_change_theme_not_spaced(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.tools.windows._run", _fake_run({}, calls))
    result = change_theme({"name": "windows 7"})
    assert not result.ok
    assert "Spaced Linux" in result.speech
