from __future__ import annotations

import subprocess

from voxa.agent.intents import route
from voxa.agent.tools import hidden_files

SCHEMA = "org.mate.caja.preferences"
KEY = "show-hidden-files"


def install_run(monkeypatch, get_stdout: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command, check=True):
        calls.append(list(command))
        stdout = get_stdout if command[:2] == ["gsettings", "get"] else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(hidden_files, "_run", run)
    return calls


def test_hidden_files_on(monkeypatch):
    calls = install_run(monkeypatch, "")
    result = hidden_files.set_hidden_files({"show": "on"})
    assert result.ok
    assert result.speech == "Showing hidden files."
    assert calls == [["gsettings", "set", SCHEMA, KEY, "true"]]


def test_hidden_files_off(monkeypatch):
    calls = install_run(monkeypatch, "")
    result = hidden_files.set_hidden_files({"show": "off"})
    assert result.ok
    assert result.speech == "Hiding hidden files."
    assert calls == [["gsettings", "set", SCHEMA, KEY, "false"]]


def test_hidden_files_toggle_from_on(monkeypatch):
    calls = install_run(monkeypatch, "true")
    result = hidden_files.set_hidden_files({"show": "toggle"})
    assert result.ok
    assert result.speech == "Hiding hidden files."
    assert calls == [
        ["gsettings", "get", SCHEMA, KEY],
        ["gsettings", "set", SCHEMA, KEY, "false"],
    ]


def test_hidden_files_toggle_from_off(monkeypatch):
    calls = install_run(monkeypatch, "false")
    result = hidden_files.set_hidden_files({"show": "toggle"})
    assert result.ok
    assert result.speech == "Showing hidden files."
    assert calls == [
        ["gsettings", "get", SCHEMA, KEY],
        ["gsettings", "set", SCHEMA, KEY, "true"],
    ]


def test_hidden_files_missing_gsettings(monkeypatch):
    def run(command, check=True):
        raise FileNotFoundError("gsettings")

    monkeypatch.setattr(hidden_files, "_run", run)
    result = hidden_files.set_hidden_files({"show": "on"})
    assert not result.ok
    assert result.speech == "I can't change hidden files on this desktop."


def test_hidden_files_missing_schema(monkeypatch):
    calls = install_run(monkeypatch, "Unable to find the specified key")
    result = hidden_files.set_hidden_files({"show": "toggle"})
    assert not result.ok
    assert result.speech == "I can't change hidden files on this desktop."
    assert calls == [["gsettings", "get", SCHEMA, KEY]]


def test_route_hidden_files_phrases():
    expected = {
        "show hidden files": "on",
        "check hidden files": "on",
        "turn on hidden files": "on",
        "hide hidden files": "off",
        "uncheck hidden files": "off",
        "turn off hidden files": "off",
        "toggle hidden files": "toggle",
    }
    for phrase, show in expected.items():
        call = route(phrase)
        assert call is not None
        assert call.tool == "hidden_files"
        assert call.args == {"show": show}
