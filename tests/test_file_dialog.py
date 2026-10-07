from __future__ import annotations

import shutil
import subprocess

from voxa.agent.tools.files import file_dialog, file_dialog_tools


def _record(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def test_save_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = file_dialog({"action": "save"})
    assert calls == [["xdotool", "key", "--clearmodifiers", "ctrl+shift+s"]]
    assert result.ok


def test_save_as_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = file_dialog({"action": "save", "name": "notes.txt"})
    assert calls == [
        ["xdotool", "key", "--clearmodifiers", "ctrl+shift+s"],
        ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", "notes.txt"],
        ["xdotool", "key", "--clearmodifiers", "Return"],
    ]
    assert result.ok


def test_load_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = file_dialog({"action": "load"})
    assert calls == [["xdotool", "key", "--clearmodifiers", "ctrl+o"]]
    assert result.ok


def test_close_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = file_dialog({"action": "close"})
    assert calls == [["xdotool", "key", "--clearmodifiers", "ctrl+w"]]
    assert result.ok


def test_new_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = file_dialog({"action": "new"})
    assert calls == [["xdotool", "key", "--clearmodifiers", "ctrl+n"]]
    assert result.ok


def test_unknown_action():
    result = file_dialog({"action": "spin"})
    assert not result.ok


def test_tool_registered():
    tools = file_dialog_tools()
    assert len(tools) == 1
    assert tools[0].name == "file_dialog"
    assert tools[0].required == ("action",)
