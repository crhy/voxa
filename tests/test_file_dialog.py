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


def test_save_tries_a_plain_save_first(monkeypatch):
    # No dialog appears in this fake desktop: a document that already has a name simply saves.
    calls = _record(monkeypatch)
    result = file_dialog({"action": "save"})
    assert ["xdotool", "key", "--clearmodifiers", "ctrl+s"] in calls
    assert result.ok and result.speech == "Saved."


def test_save_as_reports_when_no_dialog_opens(monkeypatch):
    # The dialog conversation itself is covered in tests/test_saveflow.py.
    calls = _record(monkeypatch)
    result = file_dialog({"action": "save", "name": "notes.txt"})
    assert ["xdotool", "key", "--clearmodifiers", "ctrl+shift+s"] in calls
    assert not result.ok


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
