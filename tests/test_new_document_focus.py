from __future__ import annotations

import shutil
import subprocess
import time

from voxa.agent.tools import windows as _windows
from voxa.agent.tools.files import file_dialog

_BEFORE = "0x1 0 0 pluma.pluma Pluma - notes.txt"
_AFTER = "0x1 0 0 pluma.pluma Pluma - notes.txt\n0x2 0 0 pluma.pluma Pluma - Untitled"


def _make_fake(calls: list[list[str]], listings: list[str]):
    """A subprocess.run stand-in that records calls and replays wmctrl -lx listings in order."""
    index = iter(range(len(listings)))

    def fake(cmd, *args, **kwargs):
        calls.append(list(cmd))
        stdout = ""
        if cmd[:2] == ["wmctrl", "-lx"]:
            try:
                stdout = listings[next(index)]
            except StopIteration:
                stdout = listings[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    return fake


def _fast_time(monkeypatch) -> None:
    ticks = iter(range(1000))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _real_actions(monkeypatch) -> None:
    """Force the real (non-simulated) path so the subprocess fakes actually run."""
    monkeypatch.delenv("VOXA_SIMULATE_ACTIONS", raising=False)
    import voxa.simulation

    monkeypatch.setattr(voxa.simulation, "actions_simulated", lambda: False)


def test_new_focuses_the_new_window(monkeypatch):
    calls: list[list[str]] = []
    _real_actions(monkeypatch)
    fake = _make_fake(calls, [_BEFORE, _AFTER])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(_windows, "_run", fake)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    result = file_dialog({"action": "new"})
    assert ["xdotool", "key", "--clearmodifiers", "ctrl+n"] in calls
    assert ["wmctrl", "-i", "-a", "0x2"] in calls
    assert result.ok and result.speech == "Started a new document."


def test_new_reuses_window_without_typing(monkeypatch):
    calls: list[list[str]] = []
    _real_actions(monkeypatch)
    fake = _make_fake(calls, [_BEFORE])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(_windows, "_run", fake)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    _fast_time(monkeypatch)
    result = file_dialog({"action": "new"})
    assert ["xdotool", "key", "--clearmodifiers", "ctrl+n"] in calls
    assert not any(command[:2] == ["wmctrl", "-i"] for command in calls)
    assert not any(command[:2] == ["xdotool", "type"] for command in calls)
    assert result.ok and result.speech == "Started a new document."


def test_new_clicks_nothing(monkeypatch):
    calls: list[list[str]] = []
    _real_actions(monkeypatch)
    fake = _make_fake(calls, [_BEFORE, _AFTER])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(_windows, "_run", fake)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    file_dialog({"action": "new"})
    assert not any(command[:2] == ["xdotool", "click"] for command in calls)
