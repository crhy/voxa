from __future__ import annotations

import subprocess

from voxa.agent.tools.volume import system_volume, volume_tools


def _record(monkeypatch, stdout: str = ""):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_louder_argv(monkeypatch):
    calls = _record(monkeypatch)
    result = system_volume({"action": "louder"})
    assert calls == [["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"]]
    assert result.ok


def test_quieter_argv(monkeypatch):
    calls = _record(monkeypatch)
    system_volume({"action": "quieter"})
    assert calls == [["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"]]


def test_set_clamps_high(monkeypatch):
    calls = _record(monkeypatch)
    result = system_volume({"action": "set", "percent": "999"})
    assert calls == [["pactl", "set-sink-volume", "@DEFAULT_SINK@", "150%"]]
    assert result.speech == "Volume 150 percent."


def test_set_clamps_low(monkeypatch):
    calls = _record(monkeypatch)
    result = system_volume({"action": "set", "percent": "-5"})
    assert calls == [["pactl", "set-sink-volume", "@DEFAULT_SINK@", "0%"]]
    assert result.speech == "Volume 0 percent."


def test_mute_argv(monkeypatch):
    calls = _record(monkeypatch)
    system_volume({"action": "mute"})
    assert calls == [["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"]]


def test_unmute_argv(monkeypatch):
    calls = _record(monkeypatch)
    system_volume({"action": "unmute"})
    assert calls == [["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"]]


def test_get_parses_volume(monkeypatch):
    calls = _record(monkeypatch, stdout="Volume: 60%\n")
    result = system_volume({"action": "get"})
    assert calls == [["pactl", "get-sink-volume", "@DEFAULT_SINK@"]]
    assert result.speech == "Volume 60 percent."


def test_get_unreadable(monkeypatch):
    _record(monkeypatch, stdout="no data")
    result = system_volume({"action": "get"})
    assert not result.ok


def test_set_missing_percent():
    result = system_volume({"action": "set"})
    assert not result.ok


def test_tool_registered():
    tools = volume_tools()
    assert len(tools) == 1
    assert tools[0].name == "system_volume"
    assert tools[0].required == ("action",)
