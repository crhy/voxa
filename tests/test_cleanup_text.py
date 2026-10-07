from __future__ import annotations

import subprocess

import voxa.agent.tools.textedit as te
from voxa.agent.cleanup import sane_result
from voxa.agent.tools.textedit import cleanup_text


def _fake_run(command, **kwargs):
    calls = te._calls
    calls.append(list(command))
    if command[:4] == ["xclip", "-selection", "clipboard", "-o"]:
        text = te._saved if len([c for c in calls if c[:4] == ["xclip", "-selection", "clipboard", "-o"]]) == 1 else te._copied
        return subprocess.CompletedProcess(command, 0, text, "")
    return subprocess.CompletedProcess(command, 0, "", "")


def _install(monkeypatch, saved, copied, model_out):
    te._calls = []
    te._saved = saved
    te._copied = copied
    monkeypatch.setattr(te.subprocess, "run", _fake_run)
    monkeypatch.setattr(te.time, "sleep", lambda _s: None)
    monkeypatch.setattr(te, "ask_model", lambda messages: model_out)


def test_happy_path(monkeypatch):
    _install(monkeypatch, "old clipboard", "helo wrld", "hello world")
    result = cleanup_text({})
    assert result.ok
    assert result.speech == "Text edited for clarity."
    assert sane_result("helo wrld", "hello world")
    argv = te._calls
    assert argv[0] == ["xclip", "-selection", "clipboard", "-o"]
    assert argv[1] == ["xdotool", "key", "--clearmodifiers", "ctrl+a"]
    assert argv[2] == ["xdotool", "key", "--clearmodifiers", "ctrl+c"]
    assert argv[3] == ["xclip", "-selection", "clipboard", "-o"]
    assert argv[4] == ["xclip", "-selection", "clipboard", "-i"]
    assert argv[5] == ["xdotool", "key", "ctrl+v"]
    assert argv[6] == ["xclip", "-selection", "clipboard", "-i"]
    assert argv[6] == argv[0][:3] + ["-i"]


def test_empty_selection(monkeypatch):
    _install(monkeypatch, "old clipboard", "", "anything")
    result = cleanup_text({})
    assert not result.ok
    assert result.speech == "I could not read any text in that window."
    assert te._calls[3] == ["xclip", "-selection", "clipboard", "-o"]
    assert len(te._calls) == 4


def test_insane_result_leaves_text(monkeypatch):
    _install(monkeypatch, "old clipboard", "hello world", "Here is the corrected text: hello world!!")
    result = cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert te._calls[-1] == ["xdotool", "key", "Right"]
    assert not any(c[:4] == ["xclip", "-selection", "clipboard", "-i"] for c in te._calls)


def test_clipboard_restored(monkeypatch):
    _install(monkeypatch, "old clipboard", "helo wrld", "hello world")
    cleanup_text({})
    restores = [c for c in te._calls if c[:4] == ["xclip", "-selection", "clipboard", "-i"]]
    assert len(restores) == 2
