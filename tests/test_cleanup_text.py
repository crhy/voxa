from __future__ import annotations

import subprocess

import voxa.agent.tools.textedit as te
from voxa.agent.cleanup import sane_result
from voxa.agent.tools.textedit import cleanup_text


def _install(monkeypatch, saved, copied, model_out):
    te._calls = []
    te._writes = []
    te._asked = []
    te._state = [saved]

    def fake_run(command, **kwargs):
        te._calls.append(list(command))
        if "ctrl+c" in command:
            te._state[0] = copied
        return subprocess.CompletedProcess(command, 0, "", "")

    def read() -> str:
        return te._state[0]

    def write(text: str) -> None:
        te._writes.append(text)
        te._state[0] = text

    def ask(messages):
        te._asked.append(messages)
        return model_out

    monkeypatch.setattr(te, "_run", fake_run)
    monkeypatch.setattr(te.time, "sleep", lambda _s: None)
    monkeypatch.setattr(te, "read_clipboard", read)
    monkeypatch.setattr(te, "write_clipboard", write)
    monkeypatch.setattr(te, "ask_model", ask)


def test_happy_path(monkeypatch):
    _install(monkeypatch, "old clipboard", "helo wrld", "hello world")
    result = cleanup_text({})
    assert result.ok
    assert result.speech == "Selection edited for clarity."
    assert sane_result("helo wrld", "hello world")
    argv = te._calls
    assert argv == [
        ["xdotool", "key", "--clearmodifiers", "ctrl+c"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
    ]
    assert te._writes == ["", "hello world", "old clipboard"]
    assert not any("xclip" in arg for call in argv for arg in call)


def test_empty_selection(monkeypatch):
    _install(monkeypatch, "old clipboard", "", "anything")
    result = cleanup_text({})
    assert not result.ok
    assert result.speech == "I could not read any text in that window."
    assert te._calls == [
        ["xdotool", "key", "--clearmodifiers", "ctrl+c"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+a"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+c"],
        ["xdotool", "key", "Right"],
    ]
    assert te._writes == ["", "old clipboard"]
    assert te._asked == []


def test_insane_result_leaves_text(monkeypatch):
    _install(monkeypatch, "old clipboard", "hello world", "Here is the corrected text: hello world!!")
    result = cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert not any("Right" in call for call in te._calls)
    assert not any("ctrl+v" in call for call in te._calls)
    assert te._writes == ["", "old clipboard"]


def test_clipboard_restored(monkeypatch):
    _install(monkeypatch, "old clipboard", "helo wrld", "hello world")
    cleanup_text({})
    assert te._state[0] == "old clipboard"
    assert te._writes == ["", "hello world", "old clipboard"]
