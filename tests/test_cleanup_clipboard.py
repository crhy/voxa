from __future__ import annotations

import subprocess
from pathlib import Path

from voxa.agent.tools import textedit


def make_clipboard(initial: str):
    state = [initial]
    writes: list[str] = []

    def read() -> str:
        return state[0]

    def write(text: str) -> None:
        writes.append(text)
        state[0] = text

    return state, writes, read, write


def install(monkeypatch, saved: str, window_text: str, edited: str | None = None):
    state, writes, read, write = make_clipboard(saved)
    runs: list[list[str]] = []
    asked: list[list[dict[str, str]]] = []

    def fake_run(command, stdin=None, check=True):
        runs.append(list(command))
        if "ctrl+c" in command:
            state[0] = window_text
        return subprocess.CompletedProcess(command, 0, "", "")

    def ask(messages):
        asked.append(messages)
        return edited

    monkeypatch.setattr(textedit, "_run", fake_run)
    monkeypatch.setattr(textedit.time, "sleep", lambda s: None)
    monkeypatch.setattr(textedit, "read_clipboard", read)
    monkeypatch.setattr(textedit, "write_clipboard", write)
    if edited is not None:
        monkeypatch.setattr(textedit, "ask_model", ask)
    return state, writes, runs, asked


def keys(runs: list[list[str]]) -> list[str]:
    return [argv[-1] for argv in runs]


def test_happy_path(monkeypatch):
    saved = "Hello worlde"
    state, writes, runs, asked = install(monkeypatch, saved, saved, "Hello world")
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "Selection edited for clarity."
    assert state[0] == saved
    assert writes == ["", "Hello world", saved]
    assert keys(runs) == ["ctrl+c", "ctrl+v"]
    assert all("xclip" not in argv for argv in runs)
    assert len(asked) == 1


def test_empty_window(monkeypatch):
    saved = "saved clipboard"
    state, writes, runs, asked = install(monkeypatch, saved, "", "unused")
    result = textedit.cleanup_text({})
    assert not result.ok
    assert result.speech == "I could not read any text in that window."
    assert state[0] == saved
    assert writes == ["", saved]
    assert asked == []


def test_model_returns_preamble(monkeypatch):
    saved = "Some text here."
    state, writes, runs, asked = install(monkeypatch, saved, saved, "Sure! Here is your text: Some text here.")
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert state[0] == saved
    assert writes == ["", saved]
    assert "ctrl+v" not in keys(runs)


def test_model_returns_much_longer(monkeypatch):
    saved = "Some text here."
    state, writes, runs, asked = install(monkeypatch, saved, saved, saved * 5)
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert state[0] == saved
    assert writes == ["", saved]
    assert "ctrl+v" not in keys(runs)


def test_hooks_missing(monkeypatch):
    result = textedit.cleanup_text({})
    assert not result.ok
    assert result.speech == "I cannot reach the clipboard here."
    assert textedit.read_clipboard is None
    assert textedit.write_clipboard is None


def test_no_xclip_in_source():
    assert "xclip" not in Path("voxa/agent/tools/textedit.py").read_text()
