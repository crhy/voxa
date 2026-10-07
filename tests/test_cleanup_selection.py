from __future__ import annotations

import subprocess

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import textedit


def install(monkeypatch, saved, ctrl_c_values, model_out):
    state = [saved]
    writes: list[str] = []
    runs: list[list[str]] = []
    asked: list[list[dict[str, str]]] = []
    idx = [0]

    def read() -> str:
        return state[0]

    def write(text: str) -> None:
        writes.append(text)
        state[0] = text

    def fake_run(command, **kwargs):
        runs.append(list(command))
        if "ctrl+c" in command:
            state[0] = ctrl_c_values[idx[0]]
            idx[0] += 1
        return subprocess.CompletedProcess(command, 0, "", "")

    def ask(messages):
        asked.append(messages)
        return model_out

    monkeypatch.setattr(textedit, "_run", fake_run)
    monkeypatch.setattr(textedit.time, "sleep", lambda s: None)
    monkeypatch.setattr(textedit, "read_clipboard", read)
    monkeypatch.setattr(textedit, "write_clipboard", write)
    monkeypatch.setattr(textedit, "ask_model", ask)
    return state, writes, runs, asked


def keys(runs):
    return [argv[-1] for argv in runs]


def test_selection_present(monkeypatch):
    state, writes, runs, _asked = install(monkeypatch, "clip", ["helo wrld"], "hello world")
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "Selection edited for clarity."
    assert not any("ctrl+a" in argv for argv in runs)
    assert keys(runs) == ["ctrl+c", "ctrl+v"]
    assert writes == ["", "hello world", "clip"]
    assert state[0] == "clip"


def test_nothing_selected(monkeypatch):
    state, writes, runs, _asked = install(monkeypatch, "clip", ["", "the whole doc"], "the whole document")
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "Text edited for clarity."
    assert keys(runs) == ["ctrl+c", "ctrl+a", "ctrl+c", "ctrl+v"]
    assert writes == ["", "the whole document", "clip"]
    assert state[0] == "clip"


def test_selection_nonsense(monkeypatch):
    state, writes, runs, _asked = install(
        monkeypatch, "clip", ["helo wrld"], "Sure! Here is the corrected text: hello world"
    )
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert "ctrl+v" not in keys(runs)
    assert "Right" not in keys(runs)
    assert writes == ["", "clip"]
    assert state[0] == "clip"


def test_whole_nonsense(monkeypatch):
    state, writes, runs, _asked = install(
        monkeypatch, "clip", ["", "the whole doc"], "Sure! Here is the corrected text: the whole document"
    )
    result = textedit.cleanup_text({})
    assert result.ok
    assert result.speech == "I was not sure about my edit, so I left your text as it was."
    assert "Right" in keys(runs)
    assert "ctrl+v" not in keys(runs)
    assert writes == ["", "clip"]
    assert state[0] == "clip"


def test_new_phrasings_route_to_cleanup():
    for phrase in ("clean up this selection", "fix the paragraph", "clean this up", "clean it up"):
        assert route(phrase) == ToolCall("cleanup_text", {})
