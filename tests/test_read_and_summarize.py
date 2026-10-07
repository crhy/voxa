from __future__ import annotations

import subprocess

import voxa.agent.tools.textedit as textedit
from voxa.agent.intents import route


def _install(monkeypatch, clipboard, selected, calls, model=None):
    monkeypatch.setattr(textedit, "read_clipboard", lambda: clipboard[0])
    monkeypatch.setattr(textedit, "write_clipboard", lambda value: clipboard.__setitem__(0, value))

    def recorder(command, *args, **kwargs):
        calls.append(list(command))
        if "ctrl+c" in command:
            clipboard[0] = selected
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(textedit, "_run", recorder)
    monkeypatch.setattr(textedit.time, "sleep", lambda s: None)
    if model is not None:
        monkeypatch.setattr(textedit, "ask_model", model)


def test_speakable_short_unchanged():
    assert textedit.speakable("hello   world") == "hello world"


def test_speakable_long_cut_at_sentence_end():
    text = " ".join(f"This is sentence number {i} right here." for i in range(120))
    out = textedit.speakable(text, 900)
    suffix = " … That is the first part."
    assert out.endswith(suffix)
    assert len(out) < 900 + 40
    body = out[: -len(suffix)]
    assert body.endswith(".")


def test_read_selection_happy_path(monkeypatch):
    clipboard = ["saved clipboard"]
    selected = "the chosen words"
    calls = []
    _install(monkeypatch, clipboard, selected, calls)
    result = textedit.read_selection({})
    assert result.ok
    assert result.speech == "the chosen words"
    assert clipboard[0] == "saved clipboard"
    assert all("ctrl+a" not in cmd and "ctrl+v" not in cmd for cmd in calls)


def test_read_selection_nothing_selected(monkeypatch):
    clipboard = ["saved clipboard"]
    selected = "   "
    calls = []
    _install(monkeypatch, clipboard, selected, calls)
    result = textedit.read_selection({})
    assert not result.ok
    assert result.speech == "Select some text first, then ask me to read it."
    assert clipboard[0] == "saved clipboard"


def test_read_selection_hooks_missing():
    result = textedit.read_selection({})
    assert not result.ok
    assert result.speech == "I cannot reach the clipboard here."


def test_read_clipboard_aloud_with_text(monkeypatch):
    clipboard = ["  hello there  "]
    _install(monkeypatch, clipboard, "", [])
    result = textedit.read_clipboard_aloud({})
    assert result.ok
    assert result.speech == "The clipboard says: hello there"


def test_read_clipboard_aloud_empty(monkeypatch):
    clipboard = ["   "]
    _install(monkeypatch, clipboard, "", [])
    result = textedit.read_clipboard_aloud({})
    assert not result.ok
    assert result.speech == "The clipboard is empty."


def test_summarize_happy_path(monkeypatch):
    clipboard = ["saved clipboard"]
    selected = "Some text the user selected."
    calls = []
    seen = []

    def model(messages):
        seen.append(messages)
        return "A short summary."

    _install(monkeypatch, clipboard, selected, calls, model=model)
    result = textedit.summarize_selection({})
    assert result.ok
    assert result.speech == "A short summary."
    assert seen[0][1]["content"] == selected
    assert seen[0][0]["content"] == textedit.SUMMARY_PROMPT
    assert clipboard[0] == "saved clipboard"


def test_summarize_empty_summary(monkeypatch):
    clipboard = ["saved clipboard"]
    selected = "Some text the user selected."
    calls = []

    def model(messages):
        return "   "

    _install(monkeypatch, clipboard, selected, calls, model=model)
    result = textedit.summarize_selection({})
    assert not result.ok
    assert result.speech == "I could not summarise that."


def test_summarize_too_long(monkeypatch):
    clipboard = ["saved clipboard"]
    selected = "x" * 20001
    calls = []
    _install(monkeypatch, clipboard, selected, calls)
    result = textedit.summarize_selection({})
    assert not result.ok
    assert result.speech == "That is too much text for me to summarise in one go."


def test_summarize_hooks_missing():
    result = textedit.summarize_selection({})
    assert not result.ok
    assert result.speech == "I cannot reach the clipboard here."


def test_router_read_selection_phrases():
    for phrase in (
        "read that",
        "read that to me",
        "read the selection",
        "read the selected text",
        "read what i selected",
    ):
        call = route(phrase)
        assert call is not None and call.tool == "read_selection"


def test_router_read_clipboard_phrases():
    for phrase in (
        "what's on my clipboard",
        "what is in the clipboard",
        "read my clipboard",
        "read the clipboard to me",
    ):
        call = route(phrase)
        assert call is not None and call.tool == "read_clipboard_aloud"


def test_router_summarize_phrases():
    for phrase in (
        "summarize that",
        "summarise this",
        "sum up the selection",
        "give me a summary of that",
    ):
        call = route(phrase)
        assert call is not None and call.tool == "summarize_selection"


def test_router_page_phrases_unchanged():
    for phrase in ("read this page", "read the page", "read this to me"):
        call = route(phrase)
        assert call is not None and call.tool == "read_page"
