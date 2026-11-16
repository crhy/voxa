"""The meeting note tools (crhy/voxa#65): routing, the archive, the calendar hand-off.

No microphone, no Whisper model and no network are involved: the transcription is a
fake callable, the store is a dict, and every command runs simulated.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import date, timedelta

from voxa.agent import meeting_notes as archive
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import calendar_local as calendar_tools
from voxa.agent.tools import default_registry
from voxa.agent.tools import meeting_notes as notes
from voxa.agent.tools.meeting_notes import (
    _meeting_note_actions_handler,
    _meeting_note_from_recording_handler,
    _meeting_note_recall_handler,
)

SAMPLE = (
    "Dana: We agreed to book the dentist on Friday.\n"
    "Me: OK, and we'll use the cheaper crown.\n"
    "Dana: I'll call the office back about the insurance.\n"
    "Me: I will send the invoice to Sam tomorrow.\n"
    "Dana: Let's pick up the kids on the way.\n"
)


def _archive_of(day: date) -> str:
    return archive.record("Dentist call", day, SAMPLE)


def test_the_three_tools_are_registered() -> None:
    names = default_registry().names()
    for name in ("meeting_note_from_recording", "meeting_note_recall", "meeting_note_actions"):
        assert name in names


def test_phrases_route() -> None:
    assert route("take notes on the dentist call") == ToolCall(
        "meeting_note_from_recording", {"name": "dentist call", "folder": ""}
    )
    assert route("what did we decide yesterday") == ToolCall("meeting_note_recall", {"day": "yesterday", "keyword": ""})
    assert route("what did we agree last week") == ToolCall("meeting_note_recall", {"day": "last week", "keyword": ""})
    assert route("add the action items from the dentist call to my calendar") == ToolCall(
        "meeting_note_actions", {"name": "dentist call", "day": "", "when": ""}
    )


def test_guard_phrases_keep_their_old_meaning() -> None:
    assert route("what did we decide") == ToolCall("conversation_summary", {})
    assert route("what are my action items from this conversation") == ToolCall("conversation_summary", {})
    assert route("read my notes") == ToolCall("read_notes", {})
    assert route("add buy stamps to my to-do list") == ToolCall("list_add", {"item": "buy stamps", "list": "to-do"})
    assert route("add dentist to my calendar tomorrow at 3") == ToolCall("calendar_add", {"title": "dentist", "when": "tomorrow at 3"})
    assert route("i spent 12 dollars on lunch") == ToolCall("expense_add", {"amount": "12 dollars", "what": "lunch"})


def test_a_recording_becomes_a_dated_record(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    store = {"content": ""}
    monkeypatch.setattr(notes, "_read", lambda: store["content"])
    monkeypatch.setattr(notes, "_write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr(notes, "_locate", lambda name, folder: "/home/rhy/Documents/Dentist call.mp3")
    monkeypatch.setattr(notes, "transcriber", lambda pcm: SAMPLE)

    result = _meeting_note_from_recording_handler({"name": "dentist call"})
    assert result.ok
    assert result.speech == (
        "I saved Dentist call in your meeting notes: 2 decisions and 3 action items. "
        "The transcript stays on this machine. Dana and Me spoke."
    )
    assert f"## {date.today():%Y-%m-%d} | Dentist call" in store["content"]
    assert "Transcript:\nDana: We agreed to book the dentist on Friday." in store["content"]


def test_a_recording_needs_a_file_and_a_model(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr(notes, "_locate", lambda name, folder: None)
    assert _meeting_note_from_recording_handler({"name": "nothing"}).speech == "I can't find a recording called nothing."

    monkeypatch.setattr(notes, "_locate", lambda name, folder: "/home/rhy/Documents/notes.txt")
    assert not _meeting_note_from_recording_handler({"name": "notes"}).ok

    monkeypatch.setattr(notes, "_locate", lambda name, folder: "/home/rhy/Documents/Call.mp3")
    monkeypatch.setattr(notes, "transcriber", None)
    monkeypatch.setattr("voxa.agent.tools.transcribe_file.transcriber", None)
    assert _meeting_note_from_recording_handler({"name": "call"}).speech == "My hearing model is still loading."


def test_recall_answers_yesterday(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr(notes, "_read", lambda: _archive_of(date.today() - timedelta(days=1)))

    result = _meeting_note_recall_handler({"day": "yesterday", "keyword": ""})
    assert result.ok
    assert result.speech.startswith("You have 1 meeting note for that: Dentist call yesterday.")
    assert "2 decisions: We agreed to book the dentist on Friday." in result.speech

    by_word = _meeting_note_recall_handler({"day": "", "keyword": "dentist"})
    assert by_word.ok and "Dentist call yesterday" in by_word.speech
    assert not _meeting_note_recall_handler({"day": "", "keyword": "nothing like this"}).speech.startswith("You have")


def test_recall_without_any_notes(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr(notes, "_read", lambda: "")
    assert _meeting_note_recall_handler({"day": "yesterday", "keyword": ""}).speech == "I have no meeting notes yet."


def test_action_items_reach_the_calendar(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    calendar = {"content": ""}
    monkeypatch.setattr(calendar_tools, "_read", lambda: calendar["content"])
    monkeypatch.setattr(calendar_tools, "_write", lambda content: calendar.__setitem__("content", content))
    monkeypatch.setattr(notes, "_read", lambda: _archive_of(date.today() - timedelta(days=1)))

    result = _meeting_note_actions_handler({"day": "yesterday", "name": "", "when": "tomorrow at 3"})
    assert result.ok
    assert result.speech == "I put 3 action items on your calendar for tomorrow."
    assert len([line for line in calendar["content"].splitlines() if line.startswith("- ")]) == 3
    assert "I'll call the office back about the insurance." in calendar["content"]


def test_a_meeting_without_action_items_writes_nothing(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    quiet = "Dana: We agreed to meet on Friday.\nMe: Nothing to do afterwards."
    monkeypatch.setattr(notes, "_read", lambda: archive.record("Quiet call", date.today() - timedelta(days=1), quiet))
    written: list[str] = []
    monkeypatch.setattr(calendar_tools, "_read", lambda: "")
    monkeypatch.setattr(calendar_tools, "_write", lambda content: written.append(content))
    result = _meeting_note_actions_handler({"day": "yesterday", "name": "", "when": ""})
    assert result.ok
    assert result.speech == "I heard no action items in those notes."
    assert written == []


def test_neither_module_reaches_the_network() -> None:
    """The transcript never leaves the machine: no HTTP, no sockets, no downloader."""
    forbidden = {"aiohttp", "requests", "urllib", "http", "socket", "ftplib", "smtplib"}
    for module in (archive, notes):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
        assert imported & forbidden == set(), (module.__name__, imported & forbidden)
        source = inspect.getsource(module)
        for word in ("aiohttp", "requests", "urllib", "http://", "https://"):
            assert word not in source
        assert not re.search(r"\b(?:curl|wget|nc|ssh)\b", source)
