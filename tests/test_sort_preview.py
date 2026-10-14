"""Folder-sort preview routes and works end to end."""

from __future__ import annotations

import subprocess

from voxa.agent.filematch import folder_key
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.filemanage import sort_folder_preview


def test_sort_phrases_route() -> None:
    assert route("how does my downloads folder sort") == ToolCall("sort_folder_preview", {"folder": "downloads"})
    assert route("sort my downloads folder") == ToolCall("sort_folder_preview", {"folder": "downloads"})
    assert route("preview my pictures folder") == ToolCall("sort_folder_preview", {"folder": "pictures"})


def test_sort_does_not_steal_other_routes() -> None:
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"


def test_sort_unknown_folder_not_routed() -> None:
    assert folder_key("notes") is None
    got = route("sort my notes")
    assert got is None or got.tool != "sort_folder_preview"


def test_sort_preview_handler(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: "/home/user/Downloads")
    monkeypatch.setattr(
        "voxa.agent.tools.filemanage._run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            "report.pdf\nphoto.png\narchive.zip\nnotes.txt\nsubdir/\n",
            "",
        ),
    )
    result = sort_folder_preview({"folder": "downloads"})
    assert result.ok
    assert result.speech == "Downloads has 4 files: 2 documents, 1 picture and 1 archive."


def test_sort_preview_handler_empty(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: "/home/user/Downloads")
    monkeypatch.setattr(
        "voxa.agent.tools.filemanage._run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    result = sort_folder_preview({"folder": "downloads"})
    assert result.ok
    assert result.speech == "Downloads is empty."


def test_sort_preview_handler_unknown_folder() -> None:
    result = sort_folder_preview({"folder": "notes"})
    assert not result.ok
    assert result.speech == "I do not know the folder notes."


def test_registry_has_sort_preview() -> None:
    assert "sort_folder_preview" in default_registry().names()
