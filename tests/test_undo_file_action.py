from __future__ import annotations

import subprocess

import pytest

from voxa.agent.intents import route
from voxa.agent.tools import filemanage


def fake_run(recorded, names_map, fail_on=None):
    def run(command, stdin=None, check=True):
        argv = list(command)
        if fail_on is not None and argv == fail_on:
            raise subprocess.CalledProcessError(1, argv, stderr=b"boom")
        recorded.append(argv)
        if argv[0] == "sh":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "xdg-user-dir":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "ls":
            folder = argv[-1]
            return subprocess.CompletedProcess(argv, 0, "\n".join(names_map.get(folder, [])), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


NAMES = {
    "/home/u/Downloads": ["report.pdf", "old notes.txt"],
    "/home/u/Documents": ["report.pdf"],
}


def fresh(monkeypatch, fail_on=None):
    recorded: list[list[str]] = []
    monkeypatch.setattr(filemanage, "LAST_UNDO", None)
    monkeypatch.setattr(filemanage, "_run", fake_run(recorded, NAMES, fail_on))
    return recorded


def test_move_then_undo(monkeypatch):
    recorded = fresh(monkeypatch)
    move = filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    assert move.ok
    before = len(recorded)
    undo = filemanage.undo_file_action({})
    assert undo.ok
    assert undo.speech == "Moved report.pdf back to downloads."
    assert "back to downloads" in undo.speech
    assert recorded[before:] == [["gio", "move", "/home/u/Documents/report.pdf", "/home/u/Downloads/"]]


def test_copy_then_undo(monkeypatch):
    recorded = fresh(monkeypatch)
    copy = filemanage.copy_file({"name": "report", "source": "downloads", "destination": "documents"})
    assert copy.ok
    before = len(recorded)
    undo = filemanage.undo_file_action({})
    assert undo.ok
    assert undo.speech == "Removed the copy of report.pdf."
    assert recorded[before:] == [["gio", "trash", "/home/u/Documents/report.pdf"]]


def test_rename_then_undo(monkeypatch):
    recorded = fresh(monkeypatch)
    rename = filemanage.rename_file({"name": "report", "folder": "downloads", "new_name": "final"})
    assert rename.ok
    before = len(recorded)
    undo = filemanage.undo_file_action({})
    assert undo.ok
    assert undo.speech == "Renamed it back to report.pdf."
    assert recorded[before:] == [["gio", "rename", "/home/u/Downloads/final.pdf", "report.pdf"]]


def test_make_folder_then_undo(monkeypatch):
    recorded = fresh(monkeypatch)
    made = filemanage.make_folder({"name": "taxes", "folder": "documents"})
    assert made.ok
    before = len(recorded)
    undo = filemanage.undo_file_action({})
    assert undo.ok
    assert undo.speech == "Removed the folder Taxes."
    assert recorded[before:] == [["rmdir", "--", "/home/u/Documents/Taxes"]]


def test_trash_then_undo(monkeypatch):
    recorded = fresh(monkeypatch)
    trashed = filemanage.trash_file({"name": "old notes", "folder": "downloads"})
    assert trashed.ok
    before = len(recorded)
    undo = filemanage.undo_file_action({})
    assert undo.ok
    assert undo.speech == "Put old notes.txt back."
    assert recorded[before:] == [["gio", "trash", "--restore", "trash:///old notes.txt"]]


def test_undo_twice(monkeypatch):
    fresh(monkeypatch)
    filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    first = filemanage.undo_file_action({})
    assert first.ok
    second = filemanage.undo_file_action({})
    assert not second.ok
    assert second.speech == "There is no file action to undo."


def test_failed_move_does_not_set_undo(monkeypatch):
    fresh(monkeypatch, fail_on=["gio", "move", "/home/u/Downloads/report.pdf", "/home/u/Documents/"])
    move = filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    assert not move.ok
    assert filemanage.LAST_UNDO is None


def test_empty_trash_clears_undo(monkeypatch):
    fresh(monkeypatch)
    filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    assert filemanage.LAST_UNDO is not None
    filemanage.empty_trash({})
    assert filemanage.LAST_UNDO is None


def test_find_file_does_not_clear_undo(monkeypatch):
    fresh(monkeypatch)
    filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    assert filemanage.LAST_UNDO is not None
    filemanage.find_file({"name": "report"})
    assert filemanage.LAST_UNDO is not None
    undo = filemanage.undo_file_action({})
    assert undo.ok


def test_undo_command_failing(monkeypatch):
    fresh(monkeypatch, fail_on=["gio", "move", "/home/u/Documents/report.pdf", "/home/u/Downloads/"])
    filemanage.move_file({"name": "report", "source": "downloads", "destination": "documents"})
    undo = filemanage.undo_file_action({})
    assert not undo.ok
    assert undo.speech == "That did not work: boom"
    assert filemanage.LAST_UNDO is None


@pytest.mark.parametrize(
    "phrase",
    [
        "undo that",
        "undo the move",
        "undo the copy",
        "undo the rename",
        "undo the delete",
        "undo the last file action",
        "put it back",
        "put that back",
        "put the file back",
        "move it back",
        "bring it back",
        "bring that back",
        "restore it",
        "restore that",
        "restore the file",
    ],
)
def test_router_undo(monkeypatch, phrase):
    call = route(phrase)
    assert call is not None
    assert call.tool == "undo_file_action"


def test_router_bare_undo_is_not_undo(monkeypatch):
    call = route("undo")
    assert call is not None
    assert call.tool != "undo_file_action"


def test_undo_that_without_a_file_action_is_the_ordinary_undo(monkeypatch):
    from voxa.agent import intents
    from voxa.agent.result import ToolResult
    from voxa.agent.tools import filemanage, typing

    monkeypatch.setattr(filemanage, "LAST_UNDO", None)
    pressed = []
    monkeypatch.setattr(typing, "_press_key_handler", lambda args: pressed.append(args) or ToolResult.success(""))
    call = intents.route("undo that")
    assert call.tool == "undo_file_action" and call.args == {"otherwise": "key"}
    assert filemanage.undo_file_action(call.args).ok
    assert pressed == [{"key": "undo"}]
    assert intents.route("put it back").args == {}
    assert filemanage.undo_file_action({}).speech == "There is no file action to undo."
