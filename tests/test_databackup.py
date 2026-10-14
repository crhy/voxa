"""Data-backup capability routes and works end to end."""

from __future__ import annotations

import subprocess
from datetime import datetime

from voxa.agent.databackup import backup_dir_name, parse_listing, spoken_done, spoken_last
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.databackup import _backup_last_handler, _backup_now_handler


def test_backup_phrases_route() -> None:
    assert route("back up my data") == ToolCall("backup_now", {})
    assert route("back up everything") == ToolCall("backup_now", {})
    assert route("make a backup") == ToolCall("backup_now", {})
    assert route("create a backup") == ToolCall("backup_now", {})
    assert route("when was my last backup") == ToolCall("backup_last", {})
    assert route("when did the last backup run") == ToolCall("backup_last", {})
    assert route("do i have a backup") == ToolCall("backup_last", {})


def test_backup_does_not_steal_other_routes() -> None:
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"


def test_backup_dir_name() -> None:
    assert backup_dir_name(datetime(2026, 10, 10, 14, 5)) == "2026-10-10 1405"


def test_parse_listing() -> None:
    text = "2026-10-10 1405\nnotes\n2026-10-09 0900\nrandom"
    assert parse_listing(text) == ["2026-10-09 0900", "2026-10-10 1405"]


def test_spoken_done() -> None:
    assert spoken_done(1, "x") == "I backed up 1 file into x."
    assert spoken_done(3, "x") == "I backed up 3 files into x."


def test_spoken_last() -> None:
    now = datetime(2026, 10, 11, 14, 5)
    assert spoken_last([], now) == "You have no backups yet."
    assert spoken_last(["2026-10-11 1405"], now) == "Your last backup was today at 2:05 PM."
    assert spoken_last(["2026-10-09 0900", "2026-10-10 1405"], now) == "Your last backup was yesterday at 2:05 PM."
    assert spoken_last(["2026-10-01 0900"], now) == "Your last backup was 10 days ago."


def test_backup_now_handler(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: "/home/user/Documents")
    monkeypatch.setattr("voxa.agent.tools.databackup.backup_dir_name", lambda now: "2026-10-11 1405")
    monkeypatch.setattr(
        "voxa.agent.tools.databackup._run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "3\n", ""),
    )
    result = _backup_now_handler({})
    assert result.ok
    assert result.speech == "I backed up 3 files into 2026-10-11 1405."


def test_backup_now_handler_nothing_to_back_up(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: "/home/user/Documents")
    monkeypatch.setattr(
        "voxa.agent.tools.databackup._run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "0\n", ""),
    )
    result = _backup_now_handler({})
    assert not result.ok
    assert result.speech == "There is nothing to back up yet."


def test_backup_last_handler(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: "/home/user/Documents")
    monkeypatch.setattr(
        "voxa.agent.tools.databackup._run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "2026-10-10 1405\nnotes\n", ""),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.databackup.datetime",
        type("FakeDT", (), {"now": staticmethod(lambda: datetime(2026, 10, 11, 14, 5))}),
    )
    result = _backup_last_handler({})
    assert result.ok
    assert result.speech == "Your last backup was yesterday at 2:05 PM."


def test_registry_has_backup_tools() -> None:
    names = default_registry().names()
    assert "backup_now" in names
    assert "backup_last" in names
