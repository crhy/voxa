"""Fixes from the live spoken run of 2026-10-10 19:05: pomodoro spellings, focus echo, northern lights."""

from __future__ import annotations

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.reminders import ReminderStore
from voxa.agent.tools.reminders import (
    _focus_start_handler,
)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch) -> ReminderStore:
    store = ReminderStore(tmp_path / "r.json")

    def _open_store() -> ReminderStore:
        return store

    monkeypatch.setattr("voxa.agent.tools.reminders.ReminderStore", _open_store)
    return store


def test_pomodoro_misspellings_route_to_focus_start() -> None:
    assert route("Start a pommodoro") == ToolCall("focus_start", {"duration": "", "what": ""})
    assert route("start a pomodoro timer") == ToolCall("focus_start", {"duration": "", "what": ""})
    assert route("begin a pommadoro session") == ToolCall("focus_start", {"duration": "", "what": ""})


def test_focus_start_says_the_task_back() -> None:
    result = _focus_start_handler({"duration": "10 minutes", "what": "writing the report"})
    assert result.ok
    assert result.speech == "Focus time started: 10 minutes for writing the report. I'll tell you when it's over."
    without = _focus_start_handler({"duration": "10 minutes", "what": ""})
    assert without.speech == "Focus time started: 10 minutes. I'll tell you when it's over."


def test_northern_lengths_heard_as_northern_lights() -> None:
    call = route("Search for images of the northern lengths")
    assert call is not None
    assert call.tool == "image_search"
    assert call.args["query"] == "the northern lights"


def test_ordinary_requests_still_route() -> None:
    read = route("what's on my to-do list")
    assert read is not None and read.tool == "list_read"
    time_call = route("what time is it")
    assert time_call is not None and time_call.tool == "tell_time"
    assert route("what is the capital of France") is None
    app = route("open the calculator")
    assert app is not None and app.tool == "open_app"
