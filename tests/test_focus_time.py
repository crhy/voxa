"""Focus sessions by voice: handlers, store mutations, routing and guards."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.reminders import Reminder, ReminderStore
from voxa.agent.tools.reminders import (
    _focus_start_handler,
    _focus_status_handler,
    _focus_stop_handler,
    reminders_tools,
)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch) -> ReminderStore:
    store = ReminderStore(tmp_path / "r.json")

    def _open_store() -> ReminderStore:
        return store

    monkeypatch.setattr("voxa.agent.tools.reminders.ReminderStore", _open_store)
    return store


def test_focus_start_speech(store: ReminderStore) -> None:
    result = _focus_start_handler({"duration": "25 minutes", "what": "laundry"})
    assert result.ok
    assert result.speech == "Focus time started: 25 minutes for laundry. I'll tell you when it's over."
    focus = [r for r in store.upcoming() if r.kind == "focus"]
    assert len(focus) == 1
    assert focus[0].text == "laundry"


def test_focus_start_defaults_to_25_minutes(store: ReminderStore) -> None:
    result = _focus_start_handler({"duration": "", "what": ""})
    assert result.ok
    assert result.speech == "Focus time started: 25 minutes. I'll tell you when it's over."
    focus = [r for r in store.upcoming() if r.kind == "focus"]
    assert len(focus) == 1
    assert focus[0].text == ""


def test_focus_start_replaces_existing_focus(store: ReminderStore) -> None:
    _focus_start_handler({"duration": "25 minutes", "what": "laundry"})
    _focus_start_handler({"duration": "an hour", "what": "reading"})
    focus = [r for r in store.upcoming() if r.kind == "focus"]
    assert len(focus) == 1
    assert focus[0].text == "reading"


def test_focus_start_rejects_garbage() -> None:
    result = _focus_start_handler({"duration": "banana", "what": ""})
    assert not result.ok
    assert result.speech == "I couldn't understand that duration."


def test_focus_status_rounds_up(store: ReminderStore) -> None:
    now = datetime.now()
    store.add(Reminder(id="f1", due=now + timedelta(seconds=12 * 60 - 30), text="reading", kind="focus"))
    result = _focus_status_handler({})
    assert result.ok
    assert result.speech == "You have 12 minutes of focus time left."


def test_focus_status_singular_minute(store: ReminderStore) -> None:
    now = datetime.now()
    store.add(Reminder(id="f2", due=now + timedelta(seconds=50), text="", kind="focus"))
    result = _focus_status_handler({})
    assert result.ok
    assert result.speech == "You have 1 minute of focus time left."


def test_focus_status_without_session(store: ReminderStore) -> None:
    result = _focus_status_handler({})
    assert result.ok
    assert result.speech == "You are not in a focus session."


def test_focus_stop_ends_session(store: ReminderStore) -> None:
    _focus_start_handler({"duration": "25 minutes", "what": "laundry"})
    result = _focus_stop_handler({})
    assert result.ok
    assert result.speech == "Focus time ended."
    assert [r for r in store.upcoming() if r.kind == "focus"] == []


def test_focus_stop_without_session(store: ReminderStore) -> None:
    result = _focus_stop_handler({})
    assert result.ok
    assert result.speech == "You are not in a focus session."


def test_route_focus_start() -> None:
    assert route("Start a 25 minute focus session") == ToolCall(
        "focus_start", {"duration": "25 minute", "what": ""}
    )
    assert route("start a focus session") == ToolCall("focus_start", {"duration": "", "what": ""})
    assert route("begin a focus timer for 30 minutes on laundry") == ToolCall(
        "focus_start", {"duration": "30 minutes", "what": "laundry"}
    )
    assert route("i need to focus for an hour") == ToolCall("focus_start", {"duration": "an hour", "what": ""})
    assert route("start a pomodoro") == ToolCall("focus_start", {"duration": "", "what": ""})
    assert route("protect the next two hours for laundry") == ToolCall(
        "focus_start", {"duration": "two hours", "what": "laundry"}
    )


def test_route_focus_status() -> None:
    assert route("how much focus time is left") == ToolCall("focus_status", {})
    assert route("how long until my break") == ToolCall("focus_status", {})
    assert route("am i still in a focus session") == ToolCall("focus_status", {})


def test_route_focus_stop() -> None:
    assert route("stop the focus session") == ToolCall("focus_stop", {})
    assert route("cancel my focus time") == ToolCall("focus_stop", {})
    assert route("i'm done focusing") == ToolCall("focus_stop", {})


def test_guards_still_route() -> None:
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
    assert route("set a timer for two minutes") == ToolCall("set_timer", {"duration": "two minutes"})
    assert route("cancel my reminders").tool == "cancel_reminders"


def test_focus_tools_registered() -> None:
    names = {t.name for t in reminders_tools()}
    assert {"focus_start", "focus_status", "focus_stop"} <= names
