"""Reminder tool handlers produce the right speech and mutate the store."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxa.agent.tools.reminders import (
    _cancel_reminders_handler,
    _list_reminders_handler,
    _set_reminder_handler,
    _set_timer_handler,
    reminders_tools,
)


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "actions.jsonl"))


def test_set_timer_speech() -> None:
    result = _set_timer_handler({"duration": "10 minutes"})
    assert result.ok
    assert result.speech == "Timer set for 10 minutes."


def test_set_timer_rejects_garbage() -> None:
    result = _set_timer_handler({"duration": "banana"})
    assert not result.ok


def test_set_reminder_speech() -> None:
    result = _set_reminder_handler({"when": "at 3pm", "text": "call the dentist"})
    assert result.ok
    assert "call the dentist" in result.speech


def test_list_then_cancel() -> None:
    _set_timer_handler({"duration": "10 minutes"})
    _set_reminder_handler({"when": "at 3pm", "text": "stretch"})
    listed = _list_reminders_handler({})
    assert listed.ok and "2 reminders" in listed.speech
    cancelled = _cancel_reminders_handler({})
    assert cancelled.ok and "Cancelled 2 reminders." == cancelled.speech
    assert _list_reminders_handler({}).speech == "You have no reminders."


def test_reminders_tools_registered() -> None:
    names = {t.name for t in reminders_tools()}
    assert names == {"set_timer", "set_reminder", "list_reminders", "cancel_reminders"}
