"""Timers and reminders: parse_when table, store add/due/persist/remove."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from voxa.agent.reminders import Reminder, ReminderStore, parse_when, reminder_phrase, timer_phrase

NOW = datetime(2026, 10, 4, 14, 0, 0)

WHEN_ROWS: list[tuple[str, datetime]] = [
    ("in 10 minutes", datetime(2026, 10, 4, 14, 10)),
    ("in an hour", datetime(2026, 10, 4, 15, 0)),
    ("in one hour", datetime(2026, 10, 4, 15, 0)),
    ("in two hours", datetime(2026, 10, 4, 16, 0)),
    ("in 2 hours", datetime(2026, 10, 4, 16, 0)),
    ("in 30 minutes", datetime(2026, 10, 4, 14, 30)),
    ("in 15 minutes", datetime(2026, 10, 4, 14, 15)),
    ("in 5 minutes", datetime(2026, 10, 4, 14, 5)),
    ("in a day", datetime(2026, 10, 5, 14, 0)),
    ("in 3 days", datetime(2026, 10, 7, 14, 0)),
    ("at 3pm", datetime(2026, 10, 4, 15, 0)),
    ("at 3:30pm", datetime(2026, 10, 4, 15, 30)),
    ("at 11pm", datetime(2026, 10, 4, 23, 0)),
    ("at 14:00", datetime(2026, 10, 5, 14, 0)),
    ("at 9am", datetime(2026, 10, 5, 9, 0)),
    ("at 12am", datetime(2026, 10, 5, 0, 0)),
    ("tomorrow at 9", datetime(2026, 10, 5, 9, 0)),
    ("tonight at 8", datetime(2026, 10, 4, 20, 0)),
    ("3pm", datetime(2026, 10, 4, 15, 0)),
    ("in ten minutes", datetime(2026, 10, 4, 14, 10)),
]


@pytest.mark.parametrize(("spoken", "expected"), WHEN_ROWS)
def test_parse_when_understands_spoken_times(spoken: str, expected: datetime) -> None:
    assert parse_when(spoken, NOW) == expected, spoken


def test_parse_when_rejects_garbage() -> None:
    assert parse_when("banana", NOW) is None
    assert parse_when("", NOW) is None
    assert parse_when("at 99pm", NOW) is None


def test_phrases_are_stable() -> None:
    assert reminder_phrase("stretch") == "Reminder: stretch."
    assert timer_phrase("10 minutes") == "Your 10 minute timer is done."


def test_store_add_and_due(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    store.add(Reminder(id="a", due=datetime(2026, 10, 4, 14, 5), text="stretch", kind="reminder"))
    store.add(Reminder(id="b", due=datetime(2026, 10, 4, 16, 0), text="call dentist", kind="reminder"))
    ready = store.due(datetime(2026, 10, 4, 14, 30))
    assert [r.id for r in ready] == ["a"]
    assert [r.id for r in store.upcoming()] == ["b"]


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    store = ReminderStore(path)
    store.add(Reminder(id="x", due=datetime(2026, 10, 4, 15, 0), text="hi", kind="timer"))
    reopened = ReminderStore(path)
    assert [r.id for r in reopened.upcoming()] == ["x"]


def test_store_remove(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    store.add(Reminder(id="y", due=datetime(2026, 10, 4, 15, 0), text="hi", kind="reminder"))
    assert store.remove("y") is True
    assert store.remove("nope") is False
    assert store.upcoming() == []
