from datetime import datetime, timedelta

from voxa.agent import calc
from voxa.agent.reminders import parse_when

NOW = datetime(2026, 10, 10, 11, 0)


def test_a_duration_without_in_is_understood():
    assert parse_when("two hours", NOW) == NOW + timedelta(hours=2)
    assert parse_when("ten minutes", NOW) == NOW + timedelta(minutes=10)
    assert parse_when("an hour and 15 minutes", NOW) == NOW + timedelta(minutes=75)
    assert parse_when("in two hours", NOW) == NOW + timedelta(hours=2)
    assert parse_when("5 pm", NOW) == NOW.replace(hour=17)
    assert parse_when("the two hours of sleep", NOW) is None


def test_a_written_percent_sign_is_read_as_percent():
    assert calc.answer("what is 15% of 80") == "15 percent of 80 is 12."


def test_a_reminder_added_by_another_store_is_delivered(tmp_path):
    import os

    from voxa.agent.reminders import Reminder, ReminderStore

    path = tmp_path / "reminders.json"
    window_store = ReminderStore(path)          # the long-lived one that delivers
    assert window_store.due(NOW) == []
    ReminderStore(path).add(Reminder(id="r1", due=NOW, text="stretch", kind="reminder"))  # a voice command
    os.utime(path, (1, 1))                      # same-second writes must still be noticed
    assert [r.text for r in window_store.due(NOW + timedelta(seconds=1))] == ["stretch"]
    assert ReminderStore(path).upcoming() == []
