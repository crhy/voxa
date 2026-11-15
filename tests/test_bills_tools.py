"""The bills tools: what they say, and the bill reminders they put in reminders.json.

A ``kind="bill"`` reminder is delivered by the existing reminder loop in the window
(``reminder_phrase``), so nothing in the UI had to change for a due date to fire.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from voxa.agent.reminders import Reminder, ReminderStore, reminder_phrase
from voxa.agent.tools.bills import (
    REACH_ERROR,
    _bills_due_handler,
    _bills_remind_handler,
    _subscriptions_list_handler,
    bills_tools,
)

FROZEN_TODAY = date(2026, 10, 10)

LOG = """- 2026-07-12 | 120.00 | home | Electricity
- 2026-08-12 | 120.00 | home | Electricity
- 2026-09-12 | 120.00 | home | Electricity
- 2026-10-12 | 120.00 | home | Electricity
- 2026-06-15 | 45.00 | home | Water
- 2026-07-15 | 45.00 | home | Water
- 2026-08-15 | 45.00 | home | Water
- 2026-09-15 | 45.00 | home | Water
- 2026-10-04 | 12.00 | food | Lunch
"""


@pytest.fixture(autouse=True)
def _spending_log(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "actions.jsonl"))
    monkeypatch.setattr("voxa.agent.tools.expenses._read", lambda: LOG)

    class _Today(date):
        @classmethod
        def today(cls) -> date:
            return FROZEN_TODAY

    monkeypatch.setattr("voxa.agent.tools.bills.date", _Today)
    return tmp_path


def test_bills_due_speaks_the_charge_coming_due() -> None:
    result = _bills_due_handler({})
    assert result.ok
    assert result.speech.startswith("One bill is due this week:")
    assert "your water bill" in result.speech
    assert "45 dollars" in result.speech
    assert "electricity" not in result.speech  # due on November 12th: not this week


def test_subscriptions_list_speaks_both_amounts() -> None:
    result = _subscriptions_list_handler({})
    assert result.ok
    assert result.speech == (
        "You pay for 2 subscriptions: water 45 dollars every month, electricity 120 dollars every month."
    )


def test_bills_remind_adds_one_reminder_per_subscription() -> None:
    first = _bills_remind_handler({})
    assert first.ok
    assert first.speech == "I'll remind you about 2 bills when they come due."
    waiting = [item for item in ReminderStore().upcoming() if item.kind == "bill"]
    assert len(waiting) == 2
    assert {item.id for item in waiting} == {"bill water", "bill electricity"}
    assert all(item.due.date() >= FROZEN_TODAY for item in waiting)

    second = _bills_remind_handler({})
    assert second.ok
    assert second.speech == "You are already set to be reminded about those bills."
    assert len([item for item in ReminderStore().upcoming() if item.kind == "bill"]) == 2


def test_the_bill_reminder_fires_when_its_date_arrives() -> None:
    _bills_remind_handler({})
    store = ReminderStore()
    fired = store.due(datetime(2026, 10, 15, 9, 0))
    assert [item.id for item in fired] == ["bill water"]
    assert reminder_phrase(fired[0].text) == "Reminder: your water bill is due."
    still = [item for item in store.upcoming() if item.kind == "bill"]
    assert [item.id for item in still] == ["bill electricity"]


def test_bill_reminders_survive_a_restart() -> None:
    _bills_remind_handler({})
    reloaded = ReminderStore()
    assert sorted(item.kind for item in reloaded.upcoming()) == ["bill", "bill"]


def test_bill_reminders_do_not_disturb_the_other_reminders() -> None:
    _bills_remind_handler({})
    store = ReminderStore()
    store.add(Reminder(id="r1", due=datetime(2026, 10, 11, 15, 0), text="call the dentist", kind="reminder"))
    assert [item.kind for item in store.due(datetime(2026, 10, 11, 15, 0))] == ["reminder"]
    assert len([item for item in store.upcoming() if item.kind == "bill"]) == 2


def test_a_log_that_cannot_be_read_fails_plainly(monkeypatch) -> None:
    def _boom() -> str:
        raise OSError("no documents folder")

    monkeypatch.setattr("voxa.agent.tools.expenses._read", _boom)
    for handler in (_bills_due_handler, _subscriptions_list_handler, _bills_remind_handler):
        result = handler({})
        assert not result.ok
        assert result.speech == REACH_ERROR


def test_no_recurring_charges_says_so(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.expenses._read", lambda: "- 2026-10-04 | 12.00 | food | Lunch\n")
    assert _bills_due_handler({}).speech == "Nothing you pay for is due this week."
    assert _subscriptions_list_handler({}).speech == (
        "I have not found any subscriptions in your spending log."
    )
    assert not _bills_remind_handler({}).ok


def test_bills_tools_registered() -> None:
    assert {tool.name for tool in bills_tools()} == {"bills_due", "subscriptions_list", "bills_remind"}
