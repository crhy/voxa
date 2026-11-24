"""The reusable spend guard (issue #43): read-back, a "confirm" within 20 s,
per-order and per-day caps, and one action-log record per allowed spend."""

from __future__ import annotations

from pathlib import Path

from voxa.agent import ui
from voxa.agent.actionlog import ActionLog
from voxa.agent.policy import IntendedSpend, day_spend_total, spend_guard
from voxa.agent.tools import uicontrol
from voxa.config import Settings


def _spend(price: float = 20.0, **kwargs) -> IntendedSpend:
    return IntendedSpend(what="two tickets", price=price, source="Example Store", payee="Acme", **kwargs)


def test_unconfirmed_spend_is_blocked() -> None:
    decision = spend_guard(_spend(), confirm_word="", per_order_cap=50, per_day_cap=200)
    assert not decision.allowed
    # The read-back names what, the price, the source and the payee.
    assert decision.readback == "two tickets costs 20.00, from Example Store, to Acme"
    assert "confirm" in decision.reason


def test_confirm_within_twenty_seconds_is_allowed() -> None:
    decision = spend_guard(_spend(), confirm_word="Confirm", confirm_seconds=19.9, per_order_cap=50, per_day_cap=200)
    assert decision.allowed
    assert decision.reason == ""


def test_confirm_after_the_window_is_refused() -> None:
    decision = spend_guard(_spend(), confirm_word="confirm", confirm_seconds=20.5, per_order_cap=50, per_day_cap=200)
    assert not decision.allowed
    assert "too late" in decision.reason


def test_per_order_cap_blocks_an_over_cap_order() -> None:
    decision = spend_guard(
        _spend(price=60.0), confirm_word="confirm", confirm_seconds=5, per_order_cap=50, per_day_cap=200
    )
    assert not decision.allowed
    assert "one order" in decision.reason


def test_per_day_total_cap_blocks_once_the_day_is_exceeded() -> None:
    decision = spend_guard(
        _spend(price=60.0),
        confirm_word="confirm",
        confirm_seconds=5,
        per_order_cap=100,
        per_day_cap=100,
        spent_today=60.0,
    )
    assert not decision.allowed
    assert "today" in decision.reason


def test_day_spend_total_sums_today_only_from_the_log(tmp_path: Path) -> None:
    from datetime import date

    from voxa.agent.actionlog import ActionRecord

    log = ActionLog(tmp_path / "actions.jsonl")
    for record in (
        ActionRecord("2026-10-03T09:00:00", "buy", "spend", args={"amount": "12.50"}, ok=True),
        ActionRecord("2026-10-03T12:00:00", "pay", "spend", args={"amount": "7.50"}, ok=True),
        ActionRecord("2026-10-04T09:00:00", "buy", "spend", args={"amount": "99.00"}, ok=True),
        ActionRecord("2026-10-03T13:00:00", "buy", "spend", args={"amount": "500.00"}, ok=False),
        ActionRecord("2026-10-03T14:00:00", "open", "tool", args={"amount": "500.00"}, ok=True),
    ):
        log.append(record)
    assert day_spend_total(log, date(2026, 10, 3)) == 20.0


def test_allowed_spend_presses_and_logs_exactly_one_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "actions.jsonl"))
    monkeypatch.setattr(uicontrol, "ConfigStore", lambda: _FakeStore(Settings(spend_per_order_cap=50, spend_per_day_cap=200)))
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "Shop")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": True, "pressed": {"name": label}})

    result = uicontrol.press_button({"label": "buy", "price": "20", "payee": "Acme", "confirm": "confirm", "confirm_seconds": "5"})

    assert result.ok
    assert result.speech == "Pressed buy."
    records = ActionLog(tmp_path / "actions.jsonl").read()
    assert len(records) == 1
    assert records[0]["route"] == "spend"
    assert records[0]["args"] == {"amount": "20.00", "payee": "Acme", "reference": ""}


def test_unconfirmed_spend_is_blocked_by_press_button(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "actions.jsonl"))
    monkeypatch.setattr(uicontrol, "ConfigStore", lambda: _FakeStore(Settings()))
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "Shop")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": True, "pressed": {"name": label}})

    result = uicontrol.press_button({"label": "buy", "price": "20", "payee": "Acme"})

    assert not result.ok
    assert ActionLog(tmp_path / "actions.jsonl").read() == []


class _FakeStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self) -> Settings:
        return self._settings
