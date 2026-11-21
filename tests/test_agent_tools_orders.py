from __future__ import annotations

from datetime import date, timedelta

import pytest

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

from voxa.agent import orders as ledger_module
from voxa.agent.intents import ToolCall, route
from voxa.agent.policy import RiskLevel
from voxa.agent.tools import default_registry
from voxa.agent.tools import orders as orders_tools

EMAIL = """Thanks for your purchase.
Your order for filament has been shipped.
Tracking number: 1Z999AA10123456784
A second parcel went out under FedEx 321987654321.
Expected delivery: November 18, 2026
Call 850-123-4567 or open https://shop.example.com/orders/123456789012
"""

UPS_URL = "https://parcelsapp.com/en/tracking/1Z999AA10123456784"
FEDEX_URL = "https://parcelsapp.com/en/tracking/321987654321"


def page(number: str, status: str) -> str:
    return f"<html><body><script>var legend='delivered';</script>Tracking {number}: {status}</body></html>"


@pytest.fixture
def ledger(monkeypatch):
    store = {"content": ""}
    monkeypatch.setattr(orders_tools, "_read", lambda: store["content"])
    monkeypatch.setattr(orders_tools, "_write", lambda content: store.__setitem__("content", content))
    monkeypatch.setattr(orders_tools.filemanage, "_folder", lambda spoken: "/tmp/docs")
    opened: list[str] = []
    monkeypatch.setattr(orders_tools, "_open", lambda url: opened.append(url) or True)
    return store, opened


def test_pasted_text_speaks_the_status_it_read_and_saves_the_list(ledger, monkeypatch) -> None:
    store, opened = ledger
    monkeypatch.setattr(
        orders_tools,
        "fetch",
        lambda url: page(url.rsplit("/", 1)[-1], "out for delivery"),
    )

    result = orders_tools.track_order({"text": EMAIL})

    assert result.ok
    assert result.speech == (
        "Tracking 1Z999AA10123456784 for UPS: out for delivery. "
        "Tracking 321987654321 for FedEx: out for delivery. "
        "Saved 2 orders on your order list."
    )
    assert opened == [UPS_URL, FEDEX_URL]
    rows = ledger_module.parse(store["content"])
    assert [row["number"] for row in rows] == ["1Z999AA10123456784", "321987654321"]
    assert rows[0]["carrier"] == "UPS"
    assert rows[0]["note"] == "Filament"
    assert rows[0]["status"] == "out for delivery"
    assert rows[0]["expected date"] == "2026-11-18"


def test_a_status_change_is_announced(ledger, monkeypatch) -> None:
    store, _opened = ledger
    monkeypatch.setattr(orders_tools, "fetch", lambda url: page(url.rsplit("/", 1)[-1], "in transit"))
    orders_tools.track_order({"text": "UPS 1Z999AA10123456784"})
    monkeypatch.setattr(orders_tools, "fetch", lambda url: page(url.rsplit("/", 1)[-1], "out for delivery"))

    result = orders_tools.track_order({"text": "UPS 1Z999AA10123456784"})

    assert "Tracking 1Z999AA10123456784 changed from in transit to out for delivery." in result.speech


def test_an_unreadable_page_is_saved_and_spoken_as_such(ledger, monkeypatch) -> None:
    store, _opened = ledger
    monkeypatch.setattr(orders_tools, "fetch", lambda url: "<html><body>Please wait</body></html>")

    result = orders_tools.track_order({"text": "DHL 5001234567"})

    assert result.ok
    assert result.speech == (
        "Tracking 5001234567 for DHL: I opened the page, but it states no status I can read. "
        "Saved 1 order on your order list."
    )
    assert "5001234567" in store["content"]


def test_an_offline_carrier_page_still_saves_the_order(ledger, monkeypatch) -> None:
    store, _opened = ledger

    def offline(url: str) -> str:
        raise OSError("no network in this test")

    monkeypatch.setattr(orders_tools, "fetch", offline)

    result = orders_tools.track_order({"text": "FedEx 321987654321"})

    assert result.ok
    assert "I could not reach the carrier page, so I have no status for it." in result.speech
    assert "321987654321" in store["content"]


def test_text_without_a_tracking_number_fails(ledger) -> None:
    store, opened = ledger

    result = orders_tools.track_order({"text": "no numbers in this text at all"})

    assert not result.ok
    assert result.speech == "I couldn't find a tracking number in that text."
    assert store["content"] == ""
    assert opened == []


def test_the_questions_read_only_the_ledger(ledger, monkeypatch) -> None:
    store, _opened = ledger
    monkeypatch.setattr(orders_tools, "fetch", lambda url: pytest.fail("the questions must not fetch"))
    # Dates relative to today: a hard-coded date goes past and the "this month"
    # question starts answering "nothing is due" (time bomb, strata/f160).
    due = date.today() + timedelta(days=5)
    last_seen = due - timedelta(days=2)
    store["content"] = ledger_module.render(
        [
            {
                "number": "1Z999AA10123456784",
                "carrier": "UPS",
                "note": "Filament",
                "status": "out for delivery",
                "expected date": due.isoformat(),
                "last seen": last_seen.isoformat(),
            }
        ]
    )
    spoken_due = f"{_MONTHS[due.month - 1]} {due.day}"

    assert orders_tools.order_status({"note": "filament"}).speech == (
        f"Your filament order, tracking 1Z999AA10123456784 with UPS is out for delivery, due {spoken_due}."
    )
    assert orders_tools.orders_arriving({"when": "this month"}).speech.startswith("One order is due this month:")
    assert orders_tools.order_status({"note": "cat"}).speech == "I have no cat order on your order list."
    assert orders_tools.order_status({}).speech == (
        f"Your filament order, tracking 1Z999AA10123456784 with UPS is out for delivery, due {spoken_due}."
    )


def test_an_empty_ledger_says_so(ledger) -> None:
    assert orders_tools.order_status({"note": "filament"}).speech == "Nothing is on your order list."
    assert orders_tools.orders_arriving({"when": "this week"}).speech == "Nothing is on your order list."


def test_a_missing_documents_folder_is_spoken(monkeypatch) -> None:
    monkeypatch.setattr(orders_tools.filemanage, "_folder", lambda spoken: None)

    assert not orders_tools.order_status({}).ok
    assert orders_tools.order_status({}).speech == "I could not reach your orders."
    assert not orders_tools.track_order({"text": "UPS 1Z999AA10123456784"}).ok


def test_phrases_route_to_the_order_tools() -> None:
    assert route("track this package 1Z999AA10123456784") == ToolCall(
        "track_order", {"text": "1Z999AA10123456784"}
    )
    assert route("where is my filament order") == ToolCall("order_status", {"note": "my filament"})
    assert route("what's arriving this week") == ToolCall("orders_arriving", {"when": "this week"})
    assert route("where is my order") == ToolCall("order_status", {"note": "my"})
    assert route("track my package 1Z999AA10123456784") == ToolCall(
        "track_package", {"tracking": "1Z999AA10123456784"}
    )


def test_registry_has_the_order_tools_with_the_right_risk() -> None:
    registry = default_registry()
    names = set(registry.names())
    assert {"track_order", "order_status", "orders_arriving"} <= names
    assert registry.get("track_order").risk == RiskLevel.REVERSIBLE
    assert registry.get("order_status").risk == RiskLevel.READ_ONLY
    assert registry.get("orders_arriving").risk == RiskLevel.READ_ONLY
    registry.validate("track_order", {"text": "1Z999AA10123456784"})
