from __future__ import annotations

from datetime import date

from voxa.agent import orders

TODAY = date(2026, 11, 16)

EMAIL = """Thanks for your purchase.
Your order for filament has been shipped.
Tracking number: 1Z999AA10123456784
A second parcel went out under FedEx 321987654321.
Expected delivery: November 18, 2026
Call 850-123-4567 or open https://shop.example.com/orders/123456789012
Order date: 2026-11-12 at 14:30
Tracking number: 1Z999AA10123456784
"""


def test_pasted_confirmation_email_yields_exactly_two_numbers() -> None:
    assert orders.parse_tracking_numbers(EMAIL) == [
        ("UPS", "1Z999AA10123456784"),
        ("FedEx", "321987654321"),
    ]


def test_carrier_shapes_are_named() -> None:
    assert orders.parse_tracking_numbers("USPS 9205123456789ABC") == [("USPS", "9205123456789ABC")]
    assert orders.parse_tracking_numbers("DHL 5001234567") == [("DHL", "5001234567")]
    assert orders.parse_tracking_numbers("FedEx 321987654321") == [("FedEx", "321987654321")]
    assert orders.parse_tracking_numbers("UPS 1Z999AA10123456784") == [("UPS", "1Z999AA10123456784")]


def test_a_bare_number_falls_back_to_the_shared_parser() -> None:
    assert orders.parse_tracking_numbers("my number is 4711859200134") == [("unknown", "4711859200134")]


def test_dates_phone_numbers_and_links_are_skipped() -> None:
    text = "On 2026-11-18 at 14:30 call 850-123-4567 or see https://ex.com/orders/123456789012"
    assert orders.parse_tracking_numbers(text) == []


def test_numbers_are_not_reported_twice() -> None:
    assert orders.parse_tracking_numbers("1Z999AA10123456784 and again 1Z999AA10123456784") == [
        ("UPS", "1Z999AA10123456784")
    ]


def test_the_note_and_the_expected_date_come_from_the_pasted_text() -> None:
    assert orders.note_from_text(EMAIL) == "Filament"
    assert orders.expected_from_text(EMAIL) == "2026-11-18"
    assert orders.note_from_text("nothing about an item here") == ""
    assert orders.expected_from_text("nothing about an item here") == ""


def test_the_ledger_round_trips() -> None:
    rows, changes = orders.add_or_update([], "1Z999AA10123456784", "UPS", "Filament", "in transit", "2026-11-18", "2026-11-12")
    assert changes == []
    rows, _ = orders.add_or_update(rows, "321987654321", "FedEx", "Wrench", "out for delivery", "2026-11-17", "2026-11-12")
    content = orders.render(rows)
    assert orders.parse(content) == rows
    assert "| 1Z999AA10123456784 | UPS | Filament | in transit | 2026-11-18 | 2026-11-12 |" in content


def test_a_status_change_produces_one_sentence_naming_old_and_new() -> None:
    rows, _ = orders.add_or_update([], "1Z999AA10123456784", "UPS", "Filament", "in transit", "2026-11-18", "2026-11-12")
    rows, changes = orders.add_or_update(rows, "1Z999AA10123456784", "UPS", "Filament", "out for delivery", "", "2026-11-16")
    assert changes == ["Tracking 1Z999AA10123456784 changed from in transit to out for delivery."]
    assert rows[0]["status"] == "out for delivery"
    assert rows[0]["expected date"] == "2026-11-18"
    assert rows[0]["last seen"] == "2026-11-16"


def test_an_unreadable_page_never_invents_a_status() -> None:
    assert orders.status_from_page("", "1Z999AA10123456784") == ""
    assert orders.status_from_page("<html><body>Please wait</body></html>", "1Z999AA10123456784") == ""
    page = "<html><body><script>var legend='delivered';</script>Tracking 1Z999AA10123456784 is in transit</body></html>"
    assert orders.status_from_page(page, "1Z999AA10123456784") == "in transit"


def test_the_list_is_read_back_by_note() -> None:
    rows, _ = orders.add_or_update([], "1Z999AA10123456784", "UPS", "Filament", "out for delivery", "2026-11-18", "2026-11-16")
    assert orders.spoken_for_note(rows, "filament") == (
        "Your filament order, tracking 1Z999AA10123456784 with UPS is out for delivery, due November 18."
    )
    assert orders.spoken_for_note(rows, "cat") == "I have no cat order on your order list."


def test_arriving_this_week_names_the_due_orders() -> None:
    rows, _ = orders.add_or_update([], "1Z999AA10123456784", "UPS", "Filament", "in transit", "2026-11-18", "2026-11-16")
    rows, _ = orders.add_or_update(rows, "321987654321", "FedEx", "Wrench", "out for delivery", "2026-11-17", "2026-11-16")
    rows, _ = orders.add_or_update(rows, "5001234567", "DHL", "Cable", "in transit", "2026-12-24", "2026-11-16")
    spoken = orders.spoken_arriving(rows, "this week", TODAY)
    assert spoken.startswith("2 orders are due this week:")
    assert "321987654321" in spoken and "1Z999AA10123456784" in spoken
    assert "5001234567" not in spoken
    assert orders.spoken_arriving(rows, "tomorrow", TODAY).startswith("One order is due tomorrow:")


def test_an_empty_list_says_so_in_every_wording() -> None:
    assert orders.spoken_for_note([], "filament") == "Nothing is on your order list."
    assert orders.spoken_arriving([], "this week", TODAY) == "Nothing is on your order list."
    assert orders.spoken_changes([]) == "Nothing is on your order list."


def test_no_expected_date_is_spoken_as_no_expected_date() -> None:
    rows, _ = orders.add_or_update([], "5001234567", "DHL")
    assert orders.spoken_arriving(rows, "this week", TODAY) == (
        "Nothing is due this week, and your order has no expected date."
    )
    assert orders.spoken_changes(rows) == (
        "The order, tracking 5001234567 with DHL has no status read from its carrier page yet."
    )
