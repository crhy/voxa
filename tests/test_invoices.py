"""Invoices from one spoken sentence: the parse table, the document, the read-back."""

from __future__ import annotations

from datetime import date

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.invoices import (
    Invoice,
    body,
    last_invoice,
    next_reference,
    parse_invoice_line,
    remember,
    spoken_due,
    spoken_invoice,
    title,
)

TODAY = date(2026, 10, 10)

SENTENCE = "make me an invoice for two office chairs to Acme, net 40 dollars freight 6 taxes 2, terms 15 days"


@pytest.fixture(autouse=True)
def _invoice_state(monkeypatch, tmp_path):
    """The running invoice number lives in Voxa's state; keep each test's counter its own."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def the_invoice(sentence: str) -> Invoice:
    invoice = parse_invoice_line(sentence, TODAY)
    assert invoice is not None
    return invoice


def the_total(sentence: str) -> float:
    return the_invoice(sentence).total


def test_one_sentence_makes_one_invoice() -> None:
    invoice = parse_invoice_line(SENTENCE, TODAY)
    assert invoice is not None
    assert invoice.client == "Acme"
    assert invoice.quantity == 2
    assert invoice.description == "office chairs"
    assert (invoice.net, invoice.freight, invoice.taxes) == (40.0, 6.0, 2.0)
    assert invoice.total == 48.0
    assert invoice.terms == [(TODAY, date(2026, 10, 25))]


def test_total_is_net_plus_freight_plus_taxes() -> None:
    assert the_total("make an invoice for chairs to Acme, net 40 dollars freight 6 taxes 2") == 48.0
    assert the_total("make an invoice for chairs to Acme, net 40 dollars") == 40.0
    assert the_total("make an invoice for chairs to Acme, freight 6 dollars") == 6.0
    assert the_total("make an invoice for chairs to Acme, net 10 dollars and 50 cents") == 10.5


def test_a_bare_amount_is_the_net() -> None:
    invoice = the_invoice("make an invoice for a desk to Acme for 40 dollars")
    assert (invoice.net, invoice.freight, invoice.taxes) == (40.0, 0.0, 0.0)
    assert invoice.description == "a desk"
    assert the_invoice("make an invoice for Acme for $1,200").net == 1200.0
    assert the_invoice("make an invoice for two chairs to Acme, 1,200 dollars").net == 1200.0


def test_a_sentence_without_a_client_is_not_an_invoice() -> None:
    assert parse_invoice_line("make an invoice for two chairs", TODAY) is None
    assert parse_invoice_line("make an invoice", TODAY) is None
    assert parse_invoice_line("what is the capital of France", TODAY) is None


def test_terms_counted_from_today() -> None:
    assert the_invoice("make an invoice for chairs to Acme, terms 15 days").terms == [(TODAY, date(2026, 10, 25))]
    assert the_invoice("make an invoice for chairs to Acme, terms 2 weeks").terms == [(TODAY, date(2026, 10, 24))]
    assert the_invoice("make an invoice for chairs to Acme, terms 3 months").terms == [(TODAY, date(2027, 1, 10))]


def test_terms_with_an_explicit_date_are_honoured() -> None:
    assert the_invoice("make an invoice for chairs to Acme, terms March 3rd 2027").terms == [(TODAY, date(2027, 3, 3))]
    assert the_invoice("make an invoice for chairs to Acme, due Friday").terms == [(TODAY, date(2026, 10, 16))]
    assert the_invoice("make an invoice for chairs to Acme, terms from tomorrow to Friday").terms == [
        (date(2026, 10, 11), date(2026, 10, 16))
    ]


def test_no_terms_is_still_an_invoice() -> None:
    invoice = the_invoice("invoice for Acme net 40 dollars freight 6 taxes 2")
    assert invoice.client == "Acme"
    assert invoice.terms == []
    assert invoice.total == 48.0


def test_the_document_lines() -> None:
    text = body(the_invoice(SENTENCE), TODAY, 7)
    assert title(the_invoice(SENTENCE), 7) == "Invoice 7 for Acme"
    assert "Reference: VOXA-0007" in text
    assert "Client: Acme" in text
    assert "Delivery: 2 office chairs" in text
    assert "Terms: from 2026-10-10 to 2026-10-25" in text
    assert "Net: 40.00" in text
    assert "Freight: 6.00" in text
    assert "Taxes: 2.00" in text
    assert "Total: 48.00" in text


def test_the_read_back_line() -> None:
    invoice = the_invoice(SENTENCE)
    assert spoken_invoice(invoice, TODAY, 7) == (
        "Invoice 7 for Acme: 2 office chairs, net 40 dollars, freight 6 dollars and taxes 2 dollars, "
        "48 dollars total, due on October 25th."
    )
    assert spoken_due([], TODAY) == ""


def test_the_reference_number_runs() -> None:
    assert next_reference() == 1
    assert next_reference() == 2


def test_the_last_invoice_is_remembered() -> None:
    invoice = the_invoice(SENTENCE)
    assert last_invoice(TODAY) is None
    remember(invoice, 3, TODAY)
    found = last_invoice(TODAY)
    assert found is not None
    assert found[0] == invoice
    assert found[1] == 3


def test_invoice_phrases_route() -> None:
    assert route(SENTENCE) == ToolCall("make_invoice", {"line": SENTENCE})
    assert route("invoice for Acme net 40 dollars freight 6 taxes 2").tool == "make_invoice"
    assert route("read me the invoice back") == ToolCall("invoice_read", {})
    assert route("read the invoice back") == ToolCall("invoice_read", {})


def test_invoice_rules_do_not_steal_other_commands() -> None:
    assert route("how many invoices do I have") == ToolCall("finance_count", {"kind": "invoices"})
    assert route("where are my invoices") == ToolCall("finance_count", {"kind": "invoices"})
    assert route("organize my financial documents") == ToolCall("finance_sort", {})
    assert route("open the invoice") == ToolCall("open_app", {"name": "invoice"})
    assert route("i spent 12 dollars on lunch") == ToolCall("expense_add", {"amount": "12 dollars", "what": "lunch"})
