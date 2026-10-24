"""Pure regex helpers pull dates and money amounts out of document text."""

from __future__ import annotations

from voxa.agent.extract import find_amounts, find_dates, spoken


def test_find_dates_slash_iso_and_words() -> None:
    text = "Invoice dated 01/15/2024, due 2024-02-01, signed January 5, 2024 and 3rd March 2024."
    dates = find_dates(text)
    assert "01/15/2024" in dates
    assert "2024-02-01" in dates
    assert "January 5, 2024" in dates
    assert "3rd March 2024" in dates


def test_find_dates_deduplicates() -> None:
    assert find_dates("2024-01-01 and 2024-01-01 again") == ["2024-01-01"]


def test_find_amounts_symbols_commas_and_codes() -> None:
    text = "Paid $1,234.56 and 2,500 EUR, plus 99.99 USD and a 5000 total."
    amounts = find_amounts(text)
    assert "$1,234.56" in amounts
    assert "2,500 EUR" in amounts
    assert "99.99 USD" in amounts


def test_find_amounts_deduplicates() -> None:
    assert find_amounts("$50 and $50") == ["$50"]


def test_spoken_lists_groups() -> None:
    line = spoken("invoice.pdf", ["01/15/2024"], ["$1,234.56"], ["Acme"])
    assert line == "In invoice.pdf: dates 01/15/2024; amounts $1,234.56; names Acme."


def test_spoken_reports_none_found_for_empty_groups() -> None:
    line = spoken("lease.txt", [], [], [])
    assert line == "In lease.txt: dates none found; amounts none found; names none found."
