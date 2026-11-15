"""A bank statement the user already has: read it, categorise it, compare months.

Pure module, standard library only. The sample statements below are the two shapes
Voxa has to handle: a CSV body with a header, and plain text as pdftotext leaves it.
"""

from __future__ import annotations

from datetime import date

from voxa.agent.expenses import add, category_for, entries
from voxa.agent.statement import (
    candidate_lines,
    categorise,
    compare_months,
    largest_items,
    load_corrections,
    parse_statement,
    recurring_merchants,
    remembered_correction,
    report_body,
    spoken_compare,
    spoken_largest,
    spoken_recurring,
    spoken_unusual,
    totals_by_category,
    transactions,
    unusual_charges,
)

CSV_STATEMENT = """Online account,Acme Savings Bank
Card number,4111 1111 1111 1111
Date,Amount,Description,Category
03/02/2026,45.99,Kroger Store,Food
03/03/2026,12.00,Shell Gas Station,Gas
03/05/2026,120.00,Verizon Wireless,Utilities
03/06/2026,9.50,Walgreens Pharmacy,Health
03/07/2026,250.00,Delta Air Lines,Travel
03/08/2026,45.99,Kroger Store,Food
02/15/2026,40.00,Kroger Store,Food
TOTALS,523.48,Total for the month,
Available balance,12345.67,,
"""

TEXT_STATEMENT = """Acme Savings Bank statement, March 2026
Card number 4111 1111 1111 1111
03/02/2026   $45.99   KROGER STORE
03/03/2026   $12.00   SHELL GAS STATION
05 March 2026   $120.00   VERIZON WIRELESS
03/08/2026   $45.99   KROGER STORE
TOTALS   $523.48   TOTAL FOR THE MONTH
Available balance   $12,345.67
"""


def test_parse_csv_statement() -> None:
    assert parse_statement(CSV_STATEMENT) == [
        (date(2026, 3, 2), 45.99, "Kroger Store"),
        (date(2026, 3, 3), 12.0, "Shell Gas Station"),
        (date(2026, 3, 5), 120.0, "Verizon Wireless"),
        (date(2026, 3, 6), 9.5, "Walgreens Pharmacy"),
        (date(2026, 3, 7), 250.0, "Delta Air Lines"),
        (date(2026, 3, 8), 45.99, "Kroger Store"),
        (date(2026, 2, 15), 40.0, "Kroger Store"),
    ]


def test_summary_rows_are_never_counted_twice() -> None:
    rows = parse_statement(CSV_STATEMENT)
    assert round(sum(amount for _day, amount, _what in rows), 2) == 523.48
    assert not any("total" in what.lower() or "balance" in what.lower() for _day, _amount, what in rows)
    assert candidate_lines(CSV_STATEMENT) == len(rows)


def test_parse_text_statement() -> None:
    assert parse_statement(TEXT_STATEMENT) == [
        (date(2026, 3, 2), 45.99, "KROGER STORE"),
        (date(2026, 3, 3), 12.0, "SHELL GAS STATION"),
        (date(2026, 3, 5), 120.0, "VERIZON WIRELESS"),
        (date(2026, 3, 8), 45.99, "KROGER STORE"),
    ]


def test_other_header_spellings() -> None:
    body = "Transaction Date,Withdrawal (+)/Deposit (-),Details\n03/04/2026,-32.50,Coffee Bar\n"
    assert parse_statement(body) == [(date(2026, 3, 4), 32.5, "Coffee Bar")]


def test_categorise_known_merchants() -> None:
    assert categorise("Kroger Store") == "food"
    assert categorise("Shell Gas Station") == "transport"
    assert categorise("Verizon Wireless") == "home"
    assert categorise("Walgreens Pharmacy") == "health"
    assert categorise("Delta Air Lines") == "transport"
    assert categorise("Whole Foods Market") == "food"


def test_categorise_falls_back_to_the_spoken_words() -> None:
    assert categorise("lunch") == "food"
    assert categorise("movie night") == "fun"
    assert categorise("Mystery Mart") == "other"


def test_category_breakdown_of_a_sample_statement() -> None:
    assert totals_by_category(transactions(CSV_STATEMENT)) == {
        "transport": 262.0,
        "food": 131.98,
        "home": 120.0,
        "health": 9.5,
    }


def test_largest_items() -> None:
    biggest = largest_items(transactions(CSV_STATEMENT), 2)
    assert [(amount, what) for _day, amount, _cat, what in biggest] == [(250.0, "Delta Air Lines"), (120.0, "Verizon Wireless")]


def test_recurring_merchants() -> None:
    assert recurring_merchants(transactions(CSV_STATEMENT)) == [("kroger store", 3)]


def test_unusual_charges() -> None:
    charges = unusual_charges(transactions(CSV_STATEMENT))
    assert [(amount, what) for _day, amount, _cat, what in charges] == [(250.0, "Delta Air Lines")]


def test_compare_months() -> None:
    result = compare_months(transactions(CSV_STATEMENT), "last month", date(2026, 4, 15))
    assert result["label"] == "March 2026"
    assert result["previous_label"] == "February 2026"
    assert result["categories"]["food"] == (91.98, 40.0, 51.98)
    assert result["categories"]["transport"] == (262.0, 0.0, 262.0)
    assert result["total"] == (483.48, 40.0, 443.48)


def test_a_remembered_correction_applies_on_the_next_import(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_STATEMENT_CORRECTIONS", str(tmp_path / "corrections.json"))
    assert categorise("Mystery Mart") == "other"
    assert remembered_correction("Mystery Mart", "food") == "food"
    assert categorise("Mystery Mart") == "food"
    assert transactions("03/09/2026,7.25,Mystery Mart") == [(date(2026, 3, 9), 7.25, "food", "Mystery Mart")]
    assert load_corrections() == {"mystery mart": "food"}


def test_imported_rows_keep_the_statement_category(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_STATEMENT_CORRECTIONS", str(tmp_path / "corrections.json"))
    day, amount, category, what = transactions(CSV_STATEMENT)[0]
    assert category_for(what) == "other"
    content = add("", day, amount, what, "march-2026-statement", category)
    assert content == "- 2026-03-02 | 45.99 | food | Kroger Store | trip:march-2026-statement\n"
    assert entries(content) == [(date(2026, 3, 2), 45.99, "food", "Kroger Store")]


def test_spoken_reports_name_the_categories() -> None:
    items = transactions(CSV_STATEMENT)
    result = compare_months(items, "last month", date(2026, 4, 15))
    assert spoken_compare(result) == (
        "In March 2026 you spent 483 dollars and 48 cents, against 40 dollars in February 2026. "
        "The change: transport is up 262 dollars, home is up 120 dollars, food is up 51 dollars and 98 cents "
        "and health is up 9 dollars and 50 cents."
    )
    assert spoken_recurring(items) == "You pay most often at kroger store 3 times."
    assert spoken_unusual(items) == "Unusually large: 250 dollars at delta air lines on 2026-03-07."
    assert spoken_largest(items).startswith("The biggest charges: 250 dollars at delta air lines")


def test_report_body_names_every_section() -> None:
    items = transactions(CSV_STATEMENT)
    body = report_body(items, compare_months(items, "last month", date(2026, 4, 15)))
    for section in ("Totals by category", "Largest charges", "Recurring merchants", "Unusually large charges", "March 2026 against February 2026"):
        assert section in body
    assert "food: 131.98" in body
    assert "kroger store  3 times" in body
