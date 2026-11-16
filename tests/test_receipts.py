"""Receipts into the spending ledger: parsing, routing, the handler, and one real OCR run.

The last test draws a receipt with PIL and reads it with the machine's own text reader,
so the offline path is proven end to end; it skips on a machine without tesseract.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from unittest import skipUnless

from voxa.agent.expenses import category_for, entries
from voxa.agent.intents import ToolCall, route
from voxa.agent.receipts import parse_receipt, receipt_date, receipt_what, spoken_receipt
from voxa.agent.tools import default_registry
from voxa.agent.tools.receipts import NO_READER, UNREADABLE, _ocr_available, _receipt_log_handler

TODAY = date(2026, 11, 14)

DELI = """ACCEPCO 11/14/26
LUNCH SANDWICH 1
CAFE 01
TOTAL $42.00
THANK YOU COME BACK TOMORROW
"""

FUEL = """SHELL STATION 911
FUEL 3.25
TOTAL $10.40
THANK YOU
"""

NO_TOTAL = """ACCEPCO
LUNCH SANDWICH
CAFE
"""

SPLIT_TOTALS = """CORNER MARKET 11/14/26
BAGUETTE SANDWICH 1
SODA 1
SUBTOTAL 30.00
TOTAL 42.00
THANK YOU
"""


def test_parse_receipt_reads_the_labelled_total() -> None:
    assert parse_receipt(DELI) == (42.0, "accepco", ["lunch sandwich", "cafe"])


def test_parse_receipt_prefers_the_total_line_over_the_items() -> None:
    assert parse_receipt(SPLIT_TOTALS)[0] == 42.0
    assert parse_receipt("MART 11/14/26\nTOYS 60.00\nTOTAL 42.00\n")[0] == 42.0


def test_parse_receipt_falls_back_to_the_largest_amount() -> None:
    assert parse_receipt("MART 11/14/26\nMILK 4.50\nBREAD 12.00\n")[0] == 12.0


def test_parse_receipt_refuses_to_guess_without_an_amount() -> None:
    assert parse_receipt(NO_TOTAL)[0] is None
    assert parse_receipt("")[0] is None


def test_receipt_what_gives_the_category() -> None:
    assert category_for(receipt_what(DELI)) == "food"
    assert category_for(receipt_what(FUEL)) == "transport"
    assert receipt_what(NO_TOTAL) == "accepco lunch sandwich cafe"


def test_receipt_date_reads_the_print_and_otherwise_uses_today() -> None:
    assert receipt_date(DELI, TODAY) == date(2026, 11, 14)
    assert receipt_date("MART 2026-11-03\nTOTAL $3.00\n", TODAY) == date(2026, 11, 3)
    assert receipt_date("MART November 3, 2026\nTOTAL $3.00\n", TODAY) == date(2026, 11, 3)
    assert receipt_date(NO_TOTAL, TODAY) == TODAY


def test_spoken_receipt_says_amount_category_and_day() -> None:
    assert spoken_receipt(42.0, date(2026, 11, 14), "food") == (
        "That receipt is 42 dollars, filed under food for 2026-11-14."
    )


def test_phrases_route() -> None:
    assert route("log this receipt") == ToolCall("receipt_log", {"path": ""})
    assert route("add the receipt in my downloads to my expenses") == ToolCall("receipt_log", {"path": "my downloads"})
    assert route("what did I spend according to this receipt") == ToolCall("receipt_log", {"path": ""})
    assert route("log the receipt scan.jpg") == ToolCall("receipt_log", {"path": "scan.jpg"})


def test_guard_phrases_still_route() -> None:
    assert route("log 42 dollars for lunch") == ToolCall("expense_add", {"amount": "42 dollars", "what": "lunch"})
    assert route("what did i spend this week") == ToolCall("expense_total", {"period": "this week", "category": ""})
    assert route("what is the capital of France") is None


def test_registry_has_the_receipt_tool() -> None:
    assert "receipt_log" in default_registry().names()


def _ledger_store(monkeypatch) -> dict[str, str]:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.receipts._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.receipts._write",
        lambda content: store.__setitem__("content", content),
    )
    return store


def test_handler_end_to_end_logs_one_ledger_line(monkeypatch, tmp_path) -> None:
    receipt = tmp_path / "receipt.png"
    receipt.write_bytes(b"not really an image, only its name matters here")
    store = _ledger_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr_available", lambda: True)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr", lambda path: DELI)

    result = _receipt_log_handler({"path": str(receipt)})
    assert result.ok
    assert result.speech == "That receipt is 42 dollars, filed under food for 2026-11-14. From receipt.png."
    assert entries(store["content"]) == [(date(2026, 11, 14), 42.0, "food", "Accepco lunch sandwich cafe")]


def test_handler_refuses_a_receipt_with_no_total(monkeypatch, tmp_path) -> None:
    receipt = tmp_path / "blurry.png"
    receipt.write_bytes(b"blank")
    store = _ledger_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr_available", lambda: True)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr", lambda path: NO_TOTAL)

    result = _receipt_log_handler({"path": str(receipt)})
    assert not result.ok and result.speech == UNREADABLE
    assert store["content"] == ""


def test_handler_refuses_a_file_that_is_not_a_picture(monkeypatch) -> None:
    store = _ledger_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr", lambda path: DELI)
    result = _receipt_log_handler({"path": "/home/rhy/Documents/notes.txt"})
    assert not result.ok and result.speech.startswith("That is not a picture")
    assert store["content"] == ""


def test_handler_refuses_a_missing_file(monkeypatch) -> None:
    store = _ledger_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.receipts._ocr", lambda path: DELI)
    result = _receipt_log_handler({"path": "/nonexistent-voxa-test/receipt.png"})
    assert not result.ok and result.speech.startswith("I could not find that receipt")
    assert store["content"] == ""


def test_handler_says_no_text_reader_is_installed_and_writes_nothing(monkeypatch, tmp_path) -> None:
    receipt = tmp_path / "receipt.png"
    receipt.write_bytes(b"blank")
    store = _ledger_store(monkeypatch)

    def _probe_fails(command, stdin=None, check=True):
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("voxa.agent.tools.receipts._run", _probe_fails)
    assert not _ocr_available()
    result = _receipt_log_handler({"path": str(receipt)})
    assert not result.ok and result.speech == NO_READER
    assert store["content"] == ""


@skipUnless(shutil.which("tesseract"), "no local text reader on this machine")
def test_real_text_recognition_reads_a_drawn_receipt(tmp_path) -> None:
    import pytest

    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (420, 170), "white")
    draw = ImageDraw.Draw(image)
    draw.text((14, 14), "ACCEPCO 11/14/26", fill="black", font_size=24)
    draw.text((14, 56), "LUNCH SANDWICH 1", fill="black", font_size=24)
    draw.text((14, 98), "TOTAL $42.00", fill="black", font_size=24)
    path = tmp_path / "receipt.png"
    image.save(path)

    done = subprocess.run([shutil.which("tesseract"), str(path), "-"], capture_output=True, text=True, timeout=60)
    text = done.stdout
    assert parse_receipt(text)[0] == 42.0
    assert receipt_date(text, TODAY) == date(2026, 11, 14)
    assert category_for(receipt_what(text)) == "food"
