"""Making an invoice by voice: the document on disk, opened, and read back."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from voxa import documents
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.invoices import _invoice_read_handler, _make_invoice_handler

SENTENCE = "make me an invoice for two office chairs to Acme, net 40 dollars freight 6 taxes 2, terms 15 days"


@pytest.fixture(autouse=True)
def _simulated_desktop(monkeypatch, tmp_path: Path):
    """No real Documents folder, no real LibreOffice, no shared state file."""
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda spoken: str(tmp_path / "Documents"))


def drafts(tmp_path: Path) -> Path:
    return tmp_path / "Documents" / "Voxa Drafts"


def test_make_invoice_writes_a_document(monkeypatch, tmp_path: Path) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(documents, "open_in_libreoffice", lambda path: opened.append(Path(path)))

    result = _make_invoice_handler({"line": SENTENCE})
    assert result.ok
    files = sorted(drafts(tmp_path).glob("*.odt"))
    assert files == [drafts(tmp_path) / "invoice-1-for-acme.odt"]
    assert opened == files

    with zipfile.ZipFile(files[0]) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["mimetype", "META-INF/manifest.xml", "content.xml"]
        content = archive.read("content.xml").decode("utf-8")
    assert "Client: Acme" in content
    assert "Delivery: 2 office chairs" in content
    assert "Total: 48.00" in content
    assert "Reference: VOXA-0001" in content


def test_make_invoice_speaks_the_numbers_it_wrote(tmp_path: Path) -> None:
    result = _make_invoice_handler({"line": SENTENCE})
    assert "Invoice 1 for Acme" in result.speech
    assert "48 dollars total" in result.speech
    assert "I opened it in LibreOffice." in result.speech

    read_back = _invoice_read_handler({})
    assert read_back.ok
    assert read_back.speech.split(":")[0] == "Invoice 1 for Acme"
    assert "48 dollars total" in read_back.speech
    assert "2 office chairs" in read_back.speech


def test_a_second_invoice_gets_another_number(tmp_path: Path) -> None:
    first = _make_invoice_handler({"line": "make an invoice for a desk to Acme for 40 dollars"})
    second = _make_invoice_handler({"line": "make an invoice for a desk to Bogus for 40 dollars"})
    assert first.ok and second.ok
    assert "Invoice 1 for Acme" in first.speech
    assert "Invoice 2 for Bogus" in second.speech
    assert len(list(drafts(tmp_path).glob("*.odt"))) == 2


def test_simulated_actions_never_spawn_libreoffice(monkeypatch) -> None:
    class _NoSpawn:
        def __getattr__(self, name: str):
            raise AssertionError(f"would have run {name}")

    monkeypatch.setattr(documents, "subprocess", _NoSpawn())
    result = _make_invoice_handler({"line": SENTENCE})
    assert result.ok


def test_a_sentence_without_a_client_fails_plainly() -> None:
    result = _make_invoice_handler({"line": "make an invoice for two chairs"})
    assert not result.ok
    assert result.speech == "Who is the invoice for?"


def test_reading_back_before_any_invoice_says_so() -> None:
    result = _invoice_read_handler({})
    assert not result.ok
    assert result.speech == "I have not made an invoice yet."


def test_the_invoice_phrases_route_to_the_registered_tools() -> None:
    assert route(SENTENCE) == ToolCall("make_invoice", {"line": SENTENCE})
    assert route("read me the invoice back") == ToolCall("invoice_read", {})
    names = default_registry().names()
    assert "make_invoice" in names
    assert "invoice_read" in names
