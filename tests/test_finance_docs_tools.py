"""Voice commands for organising financial documents route and work end to end."""

from __future__ import annotations

import subprocess

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools import finance_docs as fd
from voxa.agent.tools.finance_docs import _count_handler, _sort_handler, _undo_handler


def test_finance_phrases_route() -> None:
    assert route("organize my financial documents") == ToolCall("finance_sort", {})
    assert route("organise my financial documents") == ToolCall("finance_sort", {})
    assert route("sort my finance papers") == ToolCall("finance_sort", {})
    assert route("undo the filing") == ToolCall("finance_undo", {})
    assert route("undo the sorting") == ToolCall("finance_undo", {})
    assert route("how many invoices do I have") == ToolCall("finance_count", {"kind": "invoices"})
    assert route("how many receipts are there") == ToolCall("finance_count", {"kind": "receipts"})
    assert route("where are my receipts") == ToolCall("finance_count", {"kind": "receipts"})
    assert route("where is my warranty") == ToolCall("finance_count", {"kind": "warranty"})


def test_finance_rules_do_not_steal_other_commands() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("undo that").tool == "undo_file_action"
    assert route("add milk to my shopping list") == ToolCall("list_add", {"item": "milk", "list": "shopping"})


def test_kind_of_classifies_filenames() -> None:
    assert fd.kind_of("invoice-acme-2026.txt") == "invoice"
    assert fd.kind_of("receipt-hardware-store.txt") == "receipt"
    assert fd.kind_of("warranty-laptop.pdf") == "warranty"
    assert fd.kind_of("tax-return-2025.pdf") == "tax"
    assert fd.kind_of("w2-claim-form.pdf") == "tax"
    assert fd.kind_of("1099-tax.pdf") == "tax"
    assert fd.kind_of("bank-statement-march.pdf") == "statement"
    assert fd.kind_of("holiday-photo.jpg") is None


def test_sort_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda s: f"/home/test/{s.title()}")
    monkeypatch.setattr(fd, "_find_files", lambda folder: [
        f"{folder}/invoice-acme-2026.txt",
        f"{folder}/receipt-hardware-store.txt",
    ] if "Downloads" in folder else [])
    monkeypatch.setattr(fd, "_run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    written: list = []
    monkeypatch.setattr(fd, "_write_state", lambda moves: written.append(moves))

    result = _sort_handler({})
    assert result.ok
    assert result.speech == "I filed 2 documents: 1 invoice and 1 receipt."
    assert len(written[0]) == 2


def test_undo_handler_end_to_end(monkeypatch) -> None:
    moves = [
        {"from": "/home/test/Downloads/invoice-acme-2026.txt", "to": "/home/test/Documents/Finance/Invoices/invoice-acme-2026.txt"},
        {"from": "/home/test/Downloads/receipt-hardware-store.txt", "to": "/home/test/Documents/Finance/Receipts/receipt-hardware-store.txt"},
    ]
    monkeypatch.setattr(fd, "_read_state", lambda: moves)
    monkeypatch.setattr(fd, "_run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    written: list = []
    monkeypatch.setattr(fd, "_write_state", lambda moves: written.append(moves))

    result = _undo_handler({})
    assert result.ok
    assert result.speech == "I moved 2 documents back."
    assert written[0] == []


def test_undo_handler_nothing_to_undo(monkeypatch) -> None:
    monkeypatch.setattr(fd, "_read_state", lambda: [])
    result = _undo_handler({})
    assert result.ok
    assert result.speech == "Nothing to undo."


def test_count_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda s: "/home/test/Documents")
    monkeypatch.setattr(fd, "_find_files", lambda folder: [
        f"{folder}/invoice-a.txt",
        f"{folder}/invoice-b.txt",
        f"{folder}/invoice-c.txt",
    ])
    result = _count_handler({"kind": "invoice"})
    assert result.ok
    assert result.speech == "You have 3 invoices."


def test_count_handler_zero(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.filemanage._folder", lambda s: "/home/test/Documents")
    monkeypatch.setattr(fd, "_find_files", lambda folder: [])
    result = _count_handler({"kind": "receipt"})
    assert result.ok
    assert result.speech == "You have no receipts."


def test_registry_has_finance_tools() -> None:
    names = default_registry().names()
    for name in ("finance_sort", "finance_undo", "finance_count"):
        assert name in names
