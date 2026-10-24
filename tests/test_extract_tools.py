"""The doc_extract tool routes from speech and reads a file end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools import extract as extract_tool


def test_extract_phrases_route() -> None:
    assert route("what are the dates and amounts in invoice.pdf") == ToolCall(
        "doc_extract", {"name": "invoice.pdf"}
    )
    assert route("read the important details from my lease") == ToolCall(
        "doc_extract", {"name": "my lease"}
    )
    assert route("what does contract.txt say about money") == ToolCall(
        "doc_extract", {"name": "contract.txt"}
    )


def test_extract_rules_do_not_steal_lists_time_or_facts() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_handler_reads_and_reports(monkeypatch) -> None:
    monkeypatch.setattr(extract_tool, "_locate", lambda name: "/home/u/Documents/invoice-sample.txt")
    monkeypatch.setattr(
        extract_tool,
        "_read_text",
        lambda path: "Acme Corp invoice 01/15/2024 total $1,234.56 due 2024-02-01",
    )
    monkeypatch.setattr(extract_tool, "ask_model", lambda messages: "1. Acme Corp\n2. Jane Doe")
    result = extract_tool._doc_extract_handler({"name": "invoice"})
    assert result.ok
    assert result.speech == (
        "In invoice-sample.txt: dates 01/15/2024, 2024-02-01; "
        "amounts $1,234.56; names Acme Corp, Jane Doe."
    )


def test_handler_without_a_model_still_reports_dates_and_amounts(monkeypatch) -> None:
    monkeypatch.setattr(extract_tool, "_locate", lambda name: "/home/u/Documents/lease.txt")
    monkeypatch.setattr(extract_tool, "_read_text", lambda path: "Bond dated 03/03/2024 for 2,000 EUR")
    monkeypatch.setattr(extract_tool, "ask_model", None)
    result = extract_tool._doc_extract_handler({"name": "lease"})
    assert result.ok
    assert result.speech == "In lease.txt: dates 03/03/2024; amounts 2,000 EUR; names none found."


def test_handler_not_found(monkeypatch) -> None:
    monkeypatch.setattr(extract_tool, "_locate", lambda name: None)
    result = extract_tool._doc_extract_handler({"name": "ghost.pdf"})
    assert not result.ok
    assert result.speech == "I couldn't find ghost.pdf."


def test_registry_has_doc_extract() -> None:
    assert "doc_extract" in default_registry().names()
