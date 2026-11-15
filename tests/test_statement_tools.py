"""Importing a statement the user already has, comparing months, and the report.

Handlers run with VOXA_SIMULATE_ACTIONS=1 and an injected text reader, so no file
is read from or written to the developer's desktop and nothing is opened.
"""

from __future__ import annotations

import zipfile
from datetime import date, timedelta
from pathlib import Path

from voxa import documents
from voxa.agent.expenses import entries
from voxa.agent.intents import ToolCall, route
from voxa.agent.statement import month_label, month_start, remember_correction
from voxa.agent.tools import default_registry
from voxa.agent.tools.statement import (
    _import_statement_handler,
    _statement_compare_handler,
    _statement_report_handler,
)

TODAY = date.today()
THIS_LABEL = month_label(month_start(TODAY))
LAST_DAY = TODAY.replace(day=1) - timedelta(days=1)
LAST_LABEL = month_label(month_start(LAST_DAY))
THIS = TODAY.isoformat()
LAST = LAST_DAY.isoformat()

STATEMENT = (
    "Date,Amount,Description\n"
    f"{THIS},45.99,Kroger Store\n"
    f"{THIS},12.00,Shell Gas Station\n"
    f"{THIS},7.25,Mystery Mart\n"
    f"{LAST},40.00,Kroger Store\n"
    "03/09/2026   1,234.56\n"
    "TOTALS,1340.80,Total for the month,\n"
)


def _use_store(monkeypatch, content: str = "") -> dict[str, str]:
    store = {"content": content}
    monkeypatch.setattr("voxa.agent.tools.statement._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.statement._write",
        lambda content: store.__setitem__("content", content),
    )
    return store


def _use_statement_file(monkeypatch, text: str = STATEMENT) -> None:
    monkeypatch.setattr("voxa.agent.tools.statement._locate", lambda name: "/home/user/Downloads/march-2026-statement.csv")
    monkeypatch.setattr("voxa.agent.tools.statement._read_text", lambda path: text)


def test_phrases_route() -> None:
    assert route("import the statement march 2026") == ToolCall("import_statement", {"name": "march 2026"})
    assert route("read the statement called march statement") == ToolCall("import_statement", {"name": "march statement"})
    assert route("import my bank statement") == ToolCall("import_statement", {"name": ""})
    assert route("compare this month with last month") == ToolCall("statement_compare", {"month": "last month", "category": ""})
    assert route("how much did i spend on groceries last month") == ToolCall(
        "statement_compare",
        {"month": "last month", "category": "groceries"},
    )
    assert route("what is the biggest charge this month") == ToolCall("statement_report", {"month": ""})


def test_statement_rules_do_not_steal_other_commands() -> None:
    assert route("how much have i spent this week") == ToolCall("expense_total", {"period": "this week", "category": ""})
    assert route("what did i spend on food this month") == ToolCall("expense_total", {"period": "this month", "category": "food"})
    assert route("show my spending this month") == ToolCall("expense_breakdown", {"period": "this month"})
    assert route("remove the last expense") == ToolCall("expense_undo", {})
    assert route("export my expenses") == ToolCall("expense_spreadsheet", {})
    assert route("how much did the denver trip cost") == ToolCall("trip_cost", {"name": "denver"})
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_import_handler_end_to_end(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setenv("VOXA_STATEMENT_CORRECTIONS", "/nonexistent-voxa-tests/corrections.json")
    store = _use_store(monkeypatch)
    _use_statement_file(monkeypatch)
    result = _import_statement_handler({"name": "march 2026"})
    assert result.ok
    assert result.speech == (
        "I added 4 transactions from march 2026 to your spending log. I skipped 1 line that was not a transaction."
    )
    assert "| trip:march-2026-statement" in store["content"]
    assert "| food | Kroger Store" in store["content"]
    assert "| other | Mystery Mart" in store["content"]
    assert "TOTAL" not in store["content"]
    assert entries(store["content"]) == [
        (TODAY, 45.99, "food", "Kroger Store"),
        (TODAY, 12.0, "transport", "Shell Gas Station"),
        (TODAY, 7.25, "other", "Mystery Mart"),
        (LAST_DAY, 40.0, "food", "Kroger Store"),
    ]


def test_importing_the_same_statement_twice_does_not_double_count(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setenv("VOXA_STATEMENT_CORRECTIONS", "/nonexistent-voxa-tests/corrections.json")
    store = _use_store(monkeypatch)
    _use_statement_file(monkeypatch)
    assert _import_statement_handler({"name": "march 2026"}).ok
    content = store["content"]
    again = _import_statement_handler({"name": "march 2026"})
    assert again.ok
    assert again.speech == "I already imported march 2026: 4 transactions are in your spending log."
    assert store["content"] == content


def test_a_remembered_correction_applies_on_the_next_import(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setenv("VOXA_STATEMENT_CORRECTIONS", str(tmp_path / "corrections.json"))
    store = _use_store(monkeypatch)
    _use_statement_file(monkeypatch)
    assert _import_statement_handler({"name": "march 2026"}).ok
    assert "| other | Mystery Mart" in store["content"]
    remember_correction("Mystery Mart", "fun")
    store["content"] = ""
    assert _import_statement_handler({"name": "march 2026"}).ok
    assert "| fun | Mystery Mart" in store["content"]


def test_import_handler_cannot_find_the_file(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.statement._locate", lambda name: None)
    result = _import_statement_handler({"name": "april statement"})
    assert not result.ok
    assert result.speech == "I couldn't find april statement."


def test_import_handler_with_no_transactions_in_the_file(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.statement._locate", lambda name: "/home/user/Downloads/march-2026-statement.csv")
    monkeypatch.setattr("voxa.agent.tools.statement._read_text", lambda path: "Statement of March 2026\nNo rows here.\n")
    result = _import_statement_handler({"name": "march 2026"})
    assert not result.ok
    assert result.speech == "I couldn't find any transactions in march 2026."


def test_compare_handler_names_the_categories(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(
        monkeypatch,
        "- 2026-03-02 | 45.99 | food | Kroger Store\n"
        "- 2026-03-03 | 12.00 | transport | Shell Gas Station\n"
        "- 2026-02-15 | 40.00 | food | Kroger Store\n",
    )
    result = _statement_compare_handler({"month": "march 2026", "category": ""})
    assert result.ok
    assert result.speech == (
        "In March 2026 you spent 57 dollars and 99 cents, against 40 dollars in February 2026. "
        "The change: transport is up 12 dollars and food is up 5 dollars and 99 cents."
    )


def test_compare_handler_for_one_merchant_word(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(
        monkeypatch,
        f"- {THIS} | 45.99 | food | Kroger Store\n- {LAST} | 40.00 | food | Kroger Store\n",
    )
    result = _statement_compare_handler({"month": "this month", "category": "groceries"})
    assert result.ok
    assert result.speech == (
        f"In {THIS_LABEL} you spent 45 dollars and 99 cents on food, against 40 dollars in {LAST_LABEL}: "
        "food is up 5 dollars and 99 cents."
    )


def test_compare_handler_with_an_empty_log(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(monkeypatch, "")
    result = _statement_compare_handler({"month": "", "category": ""})
    assert not result.ok
    assert result.speech == "There is no spending recorded to compare."


def test_report_handler_writes_a_document_and_speaks_a_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path))
    _use_store(
        monkeypatch,
        f"- {THIS} | 45.99 | food | Kroger Store\n- {THIS} | 12.00 | transport | Shell Gas Station\n- {LAST} | 40.00 | food | Kroger Store\n",
    )
    opened: list[Path] = []
    monkeypatch.setattr(documents, "open_in_libreoffice", lambda path: opened.append(path))
    result = _statement_report_handler({"month": ""})
    assert result.ok
    assert len(opened) == 1
    path = opened[0]
    assert path.parent == tmp_path / "Voxa Drafts"
    assert path.suffix == ".odt"
    with zipfile.ZipFile(path) as archive:
        assert archive.read("mimetype") == b"application/vnd.oasis.opendocument.text"
        body = archive.read("content.xml").decode("utf-8")
    assert "Totals by category" in body
    assert "Kroger Store" in body
    assert f"{THIS_LABEL} against {LAST_LABEL}" in body
    assert "food" in result.speech and "transport" in result.speech
    assert result.speech.endswith(f"I wrote the report to {path.name} and opened it.")


def test_report_handler_with_an_empty_log(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    _use_store(monkeypatch, "")
    result = _statement_report_handler({"month": ""})
    assert not result.ok
    assert result.speech == "There are no expenses to report on."


def test_nothing_leaves_the_machine() -> None:
    from voxa.agent import statement as pure
    from voxa.agent.tools import statement as tools_module

    joined = Path(pure.__file__).read_text() + Path(tools_module.__file__).read_text()
    for forbidden in ("aiohttp", "requests", "urllib", "socket", "http", "curl", "wget", "scp", "ssh "):
        assert forbidden not in joined, forbidden


def test_registry_has_statement_tools() -> None:
    names = default_registry().names()
    for name in ("import_statement", "statement_compare", "statement_report"):
        assert name in names
