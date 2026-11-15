"""Bank statements the user already has on disk: import, compare, report.

Everything is read locally and written locally; no transaction data leaves the
machine. The statement file is located the way doc_extract locates a document and
read with the same text reader (pdftotext for a PDF, cat for a CSV or text file);
its transactions go into the same spending log the expense tools use.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import documents
from voxa.agent.expenses import CATEGORIES, add, entries, tagged
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.statement import (
    candidate_lines,
    categorise,
    compare_months,
    load_corrections,
    parse_statement,
    report_body,
    spoken_category_compare,
    spoken_compare,
    spoken_largest,
)
from voxa.agent.tools.expenses import REACH_ERROR, _read, _write
from voxa.agent.tools.extract import _locate, _read_text

log = logging.getLogger(__name__)


def _statement_tag(path: str) -> str:
    """The tag naming the statement, so its rows can be traced back to the file."""
    return os.path.splitext(os.path.basename(path))[0]


def _import_statement_handler(args: dict[str, str]) -> ToolResult:
    name = args.get("name", "").strip()
    if not name:
        return ToolResult.failure("Which statement should I import?")
    path = _locate(name)
    if path is None:
        return ToolResult.failure(f"I couldn't find {name}.")
    try:
        text = _read_text(path)
    except (subprocess.SubprocessError, OSError) as error:
        return ToolResult.failure(f"I could not read {name}.", detail=str(error))
    rows = parse_statement(text)
    if not rows:
        return ToolResult.failure(f"I couldn't find any transactions in {name}.")
    tag = _statement_tag(path)
    corrections = load_corrections()
    items = [(day, amount, categorise(description, corrections), description) for day, amount, description in rows]
    skipped = max(0, candidate_lines(text) - len(rows))
    skipped = max(0, candidate_lines(text) - len(rows))
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    already = tagged(content, tag)
    if already:
        return ToolResult.success(
            f"I already imported {name}: {len(already)} transactions are in your spending log."
        )
    try:
        for day, amount, category, description in items:
            content = add(content, day, amount, description, tag, category)
        _write(content)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    speech = f"I added {len(items)} transactions from {name} to your spending log."
    if skipped:
        speech += (
            " I skipped 1 line that was not a transaction."
            if skipped == 1
            else f" I skipped {skipped} lines that were not transactions."
        )
    return ToolResult.success(speech)


def _statement_compare_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    items = entries(content)
    if not items:
        return ToolResult.failure("There is no spending recorded to compare.")
    result = compare_months(items, args.get("month", ""), date.today())
    category = args.get("category", "").strip().lower()
    if category and category not in CATEGORIES:
        # "how much did I spend on groceries last month": the merchant word
        # picks the category through the same rules the statement import uses.
        category = categorise(category)
    if category:
        return ToolResult.success(spoken_category_compare(result, category))
    return ToolResult.success(spoken_compare(result))


def _statement_report_handler(args: dict[str, str]) -> ToolResult:
    try:
        content = _read()
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    items = entries(content)
    if not items:
        return ToolResult.failure("There are no expenses to report on.")
    month = args.get("month", "") or args.get("period", "")
    result = compare_months(items, month, date.today())
    path = documents.unique_path(f"Voxa spending {result['label']}")
    try:
        documents.write_odt(path, f"Voxa spending report, {result['label']}", report_body(items, result))
        documents.open_in_libreoffice(path)
    except OSError as error:
        return ToolResult.failure("I could not write the report.", detail=str(error))
    return ToolResult.success(
        f"{spoken_compare(result)} {spoken_largest(items)} I wrote the report to {path.name} and opened it."
    )


def statement_tools() -> list[Tool]:
    return [
        Tool(
            name="import_statement",
            description="Read a bank statement file the user already has and add its transactions to the spending log.",
            parameters={"name": "the statement file, e.g. march statement or statement of march 2026"},
            risk=RiskLevel.REVERSIBLE,
            handler=_import_statement_handler,
            required=("name",),
        ),
        Tool(
            name="statement_compare",
            description="Compare one month of spending with the month before it, overall or for one category.",
            parameters={
                "month": "which month, e.g. this month, last month or march 2026",
                "category": "which category, if any",
            },
            risk=RiskLevel.READ_ONLY,
            handler=_statement_compare_handler,
        ),
        Tool(
            name="statement_report",
            description="Write a spending report as a document and speak a short summary of it.",
            parameters={"month": "which month to report on, e.g. this month or march 2026"},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=_statement_report_handler,
        ),
    ]
