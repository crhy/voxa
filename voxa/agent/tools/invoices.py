"""Make an invoice from one spoken sentence, and read it back.

The sentence is parsed by `voxa/agent/invoices.py`; the document is written by
`voxa/documents.py` with the standard library only (an ODT, which the desktop's
LibreOffice renders and prints) and opened there for the user to keep or delete.
Nothing is sent anywhere: sending an invoice is #56.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from voxa import documents
from voxa.agent import invoices
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your documents."
NO_CLIENT = "Who is the invoice for?"


def _documents_folder() -> Path | None:
    folder = filemanage._folder("documents")
    if not folder:
        return None
    return Path(folder)


def _make_invoice_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    invoice = invoices.parse_invoice_line(args.get("line", ""), today)
    if invoice is None:
        return ToolResult.failure(NO_CLIENT)
    folder = _documents_folder()
    if folder is None:
        return ToolResult.failure(REACH_ERROR)
    reference = invoices.next_reference()
    title = invoices.title(invoice, reference)
    try:
        path = documents.unique_path(title, folder)
        documents.write_odt(path, title, invoices.body(invoice, today, reference))
    except OSError:
        return ToolResult.failure(REACH_ERROR)
    invoices.remember(invoice, reference, today)
    documents.open_in_libreoffice(path)
    return ToolResult.success(f"{invoices.spoken_invoice(invoice, today, reference)} I opened it in LibreOffice.")


def _invoice_read_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    found = invoices.last_invoice(today)
    if found is None:
        return ToolResult.failure("I have not made an invoice yet.")
    invoice, reference = found
    return ToolResult.success(invoices.spoken_invoice(invoice, today, reference))


def invoice_tools() -> list[Tool]:
    return [
        Tool(
            name="make_invoice",
            description="Turn one spoken sentence into a numbered invoice document and open it in LibreOffice.",
            parameters={"line": "the whole sentence you said, client and amounts included"},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=_make_invoice_handler,
            required=("line",),
        ),
        Tool(
            name="invoice_read",
            description="Read back the last invoice made: client, total and when it is due.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_invoice_read_handler,
        ),
    ]
