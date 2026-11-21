"""Orders and deliveries (issue #52's first step): paste text, read the tracking numbers,
open the carrier page, speak the status it states, and keep the local order list."""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date

from voxa import simulation
from voxa.agent import host, orders
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage, textedit

log = logging.getLogger(__name__)

fetch: Callable[[str], str] | None = None

REACH_ERROR = "I could not reach your orders."
NO_NUMBER = "I couldn't find a tracking number in that text."
NO_TEXT = "Paste or select the text first, then ask me to track it."


def _fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Voxa/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:  # noqa: S310 - fixed https hosts
        return response.read().decode("utf-8", "replace")


def _open(url: str) -> bool:
    return host.spawn(["xdg-open", url]) is not False


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def _path() -> str | None:
    folder = filemanage._folder("documents")
    if folder is None:
        return None
    return os.path.join(folder, orders.FILE_NAME)


def _read() -> str:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    done = _run(["cat", path], check=False)
    if done.returncode != 0:
        return ""
    return done.stdout


def _write(content: str) -> None:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    _run(["sh", "-c", 'cat > "$1"', "sh", path], stdin=content)


def _carrier_word(carrier: str) -> str:
    return "" if not carrier or carrier == "unknown" else f" for {carrier}"


def track_order(args: dict[str, str]) -> ToolResult:
    text = args.get("text", "")
    if not text.strip() and textedit.read_clipboard is not None:
        text = textedit.read_clipboard()
    if not text.strip():
        return ToolResult.failure(NO_TEXT)
    found = orders.parse_tracking_numbers(text)
    if not found:
        return ToolResult.failure(NO_NUMBER)
    note = args.get("note", "").strip() or orders.note_from_text(text)
    expected = args.get("expected", "").strip() or orders.expected_from_text(text)
    today = date.today().isoformat()
    try:
        rows = orders.parse(_read())
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    lines: list[str] = []
    details: list[str] = []
    saved = 0
    for carrier, number in found:
        url = f"https://parcelsapp.com/en/tracking/{urllib.parse.quote(number, safe='')}"
        details.append(url)
        page = ""
        reached = True
        try:
            page = (fetch or _fetch)(url)
        except Exception as exc:  # noqa: BLE001 - offline or blocked: the order is still worth saving
            log.info("carrier page unreachable: %s", exc)
            reached = False
        status = orders.status_from_page(page, number)
        rows, changes = orders.add_or_update(rows, number, carrier, note, status, expected, today)
        saved += 1
        if status:
            lines.append(f"Tracking {number}{_carrier_word(carrier)}: {status}.")
        elif reached:
            lines.append(
                f"Tracking {number}{_carrier_word(carrier)}: I opened the page, "
                "but it states no status I can read."
            )
        else:
            lines.append(
                f"Tracking {number}{_carrier_word(carrier)}: I could not reach the carrier page, "
                "so I have no status for it."
            )
        lines.extend(changes)
        if not _open(url):
            lines.append("I could not open that page in your browser.")
    try:
        _write(orders.render(rows))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    lines.append(f"Saved {saved} {'order' if saved == 1 else 'orders'} on your order list.")
    return ToolResult.success(" ".join(lines), detail=" ".join(details))


def order_status(args: dict[str, str]) -> ToolResult:
    try:
        rows = orders.parse(_read())
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    note = args.get("note", "")
    if not orders.note_key(note):
        return ToolResult.success(orders.spoken_changes(rows))
    return ToolResult.success(orders.spoken_for_note(rows, note))


def orders_arriving(args: dict[str, str]) -> ToolResult:
    try:
        rows = orders.parse(_read())
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    return ToolResult.success(orders.spoken_arriving(rows, args.get("when", "")))


def order_tools() -> list[Tool]:
    return [
        Tool(
            name="track_order",
            description=(
                "Read tracking numbers out of pasted or selected text, save each order on the "
                "order list, open its carrier page and speak the status that page states."
            ),
            parameters={
                "text": "the text you pasted or selected",
                "note": "what the order is for, e.g. filament",
                "expected": "the expected delivery date, if you say one",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=track_order,
        ),
        Tool(
            name="order_status",
            description="Say where an order on the order list stands, by what it is for.",
            parameters={"note": "what the order is for, e.g. filament; empty for every order"},
            risk=RiskLevel.READ_ONLY,
            handler=order_status,
        ),
        Tool(
            name="orders_arriving",
            description="Say which orders on the order list are due in the asked period.",
            parameters={"when": "today, tomorrow, this week, this month or soon"},
            risk=RiskLevel.READ_ONLY,
            handler=orders_arriving,
        ),
    ]
