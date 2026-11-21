"""A receipt photo or scan into the spending ledger: "log this receipt".

The image is read on the host by tesseract, the text reader that ships with the desktop;
no OCR package is imported, so nothing has to be installed for this to work. The ledger
itself is the one `voxa/agent/expenses.py` keeps, so a receipt and a spoken expense land
side by side and a wrong one is removed the same way ("undo the last expense").

Receipts that arrive **from email** are still open: they need crhy/voxa#56, which needs
#41's secrets store, and that is not on main.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import date

from voxa import simulation
from voxa.agent.expenses import add, category_for
from voxa.agent.filematch import best_match
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.receipts import parse_receipt, receipt_date, receipt_what, spoken_receipt
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage
from voxa.agent.tools.expenses import REACH_ERROR, _read, _write

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
SEARCH_FOLDERS = ("downloads", "documents", "pictures", "desktop")
OCR_TIMEOUT_SECONDS = 20

NO_RECEIPT = "I could not find that receipt. Say its name, or the folder it is in."
NOT_AN_IMAGE = "That is not a picture I can read. A receipt photo or a scan is."
NO_READER = "No text reader is installed here, so I cannot read a receipt off a picture."
UNREADABLE = "I can't read a total on that receipt."


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command[:2])
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=OCR_TIMEOUT_SECONDS,
        check=check,
    )


def _ocr_available() -> bool:
    """Whether the host has a text reader at all; the probe the screenshot tool uses."""
    try:
        probe = _run(["sh", "-c", 'command -v "$1"', "sh", "tesseract"], check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _ocr(path: str) -> str:
    """The words read out of the image, empty when the reader finds none.

    Swapped for a fake in tests; the real binary is only reached where it is installed.
    """
    if simulation.actions_simulated():
        log.info("simulated action: would run tesseract on %s", path)
        return ""
    try:
        done = _run(["tesseract", path, "-"], check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout


def _is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_SUFFIXES)


def _exists(path: str) -> bool:
    try:
        return _run(["test", "-f", path], check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _newest_image(folder: str) -> str:
    """The newest picture in a folder: what "this receipt" points at with nothing named."""
    if not folder:
        return ""
    try:
        out = _run(["ls", "-1t", "--", folder], check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    for name in out.stdout.splitlines():
        if name.startswith("."):
            continue
        if _is_image(name):
            return os.path.join(folder, name)
    return ""


def _resolve(spoken: str) -> str | None:
    """The receipt file a spoken phrase points at, or None when it points at nothing.

    An absolute or ~ path is used as it stands; a folder word ("my downloads") picks the
    newest picture in that folder; anything else is matched as a file name in the folders
    a receipt is usually kept in.
    """
    name = (spoken or "").strip()
    if not name:
        for spoken_folder in SEARCH_FOLDERS:
            found = _newest_image(filemanage._folder(spoken_folder) or "")
            if found:
                return found
        return None
    if name.startswith(("/", "~")):
        return os.path.expanduser(name)
    folder = filemanage._folder(name)
    if folder is not None:
        return _newest_image(folder) or None
    for spoken_folder in SEARCH_FOLDERS:
        directory = filemanage._folder(spoken_folder)
        if directory is None:
            continue
        try:
            names = filemanage._names(directory)
        except (OSError, subprocess.SubprocessError):
            continue
        match = best_match(name, names)
        if match is not None:
            return os.path.join(directory, match)
    return None


def _receipt_log_handler(args: dict[str, str]) -> ToolResult:
    path = _resolve(args.get("path", ""))
    if path is None:
        return ToolResult.failure(NO_RECEIPT)
    if not _is_image(path):
        return ToolResult.failure(NOT_AN_IMAGE)
    if not _exists(path):
        return ToolResult.failure(NO_RECEIPT)
    if not _ocr_available():
        return ToolResult.failure(NO_READER)
    text = _ocr(path)
    amount, _merchant, _items = parse_receipt(text)
    if amount is None:
        # Nothing is written: a receipt whose total cannot be read must not guess one.
        return ToolResult.failure(UNREADABLE)
    what = receipt_what(text)
    category = category_for(what)
    day = receipt_date(text, date.today())
    try:
        content = _read()
        _write(add(content, day, amount, what))
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(REACH_ERROR)
    # The file is named out loud: "this receipt" can point at anything, and a wrong entry
    # is then easy to hear and to undo.
    return ToolResult.success(f"{spoken_receipt(amount, day, category)} From {os.path.basename(path)}.", detail=path)


def receipt_tools() -> list[Tool]:
    return [
        Tool(
            name="receipt_log",
            description=(
                "Read a receipt photo or scan with local text recognition and add the amount "
                "it shows to the spending ledger, in the category it belongs to."
            ),
            parameters={"path": "which receipt, e.g. the scan in my downloads"},
            risk=RiskLevel.REVERSIBLE,
            handler=_receipt_log_handler,
        ),
    ]
