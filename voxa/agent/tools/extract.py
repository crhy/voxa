from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable

from voxa.agent.extract import find_amounts, find_dates, spoken
from voxa.agent.filematch import STANDARD_NAMES
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

ask_model: Callable | None = None  # set by the window: messages -> text

NAMES_PROMPT = (
    "From the document text below, list the names of people or companies mentioned. "
    "Reply with at most 3 names, one per line, numbered 1., 2., 3. "
    "If there are none, reply with NONE."
)


def _locate(name: str) -> str | None:
    home = filemanage._folder("home")
    if home is None:
        return None
    words = name.split()
    pattern = "*" + "*".join(words) + "*"
    out = filemanage._run(
        [
            "timeout",
            "15",
            "find",
            home,
            "-xdev",
            "(",
            "-name",
            ".*",
            "-o",
            "-name",
            "node_modules",
            "-o",
            "-name",
            "__pycache__",
            ")",
            "-prune",
            "-o",
            "-iname",
            pattern,
            "-print",
        ],
        check=False,
    )
    lines = out.stdout.splitlines()
    if not lines:
        return None
    top = set(STANDARD_NAMES.values())

    def rank(path: str) -> tuple[int, int]:
        parent = os.path.dirname(path)
        is_top = os.path.basename(parent) in top or parent.rstrip("/") == home.rstrip("/")
        return (0 if is_top else 1, path.count("/"))

    return sorted(lines, key=rank)[0]


def _strip_tags(xml: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", text).strip()


def _read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        done = filemanage._run(["pdftotext", path, "-"], check=False)
        raw = done.stdout
    elif ext == ".odt":
        done = filemanage._run(["unzip", "-p", path, "content.xml"], check=False)
        raw = _strip_tags(done.stdout)
    elif ext == ".docx":
        done = filemanage._run(["unzip", "-p", path, "word/document.xml"], check=False)
        raw = _strip_tags(done.stdout)
    else:
        done = filemanage._run(["cat", path], check=False)
        raw = done.stdout
    return raw[:6000]


def _parse_names(answer: str) -> list[str]:
    out: list[str] = []
    for line in (answer or "").splitlines():
        s = re.sub(r"^\s*\d+[.)]\s*", "", line).strip().strip("-*").strip()
        if not s or s.upper() == "NONE":
            continue
        out.append(s)
        if len(out) >= 3:
            break
    return out


def _doc_extract_handler(args: dict[str, str]) -> ToolResult:
    name = (args.get("name") or "").strip()
    if not name:
        return ToolResult.failure("Which file should I read?")
    path = _locate(name)
    if path is None:
        return ToolResult.failure(f"I couldn't find {name}.")
    try:
        text = _read_text(path)
    except (subprocess.SubprocessError, OSError) as error:
        return ToolResult.failure(f"I could not read {name}.", detail=str(error))
    dates = find_dates(text)
    amounts = find_amounts(text)
    names: list[str] = []
    if ask_model is not None and text:
        try:
            answer = ask_model(
                [
                    {"role": "system", "content": NAMES_PROMPT},
                    {"role": "user", "content": text[:4000]},
                ]
            )
            names = _parse_names(answer)
        except Exception as exc:  # noqa: BLE001 - the model server is outside our control
            log.info("doc_extract names failed: %s", exc)
    return ToolResult.success(spoken(os.path.basename(path), dates, amounts, names))


def extract_tools() -> list[Tool]:
    return [
        Tool(
            name="doc_extract",
            description="Read a document and report the dates, money amounts and names it contains.",
            parameters={"name": "the file to read, e.g. invoice.pdf or my lease"},
            risk=RiskLevel.READ_ONLY,
            handler=_doc_extract_handler,
            required=("name",),
        ),
    ]
