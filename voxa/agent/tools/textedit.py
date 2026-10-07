from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable

from voxa import simulation
from voxa.agent.cleanup import CLEANUP_PROMPT, sane_result
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

ask_model: Callable | None = None
read_clipboard: Callable[[], str] | None = None
write_clipboard: Callable[[str], None] | None = None


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=5,
        check=check,
    )


def cleanup_text(args: dict[str, str]) -> ToolResult:
    if read_clipboard is None or write_clipboard is None or ask_model is None:
        return ToolResult.failure("I cannot reach the clipboard here.")
    saved = read_clipboard()
    write_clipboard("")
    _run(["xdotool", "key", "--clearmodifiers", "ctrl+c"])
    time.sleep(0.25)
    copied = read_clipboard()
    if copied.strip():
        whole = False
    else:
        _run(["xdotool", "key", "--clearmodifiers", "ctrl+a"])
        _run(["xdotool", "key", "--clearmodifiers", "ctrl+c"])
        time.sleep(0.25)
        copied = read_clipboard()
        whole = True
    if not copied.strip():
        write_clipboard(saved)
        if whole:
            _run(["xdotool", "key", "Right"])
        return ToolResult.failure("I could not read any text in that window.")
    if len(copied) > 20000:
        write_clipboard(saved)
        if whole:
            _run(["xdotool", "key", "Right"])
        return ToolResult.failure("That is too much text for me to clean up in one go.")
    edited = ask_model(
        [
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": copied},
        ]
    )
    if sane_result(copied, edited):
        write_clipboard(edited)
        time.sleep(0.1)
        _run(["xdotool", "key", "--clearmodifiers", "ctrl+v"])
        time.sleep(0.4)
        write_clipboard(saved)
        speech = "Text edited for clarity." if whole else "Selection edited for clarity."
        return ToolResult.success(speech)
    write_clipboard(saved)
    if whole:
        _run(["xdotool", "key", "Right"])
    return ToolResult.success("I was not sure about my edit, so I left your text as it was.")


SUMMARY_PROMPT = "Summarise the user's text in at most three short spoken sentences. Plain words, no lists, no preamble."


def _selected_text() -> str:
    saved = read_clipboard()
    write_clipboard("")
    _run(["xdotool", "key", "--clearmodifiers", "ctrl+c"])
    time.sleep(0.25)
    text = read_clipboard()
    write_clipboard(saved)
    return text.strip()


def speakable(text: str, limit: int = 900) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    window = collapsed[:limit]
    cut = -1
    for marker in (". ", "! ", "? "):
        pos = window.rfind(marker)
        if pos != -1 and pos + 1 > cut:
            cut = pos + 1
    if cut == -1:
        space = window.rfind(" ")
        cut = space if space != -1 else limit
    return collapsed[:cut].rstrip() + " … That is the first part."


def read_selection(args: dict[str, str]) -> ToolResult:
    if read_clipboard is None or write_clipboard is None:
        return ToolResult.failure("I cannot reach the clipboard here.")
    text = _selected_text()
    if not text:
        return ToolResult.failure("Select some text first, then ask me to read it.")
    return ToolResult.success(speakable(text))


def read_clipboard_aloud(args: dict[str, str]) -> ToolResult:
    if read_clipboard is None:
        return ToolResult.failure("I cannot reach the clipboard here.")
    text = read_clipboard().strip()
    if not text:
        return ToolResult.failure("The clipboard is empty.")
    return ToolResult.success("The clipboard says: " + speakable(text, 600))


def summarize_selection(args: dict[str, str]) -> ToolResult:
    if read_clipboard is None or write_clipboard is None:
        return ToolResult.failure("I cannot reach the clipboard here.")
    text = _selected_text()
    if not text:
        return ToolResult.failure("Select some text first, then ask me to read it.")
    if len(text) > 20000:
        return ToolResult.failure("That is too much text for me to summarise in one go.")
    if ask_model is None:
        return ToolResult.failure("I cannot reach the clipboard here.")
    summary = ask_model(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    if not summary.strip():
        return ToolResult.failure("I could not summarise that.")
    return ToolResult.success(speakable(summary.strip(), 600))


def textedit_tools() -> list[Tool]:
    return [
        Tool(
            name="cleanup_text",
            description="Clean up the text in the focused window for clarity, spelling and grammar.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=cleanup_text,
            required=(),
        ),
        Tool(
            name="read_selection",
            description="Read the selected text in the focused window aloud.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=read_selection,
            required=(),
        ),
        Tool(
            name="read_clipboard_aloud",
            description="Read what is on the clipboard aloud.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=read_clipboard_aloud,
            required=(),
        ),
        Tool(
            name="summarize_selection",
            description="Summarise the selected text in the focused window in three short sentences.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=summarize_selection,
            required=(),
        ),
    ]
