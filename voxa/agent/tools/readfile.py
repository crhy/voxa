"""Read a text file aloud: "Voxa, read the file called notes", "read this text file".

The Flatpak only sees the Documents folder, so files are listed and read on the host through the
same bridge the other host tools use, read-only and capped in size. The same tool drives the
"read-file" application action, which scripted recordings use to give Voxa her lines.
"""

from __future__ import annotations

import os
import re
import subprocess
from urllib.parse import unquote, urlparse

from voxa.agent.filematch import best_match
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

MAX_BYTES = 20_000
SEARCH_FOLDERS = ("", "Documents", "Desktop", "Downloads")
TEXT_SUFFIXES = (".txt", ".md", ".text")
_MARKUP = re.compile(r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|>\s?|\d+[.)]\s+)")
_INLINE = re.compile(r"(\*\*|__|`|\*)")

def _selected_from_textedit() -> str:
    # textedit sends Ctrl+C and reads the clipboard; the window connects its clipboard at start.
    from voxa.agent.tools import textedit

    if textedit.read_clipboard is None or textedit.write_clipboard is None:
        return ""
    return textedit._selected_text()


# Swapped for a fake in tests.
selected_text = _selected_from_textedit


def _host_run(command: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(host_command(command), capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def host_read(path: str) -> bytes | None:
    """At most MAX_BYTES + 1 bytes of the file, or None when it cannot be read."""
    result = _host_run(["head", "-c", str(MAX_BYTES + 1), "--", path])
    if result is None or result.returncode != 0:
        return None
    return result.stdout


def host_list(folder: str) -> list[str]:
    result = _host_run(["ls", "-1A", "--", folder])
    if result is None or result.returncode != 0:
        return []
    return [line for line in result.stdout.decode("utf-8", "replace").splitlines() if line]


def path_from_selection(text: str) -> str | None:
    """A file selected in Caja copies as its path (or a file:// URI); anything else is not a file."""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first.startswith("file://"):
        return unquote(urlparse(first).path)
    if first.startswith("/"):
        return first
    return None


def find_file(name: str, home: str | None = None) -> str | None:
    home = home or os.path.expanduser("~")
    if name.startswith(("/", "~")):
        return os.path.expanduser(name)
    for folder in SEARCH_FOLDERS:
        directory = os.path.join(home, folder) if folder else home
        texts = [entry for entry in host_list(directory) if entry.lower().endswith(TEXT_SUFFIXES)]
        bare = {os.path.splitext(entry)[0]: entry for entry in texts}
        match = best_match(name, list(bare))
        if match is not None:
            return os.path.join(directory, bare[match])
    return None


def plain_text(raw: bytes) -> str | None:
    """Speakable text, or None when the bytes are not text (NUL bytes or invalid UTF-8)."""
    if b"\0" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [" ".join(_INLINE.sub("", _MARKUP.sub("", line)).split()) for line in text.splitlines()]
    # A line break is a pause: headings, list items and lines of a poem get a full stop.
    spoken = [line if line[-1] in ".!?:;," else line + "." for line in lines if line]
    return " ".join(spoken)


def read_text_file(args: dict[str, str]) -> ToolResult:
    name = (args.get("name") or "").strip()
    if not name:
        selection = selected_text() if selected_text is not None else ""
        path = path_from_selection(selection)
        if path is None:
            return ToolResult.failure("Select a text file first, or tell me its name.")
    else:
        path = find_file(name)
        if path is None:
            return ToolResult.failure(f"I could not find a text file called {name}.")
    raw = host_read(path)
    if raw is None:
        return ToolResult.failure(f"I could not open {os.path.basename(path)}.")
    if len(raw) > MAX_BYTES:
        # Only look for NUL bytes here: the cut can split a multi-byte character.
        if b"\0" in raw:
            return ToolResult.failure("That does not look like a text file.")
        return ToolResult.failure("That file is too long for me to read aloud.")
    text = plain_text(raw)
    if text is None:
        return ToolResult.failure("That does not look like a text file.")
    if not text:
        return ToolResult.failure(f"{os.path.basename(path)} is empty.")
    return ToolResult.success(text, detail=path)


def readfile_tools() -> list[Tool]:
    return [
        Tool(
            name="read_text_file",
            description=(
                "Read a text file aloud. name is the file's name or path; leave it empty to read the "
                "file selected in the file manager."
            ),
            parameters={"name": "file name or path, or empty for the selected file"},
            risk=RiskLevel.READ_ONLY,
            handler=read_text_file,
            required=(),
        ),
    ]
