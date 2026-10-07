from __future__ import annotations

import re

FILE_NAME = "Voxa Notes.md"

_NOTE_LINE = re.compile(r"^- \d{4}-\d{2}-\d{2} \d{2}:\d{2} — (.+)$")


def note_line(text: str, stamp: str) -> str:
    body = text.strip()
    if body:
        body = body[0].upper() + body[1:]
        if body[-1] not in ".!?":
            body += "."
    return f"- {stamp} — {body}"


def parse_notes(content: str) -> list[str]:
    notes = []
    for line in content.splitlines():
        match = _NOTE_LINE.match(line)
        if match:
            notes.append(match.group(1))
    return notes


def without_last(content: str) -> tuple[str, str | None]:
    lines = content.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        match = _NOTE_LINE.match(lines[i])
        if match:
            text = match.group(1)
            del lines[i]
            new_content = "\n".join(lines)
            if content.endswith("\n"):
                new_content += "\n"
            return new_content, text
    return content, None


def spoken_notes(notes: list[str], limit: int = 5) -> str:
    if not notes:
        return "You have no notes."
    if len(notes) == 1:
        return f"You have one note: {notes[0]}"
    newest_first = list(reversed(notes))[:limit]
    return f"You have {len(notes)} notes. The latest are: {' '.join(newest_first)}"
