"""Saving a file by voice: "What should I save the file as?", the name, then Save.

Pure conversation logic: what Voxa says and what the caller must do in the file dialog next.
The dialog itself is driven by ``voxa.agent.tools.files``. An existing file is simply
replaced; Voxa never asks first.
"""

from __future__ import annotations

import re

QUESTION = "What should I save the file as?"
ASK_NAME = "ask-save-name"  # ToolResult.detail: a save dialog is open and needs a name

_CANCEL = re.compile(r"^(?:cancel|never ?mind|forget it|stop|don'?t save(?: it)?|no(?:,? thanks)?)[.!]?$", re.IGNORECASE)
_LEAD = re.compile(
    r"^(?:(?:please\s+)?(?:save|call|name|title)\s+(?:it|this|the file|the document)\s+(?:as\s+)?|as\s+|it'?s\s+)", re.IGNORECASE
)
_EXTENSION = re.compile(r"\s+dot\s+([a-z0-9]{1,5})$", re.IGNORECASE)
_FORBIDDEN = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def clean_name(spoken: str) -> str:
    """The file name a person meant: no lead-in words, no closing full stop, nothing a file name cannot hold."""
    name = spoken.strip().strip("\"'“”‘’ ")
    name = _LEAD.sub("", name).strip()
    name = re.sub(r"[.!?,;]+$", "", name).strip()
    name = _EXTENSION.sub(lambda match: "." + match.group(1).lower(), name)
    name = _FORBIDDEN.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip(" .")[:120]


class SaveFlow:
    """One save: ``feed`` returns (what to say, what to do).

    Actions: ``"type"`` (put ``self.name`` in the dialog and press Save) or ``"cancel"``
    (close the dialog), or ``None`` (only speak).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.state = "name"

    @property
    def active(self) -> bool:
        return self.state != "done"

    @property
    def caption(self) -> str:
        return "Naming the file…"

    def feed(self, text: str) -> tuple[str, str | None]:
        said = text.strip()
        if _CANCEL.match(said):
            self.state = "done"
            return "Okay, I did not save it.", "cancel"
        name = clean_name(said)
        if not name:
            return "I didn't catch a name. " + QUESTION, None
        self.name = name
        return "", "type"

    def saved(self) -> str:
        self.state = "done"
        return f"Saved as {self.name}."

    def replaced(self) -> str:
        """The existing file was replaced without asking."""
        self.state = "done"
        return f"Replaced {self.name}."

    def failed(self) -> str:
        self.state = "done"
        return "I could not save it. The save window is still open."
