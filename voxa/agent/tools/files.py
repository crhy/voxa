from __future__ import annotations

import logging

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.typing import press_keys, type_text, xdotool_available

log = logging.getLogger("voxa.agent.tools.files")

# The focused editor (Pluma, LibreOffice Writer, ...) owns the file dialogs and
# remembers the directory it last saved or loaded from; Voxa only drives those
# dialogs with the same key combos the app itself uses.
_SAVE_AS = "ctrl+shift+s"
_OPEN_FILE = "ctrl+o"
_CLOSE_FILE = "ctrl+w"
_NEW_DOCUMENT = "ctrl+n"


def file_dialog(args: dict[str, str]) -> ToolResult:
    action = args["action"]
    name = args.get("name", "")
    if not xdotool_available():
        return ToolResult.failure("I can't control the file dialogs here.")
    try:
        if action == "save":
            press_keys(_SAVE_AS)
            if name:
                type_text(name)
                press_keys("Return")
        elif action == "load":
            press_keys(_OPEN_FILE)
        elif action == "close":
            press_keys(_CLOSE_FILE)
        elif action == "new":
            press_keys(_NEW_DOCUMENT)
        else:
            return ToolResult.failure("I don't know that file action.")
    except Exception as exc:
        return ToolResult.failure("I can't control the file dialogs here.", detail=str(exc))

    if action == "save":
        return ToolResult.success(f"Saved as {name}." if name else "Saved the file.")
    if action == "load":
        return ToolResult.success("Opened the file.")
    if action == "close":
        return ToolResult.success("Closed the file.")
    return ToolResult.success("Started a new document.")


def file_dialog_tools() -> list[Tool]:
    return [
        Tool(
            name="file_dialog",
            description=(
                "Drive the focused editor's file dialogs: save the file (optionally "
                "renamed), load a file from the current directory, close the file, "
                "or start a new document."
            ),
            parameters={
                "action": "one of save, load, close, new",
                "name": "the filename to save as, for the save action",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=file_dialog,
            required=("action",),
        ),
    ]
