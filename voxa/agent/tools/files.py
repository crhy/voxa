from __future__ import annotations

import logging
import time

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.typing import press_keys, type_text, xdotool_available

log = logging.getLogger("voxa.agent.tools.files")

# The focused editor (Pluma, LibreOffice Writer, ...) owns the file dialogs and
# remembers the directory it last saved or loaded from; Voxa only drives those
# dialogs with the same key combos the app itself uses.
_SAVE = "ctrl+s"
_SAVE_AS = "ctrl+shift+s"
_OPEN_FILE = "ctrl+o"
_CLOSE_FILE = "ctrl+w"
_NEW_DOCUMENT = "ctrl+n"


def _active_window() -> tuple[str, str]:
    """(id, title) of the window in front; ("", "") when it cannot be read."""
    from voxa.agent.tools.windows import _run

    try:
        wid = _run(["xdotool", "getactivewindow"], check=False).stdout.strip()
        title = _run(["xdotool", "getactivewindow", "getwindowname"], check=False).stdout.strip() if wid else ""
        return wid, title
    except Exception:  # noqa: BLE001 - no window information is an answer too
        return "", ""


def _wait_for(condition, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


class SaveDialog:
    """Drives one editor's save dialog. The editor's window is remembered so "saved" can be told
    apart from "another question appeared" (the file exists) and "the dialog is still there"."""

    def __init__(self) -> None:
        self.main_id, _title = _active_window()
        self.dialog_id = ""

    def open(self, combo: str) -> bool:
        """Press the save keys; True when a dialog appeared (False: the file simply saved)."""
        press_keys(combo)
        appeared = _wait_for(lambda: _active_window()[0] not in ("", self.main_id), 1.5)
        if appeared:
            self.dialog_id = _active_window()[0]
        return appeared

    def type_name(self, name: str) -> str:
        """Put the name in the dialog and press Save: "saved", "exists" or "failed"."""
        press_keys("ctrl+a")  # the name field has the focus; replace whatever it suggests
        type_text(name)
        time.sleep(0.15)
        press_keys("Return")
        return self._outcome()

    def replace(self) -> str:
        for combo in ("alt+r", "alt+y", "Return"):  # GTK "Replace", LibreOffice "Yes", then the default
            press_keys(combo)
            if _wait_for(lambda: _active_window()[0] == self.main_id, 1.2):
                return "saved"
        return "failed"

    def dismiss_question(self) -> None:
        press_keys("Escape")
        _wait_for(lambda: _active_window()[0] == self.dialog_id, 1.0)

    def cancel(self) -> None:
        for _ in range(2):
            if _active_window()[0] in ("", self.main_id):
                return
            press_keys("Escape")
            time.sleep(0.3)

    def _outcome(self) -> str:
        if _wait_for(lambda: _active_window()[0] == self.main_id, 1.5):
            return "saved"
        return "failed" if _active_window()[0] == self.dialog_id else "exists"


CURRENT_SAVE: SaveDialog | None = None  # the dialog a save conversation is talking about


def file_dialog(args: dict[str, str]) -> ToolResult:
    global CURRENT_SAVE
    action = args["action"]
    name = args.get("name", "")
    if not xdotool_available():
        return ToolResult.failure("I can't control the file dialogs here.")
    try:
        if action == "save":
            from voxa.agent import saveflow

            dialog = SaveDialog()
            if not name:
                # Plain save first: a document that already has a name just saves.
                if not dialog.open(_SAVE):
                    return ToolResult.success("Saved.")
                CURRENT_SAVE = dialog
                return ToolResult.success(saveflow.QUESTION, detail=saveflow.ASK_NAME)
            name = saveflow.clean_name(name)
            if not dialog.open(_SAVE_AS):
                return ToolResult.failure("The save window did not open.")
            outcome = dialog.type_name(name)
            if outcome == "exists":
                CURRENT_SAVE = dialog
                return ToolResult.success(
                    f"A file called {name} already exists. Replace it?", detail=saveflow.ASK_REPLACE + name
                )
            if outcome == "failed":
                return ToolResult.failure("I could not save it. The save window is still open.")
            return ToolResult.success(f"Saved as {name}.")
        if action == "load":
            press_keys(_OPEN_FILE)
        elif action == "close":
            press_keys(_CLOSE_FILE)
        elif action == "new":
            press_keys(_NEW_DOCUMENT)
        else:
            return ToolResult.failure("I don't know that file action.")
    except Exception as exc:
        return ToolResult.failure("I can't control the file dialogs here.", detail=str(exc))

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
