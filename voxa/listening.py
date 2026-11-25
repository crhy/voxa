"""Scripted listening state (issue #130): one small file that tells live tests the truth.

Voxa only starts listening when a person presses Active (privacy), so a live test
could not run cases against her: v09's live run started while she sat at "Press
Active to start listening" and every case came back `not_heard`. The app now takes
`gapplication action io.github.crhy.voxa start-listening` (and `stop-listening`)
exactly as if the button had been pressed, and this module writes what actually
happened to listening.json - `{"active": bool, "since": epoch, "reason": str}` -
from the same code path the button uses, on every change. VoxaTest reads it back.

Pure Python: no GTK, so the window's tests and the harness both use it. Writing is
never allowed to break the button path: the caller keeps the file best-effort.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

__all__ = ["default_path", "read_state", "write_state"]


def default_path(environ: dict[str, str] | None = None) -> Path:
    """Where the listening state lives: VOXA_LISTENING_STATE, else the XDG state dir."""
    env = os.environ if environ is None else environ
    override = env.get("VOXA_LISTENING_STATE")
    if override:
        return Path(override)
    state = env.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "voxa" / "listening.json"


def write_state(path: Path, active: bool, reason: str = "") -> None:
    """Record the real state: when it changed and, when it did not, why."""
    payload = {"active": bool(active), "since": int(time.time()), "reason": str(reason or "")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def read_state(path: Path) -> dict | None:
    """The recorded state, or None when the file is missing, unreadable or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
        return None
    return data
