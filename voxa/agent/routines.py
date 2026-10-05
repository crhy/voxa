"""Routines: one phrase runs several commands. Pure Python, no GTK, no host.

A routine is a small record (name, phrases, steps) kept in ``routines.json``
next to the action log, so it survives a restart. Each step is a spoken command
string that goes through the normal router/planner, so anything Voxa can do can
be a step. ``parse_create`` turns the spoken phrasings ("when I say <phrase>,
<step> and <step>" and "create a routine called <name> that <step> and <step>")
into a routine; ``match`` finds a routine from a phrase or its name.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from voxa.agent.actionlog import default_path as action_default_path
from voxa.agent.hearing import normalize
from voxa.agent.planner import COMMAND_VERBS

log = logging.getLogger(__name__)

__all__ = ["Routine", "RoutineStore", "parse_create", "parse_list", "parse_delete", "default_path"]


def default_path() -> Path:
    """Where routines.json lives: next to the action log."""
    return action_default_path().parent / "routines.json"


@dataclass(frozen=True, slots=True)
class Routine:
    """One routine: a name, the phrases that trigger it, and its steps."""

    name: str
    phrases: tuple[str, ...]
    steps: tuple[str, ...]


def _starts_with_verb(text: str) -> bool:
    """True when the text begins with a known command verb."""
    first = re.match(r"[a-z]+", text.strip().casefold())
    return first is not None and first.group(0) in COMMAND_VERBS


def _split_ambiguous(part: str) -> list[str]:
    """Split one stretch of text on ", " and " and " only when both sides are commands."""
    segments = [part]
    for sep in (", ", " and "):
        nxt: list[str] = []
        for seg in segments:
            pieces = seg.split(sep)
            if len(pieces) >= 2 and all(_starts_with_verb(p) for p in pieces):
                nxt.extend(pieces)
            else:
                nxt.append(seg)
        segments = nxt
    return segments


def split_steps(text: str) -> list[str]:
    """Split a run of spoken steps on the strong separators, then the ambiguous ones."""
    parts = re.split(r"\s+and then\s+|\s+then\s+|,\s+and\s+", text)
    out: list[str] = []
    for part in parts:
        out.extend(_split_ambiguous(part))
    return [p.strip().strip(".!?,;") for p in out if p.strip()]


_WHEN = re.compile(r"^when\s+i\s+say\s+(.+?),\s*(.+)$", re.IGNORECASE | re.DOTALL)
_CALLED = re.compile(r"^create a routine called (.+?) that (.+)$", re.IGNORECASE | re.DOTALL)
_LIST = re.compile(r"^(?:list|show)(?:\s+(?:my|the|all))? routines?$", re.IGNORECASE)
_DELETE = re.compile(r"^(?:delete|remove|forget) (?:the |a )?(.+?) routine$", re.IGNORECASE)
_TRIGGER_PREFIX = re.compile(r"^(?:run|start|do|it's|time for)\s+", re.IGNORECASE)


def parse_create(text: str) -> tuple[str, list[str]] | None:
    """Turn a spoken "make a routine" into (name, steps), or None when it is not one."""
    s = normalize(text).strip()
    match = _WHEN.match(s) or _CALLED.match(s)
    if not match:
        return None
    name = match.group(1).strip().strip(".!?,;")
    steps = split_steps(match.group(2))
    if not name or not steps:
        return None
    return name, steps


def parse_list(text: str) -> bool:
    """True for "list my routines" / "show routines"."""
    return bool(_LIST.match(normalize(text).strip()))


def parse_delete(text: str) -> str | None:
    """The routine name in "delete the <name> routine", or None when it is not one."""
    match = _DELETE.match(normalize(text).strip())
    return match.group(1).strip() if match else None


class RoutineStore:
    """The routines.json file: survives a restart, never raises on save."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._lock = threading.Lock()
        self._items: dict[str, Routine] = {}
        self._load()

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(data, list):
            return
        for entry in data:
            try:
                self._items[str(entry["name"])] = Routine(
                    name=str(entry["name"]),
                    phrases=tuple(str(p) for p in entry.get("phrases", [])),
                    steps=tuple(str(s) for s in entry.get("steps", [])),
                )
            except (KeyError, TypeError):
                continue

    def _save(self) -> None:
        data = [
            {"name": r.name, "phrases": list(r.phrases), "steps": list(r.steps)}
            for r in self._items.values()
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
        except OSError as exc:
            log.warning("could not write routines: %s", exc)

    def add(self, routine: Routine) -> None:
        with self._lock:
            self._items[routine.name] = routine
            self._save()

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._items:
                return False
            del self._items[name]
            self._save()
            return True

    def all(self) -> list[Routine]:
        with self._lock:
            return list(self._items.values())

    def match(self, text: str) -> Routine | None:
        """The routine whose name or phrase the text says, optionally after run/start/do."""
        s = normalize(text).strip().lower()
        if not s:
            return None
        stripped = _TRIGGER_PREFIX.sub("", s).strip()
        with self._lock:
            for routine in self._items.values():
                candidates = {routine.name.lower(), *(p.lower() for p in routine.phrases)}
                if s in candidates or stripped in candidates:
                    return routine
        return None
