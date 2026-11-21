"""Routines: one phrase runs several commands, and a schedule can run them too. Pure Python, no GTK, no host.

A routine is a small record (name, phrases, steps, and optionally a weekday and a time) kept
in ``routines.json`` next to the action log, so it survives a restart. Each step is a spoken
command string that goes through the normal router/planner, so anything Voxa can do can be a
step. ``parse_create`` turns the spoken phrasings ("when I say <phrase>, <step> and <step>"
and "create a routine called <name> that <step> and <step>") into a routine; ``match`` finds
a routine from a phrase or its name. ``parse_schedule`` and ``parse_unschedule`` set and clear
its schedule ("put the movie time routine on thursday at 4pm"), and ``due`` returns the routines
whose weekday and time say now; the window runs those through the same path a spoken trigger
uses, so the steps run in order and are logged like any other command.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from voxa.agent.actionlog import default_path as action_default_path
from voxa.agent.calendar_local import parse_time
from voxa.agent.dates import WEEKDAY_NAMES, WEEKDAYS
from voxa.agent.hearing import normalize
from voxa.agent.planner import COMMAND_VERBS

log = logging.getLogger(__name__)

__all__ = ["Routine", "RoutineStore", "create_reply", "default_path", "delete_reply", "list_reply", "parse_create", "parse_delete", "parse_list", "parse_schedule", "parse_unschedule", "schedule_reply", "schedule_text", "unschedule_reply"]


def default_path() -> Path:
    """Where routines.json lives: next to the action log."""
    return action_default_path().parent / "routines.json"


@dataclass(frozen=True, slots=True)
class Routine:
    """One routine: a name, the phrases that trigger it, its steps, and an optional schedule."""

    name: str
    phrases: tuple[str, ...]
    steps: tuple[str, ...]
    # ``weekday`` is 0 = Monday .. 6 = Sunday, None when the routine is not on a schedule;
    # ``at`` is (hour, minute), None for "that weekday, any time".
    weekday: int | None = None
    at: tuple[int, int] | None = None


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


_WEEKDAY_WORD = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
# The when-part both forms end with: a weekday word, then an optional spoken time.
_WHEN_PART = rf"(?P<when>(?P<day>{_WEEKDAY_WORD})(?:\s+at\s+(?P<time>.+?))?)\s*[.!?,;]*$"
# "schedule/put the <name> routine for|on|at|every <weekday> at <time>", and the same without
# the word "routine"; the weekday words and the spoken hours come from calendar_local.
_SCHEDULE_ROUTINE = re.compile(
    rf"^(?:schedule|reschedule|put|set)\s+(?:the |my |a |this )?(?P<name>.+?)\s+routine\s+(?:for|on|at|to|every)\s+{_WHEN_PART}",
    re.IGNORECASE,
)
_SCHEDULE_BARE = re.compile(
    rf"^(?:schedule|reschedule)\s+(?:the |my |a |this )?(?P<name>.+?)\s+(?:for|on|at|to|every)\s+{_WHEN_PART}",
    re.IGNORECASE,
)
# The issue's own line: "every friday at five, send me my weekly spending".
_EVERY_ROUTINE = re.compile(
    rf"^every\s+(?P<day>{_WEEKDAY_WORD})(?:\s+at\s+(?P<time>[^,.]+))?\s*[,.]\s*(?P<name>.+)$",
    re.IGNORECASE,
)
# "stop the <name> routine running ..." only counts when it says running; plain "stop the
# <name> routine" is not a scheduling phrase.
_UNSCHEDULE_STOP = re.compile(
    r"^(?:stop|cancel|clear)\s+(?:the |my |a |this )?(?P<name>.+?)\s+routine\s+running\b.*$",
    re.IGNORECASE,
)
_UNSCHEDULE_WORD = re.compile(
    r"^unschedule\s+(?:the |my |a |this )?(?P<name>.+?)(?:\s+routine)?\s*[.!?,;]*$",
    re.IGNORECASE,
)


def _when_pair(day: str, time_text: str | None) -> tuple[int, int | None] | None:
    """(weekday, (hour, minute) or None) from a weekday word and an optional spoken time.

    A weekday with no time means "that weekday, any time"; an unparsable time is not a schedule.
    """
    weekday = WEEKDAYS.get(day.lower())
    if weekday is None:
        return None
    if not (time_text or "").strip():
        return (weekday, None)
    at = parse_time(time_text)
    return (weekday, at) if at is not None else None


def parse_schedule(text: str) -> tuple[str, int | None, tuple[int, int] | None] | None:
    """(routine name, weekday, (hour, minute)) from a spoken schedule, or None when it is not one.

    Covers "schedule the movie time routine for friday at seven", "put the morning routine on
    thursday at 4pm" and the issue's own "every friday at five, send me my weekly spending". HH:MM,
    a bare hour and spoken hours ("five", "noon", "midnight") all work; a weekday with no time
    means that weekday, any time, which comes back as None for the time.
    """
    s = normalize(text).strip()
    for pattern in (_SCHEDULE_ROUTINE, _SCHEDULE_BARE, _EVERY_ROUTINE):
        match = pattern.match(s)
        if match is None:
            continue
        name = (match.group("name") or "").strip().strip(".!?,;")
        when = _when_pair(match.group("day"), match.group("time"))
        if not name or when is None:
            continue
        return name, when[0], when[1]
    return None


def parse_unschedule(text: str) -> str | None:
    """The routine name in "unschedule the <name> routine" or "stop the <name> routine running every friday"."""
    s = normalize(text).strip()
    for pattern in (_UNSCHEDULE_STOP, _UNSCHEDULE_WORD):
        match = pattern.match(s)
        if match is None:
            continue
        name = (match.group("name") or "").strip().strip(".!?,;")
        if name:
            return name
    return None


def create_reply(name: str, steps: list[str]) -> str:
    """The spoken confirmation after creating a routine."""
    joined = ", then ".join(steps)
    return f"Okay. When you say \u201c{name}\u201c I'll {joined}."


def list_reply(items: list) -> str:
    """The spoken reply to "list my routines"; each item is a Routine or a plain name.

    A routine on a schedule says so: "movie time \u2014 every friday 17:00".
    """
    if not items:
        return "No routines yet."
    return "Routines: " + ", ".join(_spoken_item(item) for item in items)


def _spoken_item(item) -> str:
    """One entry of the routine list: the name, plus its schedule when it has one."""
    name = getattr(item, "name", None)
    if name is None:
        return str(item)
    when = schedule_text(item)
    return f"{name} \u2014 {when}" if when else name


def schedule_text(routine: Routine) -> str:
    """A routine's schedule as words: "every friday 17:00", "every friday" for a weekday with
    no time, "" when the routine has no schedule."""
    if routine.weekday is None:
        return ""
    when = f"every {WEEKDAY_NAMES[routine.weekday].lower()}"
    if routine.at is not None:
        when += f" {routine.at[0]:02d}:{routine.at[1]:02d}"
    return when


def schedule_reply(name: str, weekday: int | None, at: tuple[int, int] | None) -> str:
    """The spoken confirmation after scheduling a routine; no weekday means nothing was scheduled."""
    if weekday is None:
        return unschedule_reply(name, False)
    when = f"every {WEEKDAY_NAMES[weekday].lower()}"
    if at is not None:
        when += f" at {at[0]:02d}:{at[1]:02d}"
    return f"Scheduled the {name} routine for {when}."


def unschedule_reply(name: str, ok: bool) -> str:
    """The spoken reply to "unschedule the <name> routine"."""
    return f"Stopped the {name} routine from running on its schedule." if ok else f"The {name} routine has no schedule."


def _read_weekday(value: object) -> int | None:
    """A stored weekday (0 = Monday .. 6 = Sunday), or None when it is absent or out of range."""
    return value if isinstance(value, int) and 0 <= value <= 6 else None


def _read_at(value: object) -> tuple[int, int] | None:
    """(hour, minute) from a stored schedule, or None when it is absent or out of range."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    hour, minute = value
    if not isinstance(hour, int) or not isinstance(minute, int):
        return None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return (hour, minute)


_DUE_WINDOW_MINUTES = 1  # the window's clock ticks every 5 s; a minute of drift still counts


def _at_is_due(at: tuple[int, int] | None, now: datetime) -> bool:
    """True when the scheduled minute has arrived (within a minute of drift), or no time was given."""
    if at is None:
        return True
    scheduled = at[0] * 60 + at[1]
    return 0 <= (now.hour * 60 + now.minute) - scheduled <= _DUE_WINDOW_MINUTES


def delete_reply(name: str, ok: bool) -> str:
    """The spoken reply to "delete the <name> routine"."""
    return f"Deleted the {name} routine." if ok else f"There is no {name} routine."


class RoutineStore:
    """The routines.json file: survives a restart, never raises on save.

    A file written before routines had schedules (name, phrases and steps only) still loads: the
    missing keys come back as "not on a schedule".
    """

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
                    weekday=_read_weekday(entry.get("weekday")),
                    at=_read_at(entry.get("at")),
                )
            except (KeyError, TypeError):
                continue

    def _save(self) -> None:
        data = [
            {
                "name": r.name,
                "phrases": list(r.phrases),
                "steps": list(r.steps),
                "weekday": r.weekday,
                "at": list(r.at) if r.at is not None else None,
            }
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

    def find(self, text: str) -> Routine | None:
        """The routine a schedule phrasing points at.

        Exact name or phrase first (``match``), then the most specific routine whose name the
        text contains, so "send me my weekly spending" finds a routine called "weekly spending".
        """
        exact = self.match(text)
        if exact is not None:
            return exact
        s = normalize(text).strip().lower()
        if not s:
            return None
        best: Routine | None = None
        best_len = 0
        for routine in self.all():
            for candidate in (routine.name.lower(), *(p.lower() for p in routine.phrases)):
                if candidate and (candidate in s or s in candidate) and len(candidate) > best_len:
                    best, best_len = routine, len(candidate)
        return best

    def set_schedule(self, name: str, weekday: int | None, at: tuple[int, int] | None) -> bool:
        """Put the named routine on that schedule; False when there is no such routine."""
        with self._lock:
            routine = self._items.get(name)
            if routine is None:
                return False
            self._items[name] = replace(routine, weekday=weekday, at=at)
            self._save()
            return True

    def clear_schedule(self, name: str) -> bool:
        """Take the named routine off its schedule; False when it has none to clear."""
        with self._lock:
            routine = self._items.get(name)
            if routine is None or routine.weekday is None:
                return False
            self._items[name] = replace(routine, weekday=None, at=None)
            self._save()
            return True

    def due(self, now: datetime) -> list[Routine]:
        """Routines whose schedule says now: the weekday matches and the time has come.

        A routine with no schedule is never due, and a weekday with no time is due all day. Unlike
        reminders these stay in the store, so the routine runs again next week; the window remembers
        what it already fired so the 5-second clock does not run it a dozen times in one minute.
        """
        with self._lock:
            ready = [
                routine
                for routine in self._items.values()
                if routine.weekday is not None and routine.weekday == now.weekday() and _at_is_due(routine.at, now)
            ]
        return sorted(ready, key=lambda routine: (routine.at if routine.at is not None else (0, 0), routine.name))

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
