"""Timers and reminders: pure Python, no GTK, no host access.

A reminder is a small record (id, due, text, kind) kept in a JSON file next to
the action log, so it survives a restart. ``parse_when`` turns the spoken
phrasings ("in 10 minutes", "at 3pm", "tomorrow at 9") into datetimes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

from voxa.agent.actionlog import default_path as action_default_path

log = logging.getLogger(__name__)

__all__ = ["Reminder", "ReminderStore", "parse_when", "default_path", "reminder_phrase", "timer_phrase"]


def default_path() -> Path:
    """Where reminders.json lives: next to the action log."""
    return action_default_path().parent / "reminders.json"


_IN_DURATION = re.compile(r"^in\s+(.+)$", re.IGNORECASE)
_AT_TIME = re.compile(r"^(?:(tomorrow|tonight)\s+)?at\s+(.+)$", re.IGNORECASE)
_UNIT = re.compile(r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)\s*(minutes?|hours?|days?)", re.IGNORECASE)
_NUM = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")


def _duration_seconds(text: str) -> int | None:
    """Seconds for "10 minutes", "an hour", "2 hours and 15 minutes"; None if junk."""
    total = 0
    found = False
    for match in _UNIT.finditer(text):
        raw, unit = match.group(1).lower(), match.group(2).lower()
        count = int(raw) if raw.isdigit() else _NUM.get(raw)
        if count is None:
            return None
        if unit.startswith("min"):
            total += count * 60
        elif unit.startswith("hour"):
            total += count * 3600
        else:
            total += count * 86400
        found = True
    return total if found else None


def parse_when(text: str, now: datetime) -> datetime | None:
    """Turn a spoken when-phrase into a datetime, or None when it is not one."""
    s = text.strip().strip(".!?,;").lower()
    if not s:
        return None

    duration = _IN_DURATION.match(s)
    if duration:
        seconds = _duration_seconds(duration.group(1))
        if seconds is None:
            return None
        return now + timedelta(seconds=seconds)

    at_match = _AT_TIME.match(s)
    if at_match:
        dayword, clock_text = at_match.group(1), at_match.group(2)
    else:
        dayword = None
        clock_text = s  # a bare "3pm" also counts
    clock = _CLOCK.match(clock_text)
    if not clock:
        return None
    hour = int(clock.group(1))
    minute = int(clock.group(2) or 0)
    ampm = (clock.group(3) or "").lower()
    if minute > 59 or hour > 23:
        return None
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif dayword == "tonight" and hour <= 12:
        hour += 12

    if dayword == "tomorrow":
        base = now + timedelta(days=1)
        return datetime.combine(base.date(), time(hour, minute))
    target = datetime.combine(now.date(), time(hour, minute))
    while target <= now:
        target += timedelta(days=1)
    return target


@dataclass(frozen=True, slots=True)
class Reminder:
    """One timer or reminder waiting for its moment."""

    id: str
    due: datetime
    text: str
    kind: str = "reminder"


def reminder_phrase(text: str) -> str:
    """What Voxa says when a reminder comes due."""
    return f"Reminder: {text}."


def timer_phrase(duration: str) -> str:
    """What Voxa says when a timer set for ``duration`` comes due."""
    label = duration.replace(" minutes", " minute").replace(" hours", " hour").replace(" days", " day")
    return f"Your {label} timer is done."


class ReminderStore:
    """The reminders.json file: survives a restart, never raises on save."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._lock = threading.Lock()
        self._items: dict[str, Reminder] = {}
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
                self._items[str(entry["id"])] = Reminder(
                    id=str(entry["id"]),
                    due=datetime.fromisoformat(str(entry["due"])),
                    text=str(entry["text"]),
                    kind=str(entry.get("kind", "reminder")),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _save(self) -> None:
        data = [
            {
                "id": r.id,
                "due": r.due.isoformat(timespec="seconds"),
                "text": r.text,
                "kind": r.kind,
            }
            for r in self._items.values()
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
        except OSError as exc:
            log.warning("could not write reminders: %s", exc)

    def add(self, reminder: Reminder) -> None:
        with self._lock:
            self._items[reminder.id] = reminder
            self._save()

    def remove(self, reminder_id: str) -> bool:
        with self._lock:
            if reminder_id not in self._items:
                return False
            del self._items[reminder_id]
            self._save()
            return True

    def due(self, now: datetime) -> list[Reminder]:
        """Reminders whose time has come; they are removed from the store."""
        with self._lock:
            ready = sorted((r for r in self._items.values() if r.due <= now), key=lambda r: r.due)
            if not ready:
                return []
            for r in ready:
                del self._items[r.id]
            self._save()
            return ready

    def upcoming(self) -> list[Reminder]:
        with self._lock:
            return sorted(self._items.values(), key=lambda r: r.due)
