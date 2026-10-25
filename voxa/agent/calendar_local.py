from __future__ import annotations

import difflib
import re
from datetime import date, datetime, timedelta

from voxa.agent.dates import MONTH_NAMES, WEEKDAY_NAMES, WEEKDAYS, _ordinal, parse_day

FILE_NAME = "Voxa Calendar.md"

_LINE = re.compile(r"-\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}|--:--)\s*\|\s*(.+?)\s*$")

_WEEKLY = re.compile(
    r"-\s*every\s+(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s+(?P<time>\d{2}:\d{2}|--:--)\s*\|\s*(?P<title>.+?)\s*$",
    re.IGNORECASE,
)

_WORD_HOUR = {
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
    "eleven": 11,
    "twelve": 12,
}

_H12 = r"(1[0-2]|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_H24 = r"(2[0-3]|1\d|[1-9])"


def _hour_value(raw: str) -> int:
    return int(raw) if raw.isdigit() else _WORD_HOUR[raw]


def _fix(hour: int, minute: int, ap: str | None, part: str | None) -> tuple[int, int] | None:
    if ap == "pm" and hour < 12:
        hour += 12
    if ap == "am" and hour == 12:
        hour = 0
    if part == "morning" and hour == 12:
        hour = 0
    if part in ("afternoon", "evening") and hour < 12:
        hour += 12
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return (hour, minute)
    return None


def parse_time(spoken: str) -> tuple[int, int] | None:
    """(hour, minute) from a spoken time; None when absent or unparsable."""
    s = " ".join(spoken.lower().strip().split())
    if not s:
        return None
    if s in ("noon", "midday", "twelve noon"):
        return (12, 0)
    if s in ("midnight", "twelve midnight"):
        return (0, 0)
    m = re.fullmatch(rf"(?P<h>{_H12}|{_H24})(?::(?P<m>[0-5]\d))?\s*(?P<ap>am|pm)?", s)
    if m:
        return _fix(_hour_value(m.group("h")), int(m.group("m") or 0), m.group("ap"), None)
    m = re.fullmatch(rf"(?P<h>{_H12}|{_H24}) (?P<m>[0-5]\d) (?P<ap>am|pm)", s)
    if m:
        return _fix(_hour_value(m.group("h")), int(m.group("m")), m.group("ap"), None)
    m = re.fullmatch(rf"half past (?P<h>{_H12})(?: (?P<ap>am|pm))?", s)
    if m:
        return _fix(_hour_value(m.group("h")), 30, m.group("ap"), None)
    m = re.fullmatch(
        rf"(?P<h>{_H12}|{_H24})(?::(?P<m>[0-5]\d))?"
        r" (?:in the (?P<part1>morning|afternoon|evening)|tonight|this evening)",
        s,
    )
    if m:
        part = m.group("part1") or "evening"
        return _fix(_hour_value(m.group("h")), int(m.group("m") or 0), None, part)
    return None


def split_when(spoken: str, today: date) -> tuple[date, tuple[int, int] | None] | None:
    """(day, time) from "tomorrow at noon"; a missing year means the next such date."""
    s = " ".join(spoken.lower().strip().split())
    if not s:
        return None
    day_part, time_part = s, None
    m = re.fullmatch(r"(.+?) at (.+)", s)
    if m:
        time_part = parse_time(m.group(2))
        if time_part is not None:
            day_part = m.group(1)
    parsed = parse_day(day_part, today)
    if parsed is None:
        return None
    year, month, day = parsed
    if year is None:
        when = date(today.year, month, day)
        if when < today:
            when = date(today.year + 1, month, day)
    else:
        when = date(year, month, day)
    return (when, time_part)


def _entry_key(m: re.Match) -> tuple[date, tuple[int, int]]:
    day = date.fromisoformat(m.group(1))
    if m.group(2) == "--:--":
        return (day, (99, 99))
    return (day, (int(m.group(2)[:2]), int(m.group(2)[3:])))


def add(content: str, day: date, time: tuple[int, int] | None, title: str) -> str:
    """One line per appointment, kept sorted by day then time; "--:--" when no time."""
    clean = title.strip()
    if clean:
        clean = clean[0].upper() + clean[1:]
    time_text = f"{time[0]:02d}:{time[1]:02d}" if time else "--:--"
    new_line = f"- {day:%Y-%m-%d} {time_text} | {clean}"
    new_key = (day, time if time else (99, 99))
    lines = content.splitlines()
    entry_positions = [i for i, ln in enumerate(lines) if _LINE.fullmatch(ln)]
    if not entry_positions:
        lines.append(new_line)
    else:
        placed = False
        for i in entry_positions:
            if _entry_key(_LINE.fullmatch(lines[i])) > new_key:
                lines.insert(i, new_line)
                placed = True
                break
        if not placed:
            lines.insert(entry_positions[-1] + 1, new_line)
    return "\n".join(lines) + "\n" if lines else ""


def add_weekly(content: str, weekday: int, time: tuple[int, int] | None, title: str) -> str:
    """Insert a recurring line: "- every Monday 09:00 | Team call"."""
    clean = title.strip()
    if clean:
        clean = clean[0].upper() + clean[1:]
    time_text = f"{time[0]:02d}:{time[1]:02d}" if time else "--:--"
    new_line = f"- every {WEEKDAY_NAMES[weekday]} {time_text} | {clean}"
    lines = content.splitlines()
    positions = [i for i, ln in enumerate(lines) if _LINE.fullmatch(ln) or _WEEKLY.fullmatch(ln)]
    if not positions:
        lines.append(new_line)
    else:
        lines.insert(positions[-1] + 1, new_line)
    return "\n".join(lines) + "\n" if lines else ""


def weekly_entries(content: str) -> list[tuple[int, tuple[int, int] | None, str]]:
    """(weekday index, time or None, title) per recurring line."""
    out: list[tuple[int, tuple[int, int] | None, str]] = []
    for line in content.splitlines():
        m = _WEEKLY.fullmatch(line)
        if not m:
            continue
        if m.group("time") == "--:--":
            time = None
        else:
            time = (int(m.group("time")[:2]), int(m.group("time")[3:]))
            if time[0] > 23 or time[1] > 59:
                continue
        out.append((WEEKDAYS[m.group("day").lower()], time, m.group("title")))
    return out


def entries_on(content: str, day: date) -> list[tuple[date, tuple[int, int] | None, str]]:
    """Dated entries for that day plus recurring entries whose weekday matches."""
    out = [(d, t, title) for d, t, title in entries(content) if d == day]
    for weekday, time, title in weekly_entries(content):
        if weekday == day.weekday():
            out.append((day, time, title))
    return out


def conflicts(items: list[tuple[date, tuple[int, int] | None, str]], day: date, time: tuple[int, int], minutes: int = 60) -> list[str]:
    """Titles of entries on that day whose time is within ``minutes`` of the new one."""
    if time is None:
        return []
    out: list[str] = []
    for item_day, item_time, title in items:
        if item_day != day or item_time is None:
            continue
        delta = abs((item_time[0] * 60 + item_time[1]) - (time[0] * 60 + time[1]))
        if delta <= minutes:
            out.append(title)
    return out


def entries(content: str) -> list[tuple[date, tuple[int, int] | None, str]]:
    """(day, time or None, title) per valid line."""
    out: list[tuple[date, tuple[int, int] | None, str]] = []
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        if not m:
            continue
        try:
            day = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if m.group(2) == "--:--":
            time = None
        else:
            time = (int(m.group(2)[:2]), int(m.group(2)[3:]))
            if time[0] > 23 or time[1] > 59:
                continue
        out.append((day, time, m.group(3)))
    return out


def remove(content: str, title: str) -> tuple[str, str | None, date | None, tuple[int, int] | None]:
    """By title, exact then contains, then fuzzy; (new content, removed title or None, day, time).

    Weekly lines match too; their day and time come back as None (they have no dated start).
    """
    target = " ".join(title.lower().strip().split())
    for prefix in ("my ", "the "):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    for suffix in (" appointment", " meeting"):
        if target.endswith(suffix):
            target = target[: -len(suffix)]
            break

    def match(ln: str) -> re.Match | None:
        return _LINE.fullmatch(ln) or _WEEKLY.fullmatch(ln)

    def title_of(m: re.Match) -> str:
        return m.groupdict().get("title") or m.group(3)

    def when(m: re.Match) -> tuple[date | None, tuple[int, int] | None]:
        if "day" in m.groupdict():  # weekly line
            return (None, None)
        day = date.fromisoformat(m.group(1))
        if m.group(2) == "--:--":
            return (day, None)
        return (day, (int(m.group(2)[:2]), int(m.group(2)[3:])))

    lines = content.splitlines()
    for want_exact in (True, False):
        for i, ln in enumerate(lines):
            m = match(ln)
            if not m:
                continue
            low = title_of(m).lower()
            hit = low == target if want_exact else bool(target) and target in low
            if hit:
                kept = lines[:i] + lines[i + 1 :]
                day, time = when(m)
                return ("\n".join(kept) + "\n" if kept else "", title_of(m), day, time)
    best_i, best_m, best_ratio = None, None, 0.0
    for i, ln in enumerate(lines):
        m = match(ln)
        if not m:
            continue
        low = title_of(m).lower()
        ratio = difflib.SequenceMatcher(None, target, low).ratio()
        if ratio > best_ratio:
            best_i, best_m, best_ratio = i, m, ratio
    if best_i is not None and best_ratio >= 0.6:
        kept = lines[:best_i] + lines[best_i + 1 :]
        day, time = when(best_m)
        return ("\n".join(kept) + "\n" if kept else "", title_of(best_m), day, time)
    return (content, None, None, None)


def on_day(items: list[tuple[date, tuple[int, int] | None, str]], day: date) -> list:
    """The appointments that fall on the given day."""
    return [item for item in items if item[0] == day]


def spoken_time(time: tuple[int, int] | None) -> str:
    """"noon", "3 PM", "3:30 PM", "9 AM"; "" when there is no time."""
    if time is None:
        return ""
    hour, minute = time
    if (hour, minute) == (12, 0):
        return "noon"
    if (hour, minute) == (0, 0):
        return "midnight"
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    if minute == 0:
        return f"{hour12} {suffix}"
    return f"{hour12}:{minute:02d} {suffix}"


def _day_word(day: date, today: date) -> str:
    """"Today", "Tomorrow", "On Friday" (within six days) or "On October 16th"."""
    delta = (day - today).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    if -6 <= delta <= 6:
        return f"On {WEEKDAY_NAMES[day.weekday()]}"
    return f"On {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}"


def spoken_day(items: list[tuple[date, tuple[int, int] | None, str]], day: date, today: date) -> str:
    """One spoken sentence for the appointments on a day."""
    word = _day_word(day, today)
    hits = on_day(items, day)
    if not hits:
        return f"You have nothing on your calendar {word.lower()}."
    parts = [f"{title} at {spoken_time(time)}" if time else title for _, time, title in hits]
    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    noun = "thing" if len(parts) == 1 else "things"
    return f"{word} you have {len(parts)} {noun}: {joined}."


def person_of(title: str) -> str | None:
    """The person a title belongs to: "Emma's soccer practice" or "Emma: soccer practice" -> "Emma"."""
    stripped = title.strip()
    m = re.match(r"([A-Za-z]+)'s\b", stripped)
    if m:
        return m.group(1)
    m = re.match(r"([A-Za-z]+):", stripped)
    return m.group(1) if m else None


def week(items_or_content, today: date) -> list[tuple[date, tuple[int, int] | None, str]]:
    """(day, time, title) for today..today+6 from dated and recurring entries, sorted."""
    items = entries(items_or_content) if isinstance(items_or_content, str) else list(items_or_content)
    if isinstance(items_or_content, str):
        for weekday, time, title in weekly_entries(items_or_content):
            for offset in range(7):
                probe = today + timedelta(days=offset)
                if probe.weekday() == weekday:
                    items.append((probe, time, title))
    end = today + timedelta(days=6)
    out = [item for item in items if today <= item[0] <= end]
    out.sort(key=lambda item: (item[0], item[1] if item[1] else (99, 99)))
    return out


def spoken_week(entries_list: list, today: date, person: str | None = None) -> str:
    """One spoken sentence for the week ahead, optionally filtered to one person."""
    items = entries_list
    if person:
        want = person.strip().lower()
        items = [item for item in items if (person_of(item[2]) or "").lower() == want]
    if not items:
        if person:
            return f"{person.strip().capitalize()} has nothing this week."
        return "Nothing is on the calendar this week."
    days: list[date] = []
    for day, _time, _title in items:
        if day not in days:
            days.append(day)
    segments = []
    for day in days:
        delta = (day - today).days
        word = "today" if delta == 0 else "tomorrow" if delta == 1 else WEEKDAY_NAMES[day.weekday()]
        parts = [f"{title} at {spoken_time(time)}" if time else title for d, time, title in items if d == day]
        segments.append(f"{word}, {', '.join(parts)}.")
    if person:
        noun = "thing" if len(items) == 1 else "things"
        head = f"{person.strip().capitalize()} has {len(items)} {noun} this week:"
    else:
        head = "This week:"
    return f"{head} " + " ".join(segments)


def spoken_count(entries_list: list) -> str:
    """How many appointments fall this week and which day is busiest."""
    if not entries_list:
        return "You have nothing this week."
    counts: dict[date, int] = {}
    for day, _time, _title in entries_list:
        counts[day] = counts.get(day, 0) + 1
    busiest = max(counts, key=lambda d: counts[d])
    noun = "appointment" if len(entries_list) == 1 else "appointments"
    return f"You have {len(entries_list)} {noun} this week, the busiest day is {WEEKDAY_NAMES[busiest.weekday()]}."


def spoken_next(content: str, now: datetime) -> str:
    """One spoken sentence for the next appointment (dated or recurring), or the empty announcement."""
    today = now.date()
    items: list[tuple[date, tuple[int, int] | None, str]] = []
    for day in range(8):
        probe = today + timedelta(days=day)
        items.extend(entries_on(content, probe))
    upcoming = sorted(
        (item for item in items if item[0] >= today),
        key=lambda item: (item[0], item[1] if item[1] else (99, 99)),
    )
    if not upcoming:
        return "You have nothing coming up."
    day, time, title = upcoming[0]
    when = _day_word(day, today).lower()
    if time:
        return f"Your next appointment is {title}, {when} at {spoken_time(time)}."
    return f"Your next appointment is {title}, {when}."
