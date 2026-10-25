from __future__ import annotations

import re
from datetime import date

from voxa.agent.dates import parse_day, spoken_date

FILE_NAME = "Voxa Travel Documents.md"

_LINE = re.compile(r"^-\s*(.+?):\s*expires\s+(\d{4}-\d{2}-\d{2})\s*$")


def _clean_name(spoken: str) -> str:
    name = " ".join(spoken.strip().split())
    if not name:
        return ""
    return name[0].upper() + name[1:]


def parse(content: str) -> list[tuple[str, str]]:
    """(document, ISO date) per valid line, in file order."""
    out: list[tuple[str, str]] = []
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        if not m:
            continue
        try:
            date.fromisoformat(m.group(2))
        except ValueError:
            continue
        out.append((m.group(1), m.group(2)))
    return out


def render(docs: list[tuple[str, str]]) -> str:
    return "".join(f"- {name}: expires {iso}\n" for name, iso in docs)


def parse_expiry(spoken: str, today: date) -> date | None:
    """A full calendar date from spoken text; None when it is not one."""
    s = " ".join(spoken.strip().split())
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    year, month, day = parse_day(s, today) or (None, None, None)
    if year is None or month is None or day is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def add(content: str, document: str, iso: str) -> str:
    """One line per document, case-insensitive: adding again replaces the old line."""
    name = _clean_name(document)
    docs = [d for d in parse(content) if d[0].lower() != name.lower()]
    docs.append((name, iso))
    return render(docs)


def when(content: str, document: str) -> str | None:
    """ISO expiry for a saved document, matched case-insensitively; None when absent."""
    target = " ".join(document.strip().split()).lower()
    for name, iso in parse(content):
        if name.lower() == target:
            return iso
    return None


def forget(content: str, document: str) -> tuple[str, str | None]:
    """Remove one document by name; (new content, removed name or None)."""
    target = " ".join(document.strip().split()).lower()
    kept: list[tuple[str, str]] = []
    removed: str | None = None
    for name, iso in parse(content):
        if removed is None and name.lower() == target:
            removed = name
            continue
        kept.append((name, iso))
    if removed is None:
        return (content, None)
    return (render(kept), removed)


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _soon(day: date, today: date) -> bool:
    delta = (day - today).days
    return 0 <= delta <= 183


def spoken_saved(document: str, iso: str, today: date) -> str:
    name = _clean_name(document)
    day = date.fromisoformat(iso)
    out = f"I noted that your {name.lower()} expires {spoken_date(day, today)}."
    if _soon(day, today):
        out += " That's soon, so renew it in time."
    return out


def spoken_when(document: str, iso: str, today: date) -> str:
    name = _clean_name(document)
    day = date.fromisoformat(iso)
    label = name.lower()
    if (day - today).days < 0:
        return f"Your {label} expired {spoken_date(day, today)}."
    out = f"Your {label} expires {spoken_date(day, today)}."
    if _soon(day, today):
        out += " That's soon, so renew it in time."
    return out


def spoken_list(docs: list[tuple[str, str]], today: date) -> str:
    if not docs:
        return "You have no travel documents saved."
    parts = [
        f"{name} expires {spoken_date(date.fromisoformat(iso), today)}"
        for name, iso in docs
    ]
    thing = "document" if len(docs) == 1 else "documents"
    return f"You have {len(docs)} travel {thing}: {_join(parts)}."
