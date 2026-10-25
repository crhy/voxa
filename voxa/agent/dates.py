from __future__ import annotations

import re
from datetime import date, timedelta

FILE_NAME = "Voxa Dates.md"

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

NUMBER_WORDS = {
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
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "thirty": 30,
}

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_WEEKDAY = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_NUMWORD = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))

_ORD1 = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth",
}
_TENS = {20: "twenty", 30: "thirty"}
ORDINALS: dict[str, int] = {w: n for n, w in _ORD1.items()}
for _t, _tw in _TENS.items():
    ORDINALS[_tw + "ieth"] = _t
    for _n in range(1, 10):
        ORDINALS[_tw + " " + _ORD1[_n]] = _t + _n
_ORD = "|".join(sorted(ORDINALS, key=len, reverse=True))


def _next_day_of_month(day: int, today: date) -> tuple[int | None, int, int] | None:
    """Next date on or after today whose day-of-month is day; (year, month, day)."""
    year, month = today.year, today.month
    if day < today.day:
        month += 1
        if month == 13:
            month, year = 1, year + 1
    for _ in range(24):
        if _valid(year, month, day):
            return (year, month, day)
        month += 1
        if month == 13:
            month, year = 1, year + 1
    return None

_LINE = re.compile(r"-\s*(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}|\d{1,2})\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$")


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        suffix = "th"
    elif day % 10 == 1:
        suffix = "st"
    elif day % 10 == 2:
        suffix = "nd"
    elif day % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{day}{suffix}"


def _valid(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
        return True
    except ValueError:
        return False


def _clamped(year: int, month: int, day: int) -> date:
    """date(year, month, day), or the month's last day when the day does not exist."""
    try:
        return date(year, month, day)
    except ValueError:
        next_month = date(year + (month == 12), month % 12 + 1, 1)
        return next_month - timedelta(days=1)


def _monthly_occurrence(day: int, today: date) -> date:
    """Monthly "DD": this month if not yet passed, else next month; missing days clamp to the last day."""
    occ = _clamped(today.year, today.month, day)
    if occ < today:
        year, month = today.year, today.month + 1
        if month == 13:
            year, month = year + 1, 1
        occ = _clamped(year, month, day)
    return occ


def parse_day(spoken: str, today: date) -> tuple[int | None, int, int] | None:
    """(year or None, month, day) from a spoken day; None when it is not a date."""
    s = " ".join(spoken.lower().strip().split())
    if not s:
        return None
    if s == "today":
        return (today.year, today.month, today.day)
    if s == "tomorrow":
        d = today + timedelta(days=1)
        return (d.year, d.month, d.day)
    m = re.fullmatch(rf"(?:(?:next|this) )?({_WEEKDAY})", s)
    if m:
        target = WEEKDAYS[m.group(1)]
        ahead = (target - today.weekday()) % 7
        if ahead == 0:
            ahead = 7
        d = today + timedelta(days=ahead)
        return (d.year, d.month, d.day)
    m = re.fullmatch(rf"in ({_NUMWORD}|\d+) (day|days|week|weeks)", s)
    if m:
        raw = m.group(1)
        count = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        if m.group(2).startswith("week"):
            count *= 7
        d = today + timedelta(days=count)
        return (d.year, d.month, d.day)
    m = re.fullmatch(rf"({_MONTH}) (\d{{1,2}})(?:st|nd|rd|th)?(?: (\d{{4}}))?", s)
    if not m:
        m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)? of ({_MONTH})(?: (\d{{4}}))?", s)
    if not m:
        m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)? ({_MONTH})(?: (\d{{4}}))?", s)
    if m:
        if m.group(1) in MONTHS:
            month = MONTHS[m.group(1)]
            day = int(m.group(2))
        else:
            day = int(m.group(1))
            month = MONTHS[m.group(2)]
        year = int(m.group(3)) if m.group(3) else None
        if _valid(year or 2020, month, day):
            return (year, month, day)
    s_ord = s.replace("-", " ")
    m = re.fullmatch(rf"({_MONTH}) ({_ORD})(?: (\d{{4}}))?", s_ord)
    if m:
        month = MONTHS[m.group(1)]
        day = ORDINALS[m.group(2)]
        year = int(m.group(3)) if m.group(3) else None
        if _valid(year or 2020, month, day):
            return (year, month, day)
    m = re.fullmatch(rf"(?:the )?({_ORD}) of ({_MONTH})(?: (\d{{4}}))?", s_ord)
    if m:
        month = MONTHS[m.group(2)]
        day = ORDINALS[m.group(1)]
        year = int(m.group(3)) if m.group(3) else None
        if _valid(year or 2020, month, day):
            return (year, month, day)
    m = re.fullmatch(rf"the ({_ORD})", s_ord)
    if m:
        return _next_day_of_month(ORDINALS[m.group(1)], today)
    m = re.fullmatch(r"the (\d{1,2})(?:st|nd|rd|th)?", s)
    if m:
        return _next_day_of_month(int(m.group(1)), today)
    return None


_MONTHLY = re.compile(
    rf"(?:on )?(?:the )?(?P<d1>\d{{1,2}}(?:st|nd|rd|th)?|{_ORD}) of (?:every|each) month"
    rf"|(?:every|each) month on (?:the )?(?P<d2>\d{{1,2}}(?:st|nd|rd|th)?|{_ORD})"
    rf"|monthly on (?:the )?(?P<d3>\d{{1,2}}(?:st|nd|rd|th)?|{_ORD})"
    rf"|on (?:the )?(?P<d4>\d{{1,2}}(?:st|nd|rd|th)?|{_ORD}) (?:every|each) month",
    re.IGNORECASE,
)


def parse_monthly(spoken: str) -> int | None:
    """Day-of-month from "the 15th of every month", "every month on the 15th", "monthly on the 3rd", "on the 15th each month"."""
    s = " ".join(spoken.lower().strip().split())
    m = _MONTHLY.fullmatch(s)
    if not m:
        return None
    raw = m.group("d1") or m.group("d2") or m.group("d3") or m.group("d4")
    digits = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", raw)
    if digits:
        day = int(digits.group(1))
    else:
        day = ORDINALS[raw.replace("-", " ")]
    return day if 1 <= day <= 31 else None


def entries(content: str) -> list[tuple[str, str, str]]:
    """(date text, kind, name) per valid line."""
    out: list[tuple[str, str, str]] = []
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        if not m:
            continue
        date_text, kind, name = m.group(1), m.group(2).lower(), m.group(3)
        if next_occurrence(date_text, date(2020, 1, 1)) is None:
            continue
        out.append((date_text, kind, name))
    return out


def add(content: str, kind: str, name: str, when: tuple[int | None, int, int]) -> str:
    """One line per (kind, name) pair, case-insensitive: adding again replaces the old line."""
    year, month, day = when
    if year is None and month is None:
        date_text = f"{day:02d}"
    elif year is None:
        date_text = f"{month:02d}-{day:02d}"
    else:
        date_text = f"{year:04d}-{month:02d}-{day:02d}"
    clean = name.strip()
    if clean:
        clean = clean[0].upper() + clean[1:]
    kind_clean = kind.strip().lower()
    lines: list[str] = []
    replaced = False
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        if m and m.group(2).lower() == kind_clean and m.group(3).lower() == clean.lower():
            if not replaced:
                lines.append(f"- {date_text} | {kind_clean} | {clean}")
                replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"- {date_text} | {kind_clean} | {clean}")
    return "\n".join(lines) + "\n" if lines else ""


def remove(content: str, name: str) -> tuple[str, str | None]:
    """By name, case-insensitive; every entry whose name contains the spoken name is dropped."""
    target = name.strip().lower()
    kept: list[str] = []
    removed: str | None = None
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        low = m.group(3).lower() if m else ""
        if m and (low == target or (target and target in low)):
            if removed is None:
                removed = m.group(3)
            continue
        kept.append(line)
    if removed is None:
        return (content, None)
    return ("\n".join(kept) + "\n" if kept else "", removed)


def next_occurrence(date_text: str, today: date) -> date | None:
    """Yearly "MM-DD" -> this year's date, or next year's if already past; full dates as they are."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date_text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{2})-(\d{2})", date_text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            candidate = date(today.year, month, day)
        except ValueError:
            return None
        if candidate < today:
            candidate = date(today.year + 1, month, day)
        return candidate
    m = re.fullmatch(r"(\d{1,2})", date_text)
    if m:
        day = int(m.group(1))
        if not 1 <= day <= 31:
            return None
        return _monthly_occurrence(day, today)
    return None


def find(content: str, name: str) -> tuple[str, str, str] | None:
    """Entry whose name equals or contains the spoken name (case-insensitive; a trailing "'s" is dropped)."""
    spoken = name.strip().lower()
    if spoken.endswith("'s"):
        spoken = spoken[:-2]
    for line in content.splitlines():
        m = _LINE.fullmatch(line)
        if not m:
            continue
        entry_name = m.group(3)
        low = entry_name.lower()
        if low == spoken or (spoken and spoken in low):
            return (m.group(1), m.group(2).lower(), entry_name)
    return None


def spoken_date(day: date, today: date) -> str:
    """Spoken form of a date: "today", "tomorrow", "on Friday" (within 6 days), else "on March 3rd"."""
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if 2 <= delta <= 6:
        return f"on {WEEKDAY_NAMES[day.weekday()]}"
    text = f"on {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}"
    if day.year != today.year:
        text += f" {day.year}"
    return text


def _spoken_date_short(day: date, today: date) -> str:
    """Like spoken_date but never mentions the year."""
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if 2 <= delta <= 6:
        return f"on {WEEKDAY_NAMES[day.weekday()]}"
    return f"on {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}"


def spoken_when(entry, today: date) -> str:
    """"Mom's birthday is on March 3rd, in 144 days." / "The report is due on Friday, in 6 days." / monthly bills and subscriptions."""
    date_text, kind, name = entry
    day = next_occurrence(date_text, today)
    if day is None:
        return ""
    monthly = re.fullmatch(r"\d{1,2}", date_text) is not None
    delta = (day - today).days
    if kind == "bill":
        if not name.lower().startswith("the "):
            name = f"The {name[:1].lower()}{name[1:]}"
        subject = f"{name} bill is due"
    elif kind == "subscription":
        if name.lower().startswith("your "):
            subject = f"{name} subscription renews"
        else:
            subject = f"Your {name[:1].upper()}{name[1:]} subscription renews"
    elif kind in ("service", "deadline"):
        if not name.lower().startswith("the "):
            name = f"The {name[:1].lower()}{name[1:]}"
        subject = f"{name} is due"
    else:
        subject = f"{name}'s {kind} is"
    if monthly:
        every = f"on the {_ordinal(int(date_text))} of each month"
        if delta == 0:
            return f"{subject} {every}, today."
        return f"{subject} {every}; next in {delta} days."
    if delta < 0:
        if kind in ("deadline", "bill", "service"):
            return f"{name} was due on {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}."
        if kind == "subscription":
            return f"Your {name} subscription renewed on {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}."
        return f"{name}'s {kind} was on {MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}."
    if kind in ("bill", "subscription", "service"):
        where = _spoken_date_short(day, today)
    else:
        where = spoken_date(day, today)
    if delta == 0:
        return f"{subject} today."
    if delta == 1:
        return f"{subject} tomorrow."
    return f"{subject} {where}, in {delta} days."


def upcoming(content: str, today: date, days: int = 30, kind: str | None = None) -> list[tuple[date, str, str]]:
    """(date, kind, name) within `days`, soonest first; passed one-off dates left out; optional kind filter."""
    wanted = kind.strip().lower() if kind else None
    found: list[tuple[date, str, str]] = []
    for date_text, entry_kind, name in entries(content):
        if wanted and entry_kind != wanted:
            continue
        day = next_occurrence(date_text, today)
        if day is None:
            continue
        delta = (day - today).days
        if delta < 0 or delta > days:
            continue
        found.append((day, entry_kind, name))
    found.sort(key=lambda item: item[0])
    return found


def spoken_upcoming(content: str, today: date, days: int = 30, kind: str | None = None) -> str:
    """One spoken line for everything coming up, or the empty announcement."""
    items = upcoming(content, today, days, kind)
    if not items:
        if kind:
            return f"Nothing is coming up for {kind}s in the next {days} days."
        return f"Nothing is coming up in the next {days} days."
    parts = []
    for day, entry_kind, name in items:
        when = spoken_date(day, today)
        if entry_kind == "deadline":
            parts.append(f"{name} due {when}")
        elif entry_kind == "bill":
            bill_name = name if name.lower().startswith("the ") else f"The {name[:1].lower()}{name[1:]}"
            parts.append(f"{bill_name} bill {when}")
        elif entry_kind == "subscription":
            if name.lower().startswith("your "):
                parts.append(f"{name} subscription {when}")
            else:
                parts.append(f"Your {name[:1].upper()}{name[1:]} subscription {when}")
        elif entry_kind == "service":
            svc_name = name if name.lower().startswith("the ") else f"The {name[:1].lower()}{name[1:]}"
            parts.append(f"{svc_name} {when}")
        else:
            parts.append(f"{name}'s {entry_kind} {when}")
    if len(parts) == 1:
        return f"Coming up: {parts[0]}."
    joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"Coming up: {joined}."
