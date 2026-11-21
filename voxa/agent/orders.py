"""Orders and deliveries (issue #52's first step).

Pure logic: tracking numbers out of text the user pasted or selected, the local
order list, and the wording used when the list is asked about. No I/O: the tools
layer passes the ledger content in and gets content back.
"""

from __future__ import annotations

import html
import re
from datetime import date, timedelta

from voxa.agent import lookups

FILE_NAME = "Voxa Orders.md"

COLUMNS = ("number", "carrier", "note", "status", "expected date", "last seen")

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Text that only looks like a tracking number: links, phone numbers, dates, times.
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE = re.compile(
    r"\+?\d{1,3}[\s.-]\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
    r"|\+?\d{3}[\s.-]\d{4}[\s.-]\d{4}"
    r"|\+?\(?\d{3}\)?[\s.-]\d{4}[\s.-]\d{4}",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\(?(?:19|20)?\d{2})?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE)

# Carrier shapes, most specific first. A 12-digit run is FedEx, a 10 or 11 digit
# run is DHL, 16 hex-ish characters are USPS, and 1Z starts a UPS number.
_CARRIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("UPS", re.compile(r"\b1Z[A-Z0-9]{10,}\b", re.IGNORECASE)),
    ("USPS", re.compile(r"\b[0-9A-F]{16}\b", re.IGNORECASE)),
    ("FedEx", re.compile(r"\b\d{12}\b")),
    ("DHL", re.compile(r"\b\d{10,11}\b")),
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")

# Status wording a carrier page may state, most specific first.
_STATUS_PHRASES: tuple[tuple[str, str], ...] = (
    ("delivered", "delivery completed"),
    ("delivered", "delivered"),
    ("out for delivery", "out for delivery"),
    ("out for delivery", "on the truck to your address"),
    ("delayed", "delayed"),
    ("delayed", "lost"),
    ("carrier problem", "no results found"),
    ("carrier problem", "unable to track"),
    ("carrier problem", "not valid"),
    ("carrier problem", "not supported"),
    ("in transit", "in transit"),
    ("in transit", "in network"),
    ("in transit", "en route"),
    ("in transit", "pickup"),
    ("scheduled", "scheduled"),
    ("waiting to ship", "awaiting"),
)

_NOTE_HINTS = re.compile(
    r"(?:order|shipment|package|parcel|delivery)\s+(?:for|of)\s+(?P<note>[^\n,.;]{2,60})"
    r"|(?:item|product|description)\s*[:=-]\s*(?P<note2>[^\n,.;]{2,60})",
    re.IGNORECASE,
)
_DELIVERY_WORDS = re.compile(
    r"\b(?:expected|arrival|arrives?|arriving|due|deliver(?:y|ed)?|ship(?:ping|s)?)\b",
    re.IGNORECASE,
)
_ISO_DAY = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")
_SLASH_DAY = re.compile(r"\b(?P<m>\d{1,2})[/-](?P<d>\d{1,2})[/-](?P<y>\d{2,4})\b")
_MONTH_DAY = re.compile(
    r"\b(?P<mname>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?),?\s+"
    r"(?P<d>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<y>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH = re.compile(
    r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<mname>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)(?:,?\s+(?P<y>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)


def _mask(text: str) -> str:
    """Blank out links, phone numbers, dates and times, keeping every offset."""
    masked = text
    for pattern in (_URL, _PHONE, _DATE, _TIME):
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def _token_is_number(token: str) -> bool:
    if not any(char.isdigit() for char in token):
        return False
    if not any(char.isalpha() for char in token) and len(token) < 10:
        return False
    return True


def parse_tracking_numbers(text: str) -> list[tuple[str, str]]:
    """Return (carrier, number) pairs found in pasted or selected text.

    Carrier shapes are tried first; anything left that is a bare token goes through
    lookups.tracking_number and is reported as carrier "unknown". Dates, phone
    numbers, links, already matched spans and duplicates are skipped.
    """
    masked = _mask(text)
    found: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for carrier, pattern in _CARRIER_PATTERNS:
        for match in pattern.finditer(masked):
            if any(match.start() < end and match.end() > start for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            found.append((match.start(), carrier, match.group(0).upper()))
    for match in _TOKEN.finditer(masked):
        span = (match.start(), match.end())
        if any(span[0] < end and span[1] > start for start, end in taken):
            continue
        token = match.group(0)
        if not _token_is_number(token):
            continue
        number = lookups.tracking_number(token)
        if number is None:
            continue
        taken.append(span)
        found.append((match.start(), "unknown", number))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _start, carrier, number in sorted(found, key=lambda item: item[0]):
        if number in seen:
            continue
        seen.add(number)
        out.append((carrier, number))
    return out


_NOTE_STOP = frozenset({
    "a", "an", "the", "your", "you", "yours", "my", "our", "i", "we", "us",
    "is", "was", "has", "have", "had", "will", "would", "can", "be", "been",
    "for", "to", "of", "and", "or", "in", "on", "at", "from", "that", "this",
    "it", "order", "orders", "item", "package", "parcel", "shipment",
    "ship", "ships", "shipped", "send", "sent", "arrive", "arrives", "arrived",
    "due", "today", "tomorrow", "next", "week", "number", "note", "details",
})


_NOTE_FILLER = frozenset({
    "my", "the", "this", "that", "these", "those", "next", "a", "an", "your",
    "yours", "is", "where", "what", "order", "orders", "please", "me", "i", "s",
})


def note_key(spoken: str) -> str:
    """Reduce a spoken "where is my X order" to the words worth matching on."""
    words = [
        word
        for word in " ".join(spoken.strip().lower().split()).split()
        if word not in _NOTE_FILLER
    ]
    return " ".join(words)


def note_from_text(text: str) -> str:
    """What the order is for, when the pasted text says so; "" when it does not."""
    match = _NOTE_HINTS.search(text)
    if match is None:
        return ""
    raw = (match.group("note") or match.group("note2") or "").strip(" -–—\"'“”")
    words: list[str] = []
    for word in raw.split():
        if word.lower().strip(".,;:") in _NOTE_STOP:
            break
        words.append(word)
        if len(words) == 4:
            break
    if not words:
        return ""
    note = " ".join(words)
    return note[0].upper() + note[1:]


def _month_number(spoken: str) -> int:
    return _MONTH_ABBR.get(spoken[:3].lower(), 0)


def _iso(month: int, day: int, year: str) -> str:
    today = date.today()
    if not month or not 1 <= day <= 31:
        return ""
    number = today.year
    if year:
        digits = int(year)
        number = (2000 + digits if digits < 70 else 1900 + digits) if len(year) == 2 else digits
    try:
        found = date(number, month, day)
    except ValueError:
        return ""
    if not year and found < today:
        found = date(found.year + 1, month, day)
    return found.isoformat()


def expected_from_text(text: str) -> str:
    """The expected delivery date in ISO form, when the text states one near delivery wording."""
    for pattern in (_ISO_DAY, _SLASH_DAY, _MONTH_DAY, _DAY_MONTH):
        for match in pattern.finditer(text):
            before = text[max(0, match.start() - 30):match.start()]
            if not _DELIVERY_WORDS.search(before):
                continue
            groups = match.groupdict()
            year = groups.get("y") or ""
            if pattern is _MONTH_DAY or pattern is _DAY_MONTH:
                iso = _iso(_month_number(groups["mname"]), int(groups["d"]), year)
            else:
                iso = _iso(int(groups["m"]), int(groups["d"]), year)
            if iso:
                return iso
    return ""


def _page_text(page: str) -> str:
    without_code = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
    text = re.sub(r"(?s)<[^>]+>", " ", without_code)
    text = html.unescape(text)
    return " ".join(text.split())


def status_from_page(page: str, number: str = "") -> str:
    """The status a carrier page states, or "" when it states none we understand.

    Never guesses: an unreadable page, or a page that says nothing about the
    parcel, returns "".
    """
    text = _page_text(page)
    if not text:
        return ""
    window = text
    if number:
        index = text.casefold().find(number.casefold())
        if index >= 0:
            window = text[index:index + 600]
    low = window.casefold()
    for status, phrase in _STATUS_PHRASES:
        if phrase in low:
            return status
    return ""


def parse(content: str) -> list[dict[str, str]]:
    """The order rows in a ledger file's content."""
    rows: list[dict[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            continue
        if not cells[0] or cells[0].casefold() == COLUMNS[0] or set(cells[0]) <= {"-"}:
            continue
        rows.append({key: ("" if value == "-" else value) for key, value in zip(COLUMNS, cells, strict=True)})
    return rows


def render(rows: list[dict[str, str]]) -> str:
    """The ledger file's content for these rows."""

    def cell(value: str) -> str:
        return " ".join(value.replace("|", "/").split()) or "-"

    lines = ["# Voxa Orders", ""]
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(column, "")) for column in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def add_or_update(
    rows: list[dict[str, str]],
    number: str,
    carrier: str = "",
    note: str = "",
    status: str = "",
    expected: str = "",
    today: str = "",
) -> tuple[list[dict[str, str]], list[str]]:
    """Add or update one order; returns the new rows and a sentence per status change."""
    day = today or date.today().isoformat()
    wanted = number.strip().upper()
    if not wanted:
        return ([dict(row) for row in rows], [])
    changes: list[str] = []
    updated: list[dict[str, str]] = []
    found = False
    for row in rows:
        copy = dict(row)
        if copy["number"].upper() == wanted:
            old = copy.get("status", "")
            new = status.strip() or old
            if status.strip() and old and status.strip().casefold() != old.casefold():
                changes.append(f"Tracking {wanted} changed from {old} to {status.strip()}.")
            copy["status"] = new
            copy["carrier"] = carrier.strip() or copy.get("carrier", "")
            copy["note"] = note.strip() or copy.get("note", "")
            copy["expected date"] = expected.strip() or copy.get("expected date", "")
            copy["last seen"] = day
            found = True
        updated.append(copy)
    if not found:
        updated.append({
            "number": wanted,
            "carrier": carrier.strip(),
            "note": note.strip(),
            "status": status.strip(),
            "expected date": expected.strip(),
            "last seen": day,
        })
    return updated, changes


def _spoken_day(iso: str) -> str:
    try:
        day = date.fromisoformat(iso)
    except ValueError:
        return ""
    return f"{_MONTHS[day.month - 1]} {day.day}"


def _spoken_row(row: dict[str, str]) -> str:
    number = row.get("number", "")
    carrier = row.get("carrier", "")
    note = row.get("note", "")
    status = row.get("status", "")
    expected = _spoken_day(row.get("expected date", ""))
    label = f"your {note.lower()} order" if note else "the order"
    head = f"{label}, tracking {number}"
    if carrier and carrier != "unknown":
        head += f" with {carrier}"
    if not status:
        return f"{head} has no status read from its carrier page yet"
    sentence = f"{head} is {status}"
    if expected:
        sentence += f", due {expected}"
    return sentence


def _sentence(text: str) -> str:
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def spoken_for_note(rows: list[dict[str, str]], note: str) -> str:
    """Answer "where is my <note> order?" from the list, without touching the network."""
    if not rows:
        return "Nothing is on your order list."
    wanted = note_key(note)
    if wanted:
        matches = [
            row for row in rows
            if wanted in row.get("note", "").lower() or wanted in row.get("number", "").lower()
        ]
        if not matches:
            return f"I have no {wanted} order on your order list."
    else:
        matches = rows
    return " ".join(_sentence(_spoken_row(row)) for row in matches)


def spoken_arriving(rows: list[dict[str, str]], when: str, today: date | None = None) -> str:
    """Answer "what's arriving this week?" from the list, without touching the network."""
    if not rows:
        return "Nothing is on your order list."
    base = today or date.today()
    word = " ".join(when.lower().split())
    if "today" in word:
        start, end, label = base, base, "today"
    elif "tomorrow" in word:
        start, end, label = base + timedelta(days=1), base + timedelta(days=1), "tomorrow"
    elif "week" in word:
        monday = base - timedelta(days=base.weekday())
        start, end, label = monday, monday + timedelta(days=6), "this week"
    elif "month" in word:
        start, end, label = base, base + timedelta(days=30), "this month"
    elif "soon" in word:
        start, end, label = base, base + timedelta(days=7), "soon"
    else:
        start, end, label = base, base + timedelta(days=7), "soon"
    due: list[tuple[date, dict[str, str]]] = []
    for row in rows:
        try:
            day = date.fromisoformat(row.get("expected date", ""))
        except ValueError:
            continue
        if start <= day <= end:
            due.append((day, row))
    if not due:
        upcoming = sorted(
            (
                (date.fromisoformat(row.get("expected date", "")), row)
                for row in rows
                if _is_iso(row.get("expected date", ""))
            ),
            key=lambda item: item[0],
        )
        if not upcoming:
            count = len(rows)
            if count == 1:
                return f"Nothing is due {label}, and your order has no expected date."
            return (
                f"Nothing is due {label}, and none of your {count} "
                "orders has an expected date."
            )
        day, row = next((item for item in upcoming if item[0] >= base), upcoming[0])
        label_next = f"your {row['note'].lower()} order" if row.get("note") else "the next order"
        head = f"{label_next}, tracking {row['number']}"
        if row.get("carrier") and row["carrier"] != "unknown":
            head += f" with {row['carrier']}"
        return f"Nothing is due {label}. {_sentence(head + f' is due {_spoken_day(day.isoformat())}')}"
    due.sort(key=lambda item: (item[0], item[1]["number"]))
    count = len(due)
    lead = f"one order is due {label}" if count == 1 else f"{count} orders are due {label}"
    return _sentence(lead + ": " + "; ".join(_spoken_row(row) for _day, row in due))


def _is_iso(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def spoken_changes(rows: list[dict[str, str]]) -> str:
    """Read back where every order on the list stands."""
    if not rows:
        return "Nothing is on your order list."
    return " ".join(_sentence(_spoken_row(row)) for row in rows)
