"""Restaurant table reservations: what was said, which page to open, which fields to fill.

This is the first slice of crhy/voxa#47. It parses the spoken booking, opens the
restaurant's booking page, and types the details into the fields that page offers.
It never submits: pressing Book/Confirm is the spend rule's call (crhy/voxa#43), and
the nearest-alternatives rule, the phone fallback (#42) and the calendar half stay
open. Nothing here touches the network.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date, timedelta

from voxa.agent import dates
from voxa.agent.calendar_local import parse_time

__all__ = [
    "Reservation",
    "booking_url",
    "describe_reservation",
    "form_values",
    "missing_details",
    "parse_reservation",
    "spoken_site",
]


@dataclass(frozen=True, slots=True)
class Reservation:
    """A spoken table booking; the empty fields are the ones to ask the user for."""

    restaurant: str
    party_size: int | None = None
    when: date | None = None
    at: str = ""
    site: str = ""


_SITES = {
    "opentable": ("OpenTable", "opentable.com"),
    "resy": ("Resy", "resy.com"),
    "tock": ("Tock", "tock.us"),
}
_SITE_KEYS = {
    "opentable": "opentable",
    "opentable.com": "opentable",
    "open table": "opentable",
    "resy": "resy",
    "resy.com": "resy",
    "tock": "tock",
    "tock.us": "tock",
}
_SITE = re.compile(
    r"\b(?:on|via|through)\s+(open table|opentable\.com|opentable|resy\.com|resy|tock\.us|tock)\b",
    re.IGNORECASE,
)

# A booking verb is required; sentences that clearly belong to another intent (a room,
# a ticket, a car, a calendar entry) are left to that intent.
_BOOK = re.compile(
    r"\b(?:book|books|booking|booked|reserve|reserves|reserving|reserved|reservation|reservations)\b",
    re.IGNORECASE,
)
_NOT_A_TABLE = re.compile(
    r"\b(?:hotel|motel|resort|hostel|bnb|flight|flights|airfare|ticket|tickets|room|rooms|cabin|"
    r"cruise|rental|car|cars|truck|vehicle|scooter|bicycle|calendar|appointment|dentist|"
    r"oil change|tow|repair|spoon|bed and breakfast)\b",
    re.IGNORECASE,
)

_WEEK_WORDS = "|".join(sorted(dates.WEEKDAYS, key=len, reverse=True))
_MONTH_WORDS = "|".join(sorted(dates.MONTHS, key=len, reverse=True))
_ORD_WORDS = "|".join(sorted(dates.ORDINALS, key=len, reverse=True))
_COUNT_WORDS = "|".join(sorted((w for w in dates.NUMBER_WORDS if w not in ("a", "an")), key=len, reverse=True))
_PARTY_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty one": 21, "twenty two": 22, "twenty three": 23,
    "twenty four": 24, "twenty five": 25, "thirty": 30,
}
_PARTY_ALT = "|".join(sorted(_PARTY_WORDS, key=len, reverse=True))
_PARTY_NUMBER = rf"(?:{_PARTY_ALT}|\d{{1,2}})(?!\d)(?!\s*(?:am|pm|o'clock))"
_PARTY = re.compile(
    rf"\b(?:for|of|with|party(?:\s+of)?|group(?:\s+of)?)\s+(?P<n>{_PARTY_NUMBER})"
    rf"(?:\s+(?:people|persons?|guests?|diners?|adults?|kids?|of us))?\b"
    rf"|\b(?P<n2>{_PARTY_NUMBER})\s+(?:people|persons?|guests?|diners?|adults?|kids?|of us)\b",
    re.IGNORECASE,
)

_DAY = re.compile(
    rf"\b(?:(?:on|for|by|until|starting|leaving|showing up|going)\s+)?"
    rf"(?:"
    rf"(?:next|this|coming)\s+(?:the\s+)?(?:{_WEEK_WORDS})"
    r"|today|tonight|tomorrow"
    rf"|in (?:{_COUNT_WORDS}|\d+) (?:day|days|week|weeks)"
    rf"|(?:{_WEEK_WORDS})"
    rf"|(?:{_MONTH_WORDS}) (?:\d{{1,2}}(?:st|nd|rd|th)?|{_ORD_WORDS})(?: \d{{4}})?"
    rf"|(?:\d{{1,2}}(?:st|nd|rd|th)?|{_ORD_WORDS}) (?:of\s+)?(?:{_MONTH_WORDS})(?: \d{{4}})?"
    rf"|the (?:{_ORD_WORDS}|\d{{1,2}}(?:st|nd|rd|th)?)"
    r"|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    rf")",
    re.IGNORECASE,
)
_DAY_PREFIXES = ("on ", "for ", "by ", "until ", "starting ", "leaving ", "showing up ", "going ")

_AT = re.compile(r"\b(?:at|by|around)\s+", re.IGNORECASE)
_MARKED_TIME = re.compile(
    r"\b(?:am|pm|a\.m\.|p\.m\.|o'clock|morning|afternoon|evening|tonight|noon|midnight)\b",
    re.IGNORECASE,
)
_AT_PREP = re.compile(r"\bat\s+(.+)$", re.IGNORECASE)
_IN_PREP = re.compile(r"\b(?:in|inside)\s+(.+)$", re.IGNORECASE)
_NAME_STOP = re.compile(r"\s+(?:and|with|for|on|at|in|near|by|around|next to|across from)\b.*$", re.IGNORECASE)
_FILLER_WORDS = (
    "at", "in", "inside", "the", "a", "an", "some", "my", "our", "their", "his", "her",
    "new", "local", "nearby", "favorite", "favourite", "best", "good", "nice", "great",
    "little", "big", "other", "own", "table", "tables", "booth", "booths", "seat", "seats",
    "seating", "spot", "spots", "place", "places", "room", "rooms", "reservation",
    "reservations", "booking", "bookings", "restaurant", "restaurants", "dinner", "lunch",
    "brunch", "breakfast", "book", "books", "reserve", "reserving", "make", "order",
    "open", "opens", "get", "gets", "have", "want", "need", "try", "visit", "go", "fill",
    "pre", "online", "site", "web", "website", "page", "pages", "search", "look", "find",
    "show", "see", "for", "on", "to", "with",
)
_FILLER = re.compile(rf"^(?:{'|'.join(_FILLER_WORDS)})\s+", re.IGNORECASE)
_FILLER_ONLY = re.compile(rf"^(?:{'|'.join(_FILLER_WORDS)})(?:\s+(?:{'|'.join(_FILLER_WORDS)}))*$", re.IGNORECASE)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip(" ,.;:").strip()


def _party(text: str) -> tuple[int | None, str]:
    m = _PARTY.search(text)
    if not m:
        return None, text
    raw = (m.group("n") or m.group("n2") or "").lower()
    size = int(raw) if raw.isdigit() else _PARTY_WORDS.get(raw)
    if size is None or not 1 <= size <= 99:
        return None, text
    return size, _clean(text[: m.start()] + " " + text[m.end():])


def _day(text: str, today: date) -> tuple[date | None, str]:
    """The day phrase, taken out of the sentence; None when it cannot be resolved."""
    m = _DAY.search(text)
    if not m:
        return None, text
    phrase = " ".join(m.group(0).lower().split())
    for prefix in _DAY_PREFIXES:
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):]
            break
    rest = _clean(text[: m.start()] + " " + text[m.end():])
    if phrase == "today" or phrase == "tonight":
        return today, rest
    if phrase == "tomorrow":
        return today + timedelta(days=1), rest
    found = dates.parse_day(phrase, today)
    if found is not None:
        year, month, day = found
        if year:
            return date(year, month, day), rest
        return dates.next_occurrence(f"{month:02d}-{day:02d}", today), rest
    m_num = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", phrase)
    if m_num:
        month, day = int(m_num.group(1)), int(m_num.group(2))
        year = int(m_num.group(3)) if m_num.group(3) else None
        if year is not None and year < 100:
            year += 2000
        try:
            if year:
                return date(year, month, day), rest
            return dates.next_occurrence(f"{month:02d}-{day:02d}", today), rest
        except ValueError:
            return None, rest
    return None, rest


def _time(text: str) -> tuple[str, str]:
    """The time of day as "7:00 PM", taken out of the sentence; "" when it said none."""
    for m in _AT.finditer(text):
        tail = text[m.end():]
        words = tail.split()
        for take in (3, 2, 1):
            if take > len(words):
                continue
            cand = " ".join(words[:take])
            parsed = parse_time(cand)
            if parsed is None:
                continue
            hour, minute = parsed
            if not _MARKED_TIME.search(cand) and hour < 12:
                # A bare hour at a restaurant is dinner service: "at seven" is 7 PM.
                hour += 12
            end = m.end() + len(cand)
            spoken = f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"
            return spoken, _clean(text[: m.start()] + " " + text[end:])
    return "", text


def _restaurant(text: str) -> str:
    """The restaurant's name, or "" when the sentence never named one."""
    m = _AT_PREP.search(text) or _IN_PREP.search(text)
    if m:
        name = m.group(1)
        stop = _NAME_STOP.search(name)
        if stop:
            name = name[: stop.start()]
    else:
        name = text
    name = " ".join(name.strip(" ,.;:").split())
    name = re.sub(r"\s+(?:on|at|in|for|with|near|by|around)$", "", name, flags=re.IGNORECASE).strip()
    changed = True
    while changed:
        changed = False
        stripped = _FILLER.sub("", name, count=1)
        if stripped != name:
            name = stripped.strip()
            changed = True
    if len(name.split()) > 6:
        name = " ".join(name.split()[:6])
    if not re.search(r"[a-z0-9]", name, re.IGNORECASE) or _FILLER_ONLY.fullmatch(name):
        return ""
    return name


def parse_reservation(text: str, today: date) -> Reservation | None:
    """Parse a spoken table booking; None when it is not one or names no restaurant."""
    s = " ".join((text or "").strip().split())
    s = re.sub(r"[?!]", "", s).strip(" .,;:").strip()
    if not s or not _BOOK.search(s) or _NOT_A_TABLE.search(s):
        return None

    rest = s
    site = ""
    m = _SITE.search(rest)
    if m:
        site = _SITE_KEYS.get(m.group(1).lower(), "")
        rest = _clean(rest[: m.start()] + " " + rest[m.end():])

    party, rest = _party(rest)
    at, rest = _time(rest)
    when, rest = _day(rest, today)
    restaurant = _restaurant(rest)
    if not restaurant:
        return None
    return Reservation(restaurant=restaurant, party_size=party, when=when, at=at, site=site)


def spoken_site(res: Reservation) -> str:
    """The booking site the user named, or the web in general."""
    return _SITES.get(res.site, ("the web", ""))[0]


def booking_url(res: Reservation, location: str = "") -> str:
    """The page to open: a search for the restaurant's booking page, narrowed to the named site."""
    if not res.restaurant:
        return ""
    bits = [res.restaurant]
    if location:
        bits.append(location)
    bits.append("reservation")
    _label, domain = _SITES.get(res.site, ("", ""))
    if domain:
        bits.append(f"site:{domain}")
    return f"https://duckduckgo.com/?q={urllib.parse.quote(' '.join(bits))}"


def form_values(res: Reservation) -> list[tuple[str, str]]:
    """The labelled values a booking page expects, in the order a diner would read them."""
    out: list[tuple[str, str]] = []
    if res.party_size is not None:
        out.append(("Party size", str(res.party_size)))
    if res.when:
        out.append(("Date", res.when.isoformat()))
    if res.at:
        out.append(("Time", res.at))
    if res.restaurant:
        out.append(("Restaurant", res.restaurant))
    return out


def missing_details(res: Reservation) -> list[str]:
    """What the sentence did not say, in the order it would be asked for."""
    out: list[str] = []
    if res.party_size is None:
        out.append("the party size")
    if res.when is None:
        out.append("the day")
    if not res.at:
        out.append("the time")
    return out


def describe_reservation(res: Reservation, today: date) -> str:
    """Speak the reservation back: "A table for 4 at Luigi's on Friday at 7:00 PM." """
    text = f"A table for {res.party_size}" if res.party_size is not None else "A table"
    text += f" at {res.restaurant}"
    if res.when:
        text += f" {dates.spoken_date(res.when, today)}"
    if res.at:
        text += f" at {res.at}"
    if res.site:
        text += f" on {spoken_site(res)}"
    return f"{text}."
