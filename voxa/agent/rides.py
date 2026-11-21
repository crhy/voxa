"""Read a ride quote for crhy/voxa#50 ("Order a ride (Lyft / Uber / taxi)").

This slice is only the first step of #50's plan: *open the provider with the
destination pre-filled and read the quote back*. It never requests, schedules or
cancels a ride, and it makes no announcement about a driver, car or the car
being close -- those halves of #50 stay behind the undecided #43 spend rule.

``parse_ride_request`` turns a spoken ride request into a :class:`RideRequest`;
``provider_url`` builds the provider's request page with the pickup and
destination pre-filled (or a search that names them); ``describe_ride`` is the
spoken sentence; ``quote_from_text`` reads the price estimate and the arrival or
ETA wording back out of the provider's own page text.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date, timedelta

__all__ = [
    "RideRequest",
    "parse_ride_request",
    "provider_url",
    "describe_ride",
    "quote_from_text",
]


@dataclass(frozen=True, slots=True)
class RideRequest:
    provider: str
    pickup: str
    destination: str
    when: date | None
    at: str


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

# A ride request must name one of these; anything else is another intent.
_PROVIDER = (
    (re.compile(r"\blyft\b", re.IGNORECASE), "Lyft"),
    (re.compile(r"\buber(?:\s+x)?\b|\bubers\b", re.IGNORECASE), "Uber"),
    (re.compile(r"\btaxi(?:es)?\b|\bcabs?\b|\bcabbies?\b", re.IGNORECASE), "taxi"),
)

# Pick-up phrasings that mean the stored home location.
_HOME_WORDS = frozenset({"home", "my home", "my place", "here", "my location"})

_CLOCK = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))\b", re.IGNORECASE
)
_WHEN_WORD = re.compile(
    r"\b(next\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|this\s+(?:morning|afternoon|evening)|tomorrow|tonight|today"
    r"|morning|afternoon|evening|noon|midnight)\b",
    re.IGNORECASE,
)

_PRICE = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*[-\u2013]\s*\$\s?\d[\d,]*(?:\.\d{1,2})?)?"
)
_ARRIVE_AT = re.compile(r"\b(?:arrives?|arrive)\s+at\s+(\d{1,2}:\d{2}\s*[ap]\.?m)", re.IGNORECASE)
_ETA_LABEL = re.compile(r"\bETA[:\s]+(\d+\s*(?:min|mins|minutes))", re.IGNORECASE)
_MIN_AWAY = re.compile(
    r"\b(\d+)\s*(?:min|mins|minutes)\b(?:\s+(?:away|out|to\s+go|to\s+arrival|to\s+pickup|until\s+arrival))?",
    re.IGNORECASE,
)


def _date_word(word: str, today: date) -> date | None:
    w = re.sub(r"\s+", " ", word.lower()).strip()
    if w in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return today
    if w == "tomorrow":
        return today + timedelta(days=1)
    if w == "next week":
        return today + timedelta(days=7)
    if w.startswith("next "):
        target = _WEEKDAYS.get(w[5:])
        if target is not None:
            found = today + timedelta(days=1)
            while found.weekday() != target:
                found += timedelta(days=1)
            return found
    return None


def parse_ride_request(text: str, today: date, home: str) -> RideRequest | None:
    """Parse a spoken ride request, or None when it belongs to another intent."""
    s = re.sub(r"\s+", " ", re.sub(r"[?!]", "", text.strip()))
    if not s:
        return None

    provider = ""
    for pattern, name in _PROVIDER:
        if pattern.search(s):
            provider = name
            break
    if not provider:
        return None  # no Lyft/Uber/taxi -> a rental car, car for sale, flight, hotel, table...

    # Collect the clock and the calendar word before removing them, keeping order.
    found_time: list[tuple[int, int, str]] = []
    for rx in (_CLOCK, _WHEN_WORD):
        for m in rx.finditer(s):
            found_time.append((m.start(), m.end(), re.sub(r"\s+", " ", m.group(1)).strip()))
    found_time.sort()
    kept: list[tuple[int, int, str]] = []
    end = -1
    when: date | None = None
    for start, stop, word in found_time:
        if start < end:
            continue
        kept.append((start, stop, word))
        end = stop
        when = _date_word(word, today) or when
    at = " ".join(word for _, _, word in kept)
    for start, stop, _ in reversed(kept):
        s = (s[:start] + " " + s[stop:]).strip()
    # Drop a dangling preposition left behind by a removed time ("... station at").
    s = re.sub(r"\s+(?:at|for|in|by|around|near)\b\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Pick-up: "from <place>" / "from home"; destination: everything after "to".
    pickup = ""
    m = re.search(r"\bfrom\s+(.+?)\s+to\b", s, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        pickup = home if raw.lower() in _HOME_WORDS else raw
    dest = ""
    m = re.search(r"\bto\s+(.+)$", s, re.IGNORECASE)
    if m:
        dest = m.group(1).strip()
    if not dest:
        return None
    dest = re.sub(r"^(?:a|an|the)\s+", "", dest).strip()

    return RideRequest(provider=provider, pickup=pickup, destination=dest, when=when, at=at)


def provider_url(req: RideRequest, pickup: str) -> str:
    """The provider's request page with pick-up and destination pre-filled.

    Falls back to a search that names the provider and both places when there is
    no known deep-link form. The exact form is pinned by tests, so a provider
    that changes its link is a one-line fix here.
    """
    origin = req.pickup or pickup
    if req.provider == "Lyft":
        return (
            "https://www.lyft.com/ride/"
            f"{urllib.parse.quote(origin)}/{urllib.parse.quote(req.destination)}"
        )
    query = f"{req.provider} ride from {origin} to {req.destination}"
    return f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"


def describe_ride(req: RideRequest) -> str:
    """The spoken sentence for the request, without ordering anything."""
    pickup = req.pickup or "your location"
    lead = {"Lyft": "a Lyft", "Uber": "an Uber", "taxi": "a taxi"}[req.provider]
    text = f"Here's {lead} from {pickup} to {req.destination}"
    if req.at:
        text += f" at {req.at}"
    if req.when:
        text += f", on {req.when.strftime('%B %d')}"
    return f"{text}."


def quote_from_text(page_text: str) -> tuple[str | None, str | None]:
    """Return ``(price, eta)`` from the provider's own text, ``None`` when absent."""
    m = _PRICE.search(page_text)
    price = m.group(0).strip() if m else None

    eta: str | None = None
    m = _ARRIVE_AT.search(page_text)
    if m:
        wording = re.sub(r"\s+", " ", m.group(1)).strip()
        eta = f"arrives at {wording}"
    else:
        m = _ETA_LABEL.search(page_text)
        if m:
            eta = m.group(1).strip()
        else:
            m = _MIN_AWAY.search(page_text)
            if m:
                eta = m.group(0).strip()
    return price, eta
