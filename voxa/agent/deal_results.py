"""Turn a price or booking results page into the few entries worth saying aloud.

The page arrives as the collapsed visible text of a ``BrowserSession.text()`` call
(or as the joined ``text`` fields of a ``BrowserSession.outline()`` result). Nothing
here touches the network or the browser: it is a deterministic stdlib parse, so the
flight, hotel and product pages can be tested with canned text.
"""

from __future__ import annotations

import re

__all__ = ["parse_results", "spoken", "compact"]

# A price: a currency sign in front of an amount, or an amount followed by a
# currency word. Amounts may carry thousands separators and cents.
_PRICE = re.compile(
    r"(?P<symbol>[$€£¢])\s*(?P<amount>\d[\d,]*(?:[.,]\d{1,2}(?!\d))?)|"
    r"(?P<bare>\d[\d,]*)\s*(?:usd|dollars?|bucks)\b",
    re.IGNORECASE,
)

# Stops and durations, in the shapes the flight pages write them.
_STOP_COUNT = re.compile(r"\b(\d{1,2})\s*(?:stops?|stopovers?|layovers?)\b", re.IGNORECASE)
_NONSTOP = re.compile(r"\b(?:non[-\s]?stops?|direct|point[-\s]to[-\s]point)\b", re.IGNORECASE)
_HOURS = re.compile(r"\b(\d{1,2})\s*(?:h|hr|hrs|hour|hours)(?![a-z])\.?", re.IGNORECASE)
_MINUTES = re.compile(r"\b(\d{1,3})\s*(?:m|min|mins|minute|minutes)(?![a-z])\.?", re.IGNORECASE)

# Words that start a new row or a page heading; the text right before a price is
# cut at the last one of these so the previous row's tail cannot join the label.
_ROW_BREAK = re.compile(
    r"\b(?:depart(?:s|ing|ed)?|return(?:s|ing|ed)?|arriv(?:e|es|ed|ing)?"
    r"|checking in|per (?:night|nights|room|rooms|stay|day|days)|starting from"
    r"|prices?|cheapest|lowest|best (?:price|matches|value)|results?|listing"
    r"|sorted by|sort by|nonstop only|for sale|top \d+|filters?|sign in|log in|book now)\b",
    re.IGNORECASE,
)

# Dates, times and night counts a page repeats around its rows.
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b"
    r"|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:[ap]\.?\s?m)?\b", re.IGNORECASE)
_NIGHTS = re.compile(r"\bfor\s+\d+\s+nights?\b", re.IGNORECASE)

# A row's own tail ("via Chicago") can sit in front of the next label.
_CONNECTOR = re.compile(r"^(?:via|thru|through|plus|incl(?:ud|e)?)\b", re.IGNORECASE)
# A row's tail is lower case; the thing being priced starts with a capital.
_LOWER_TAIL = re.compile(r"^(?:[a-z'’.-]+ ){1,3}(?=[A-Z0-9])")

# Page furniture that is never a result: ads, sort controls, fare legends.
_CHROME = re.compile(
    r"(?:starting (?:from|at)|prices? (?:from|starting|only)|cheapest (?:price|fare|airfare)"
    r"|sorted by|sort by|sponsored|advertisement|sign in|log in|cookie|select your"
    r"|round ?trip|one ?way|nonstop (?:only|flights only)|book (?:now|flights?))",
    re.IGNORECASE,
)

# A bare number with no currency sign is read as dollars.
_CURRENCY_WORDS = {"$": "dollar", "¢": "cent", "€": "euro", "£": "pound", "": "dollar"}

_HEAD_CHARS = 120
_TAIL_CHARS = 60
_LABEL_CHARS = 80
_LABEL_WORDS = 8


def _price_of(match: re.Match) -> tuple[float, str]:
    raw = match.group("amount") or match.group("bare") or ""
    symbol = match.group("symbol") or ""
    if not raw:
        return 0.0, ""
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return 0.0, ""
    if symbol == "¢":
        value /= 100.0
    return value if value > 0 else 0.0, symbol


def _strip_connector(text: str) -> str:
    """Drop a leading "via <place>" tail of the previous row.

    At most two words after the connector belong to it; the label keeps the rest.
    """
    match = _CONNECTOR.match(text)
    if not match:
        return text
    words = text[match.end():].split()
    if not words:
        return ""
    return " ".join(words[min(2, len(words) - 1) :])


def _clean_label(fragment: str) -> str:
    """Strip the numbers and control words a result row carries around its label."""
    text = fragment
    cut = 0
    for match in _ROW_BREAK.finditer(text):
        cut = match.end()
    text = text[cut:]
    for pattern in (_HOURS, _MINUTES, _STOP_COUNT, _NONSTOP, _DATE, _TIME, _NIGHTS):
        text = pattern.sub(" ", text)
    text = re.sub(r"[$€£¢]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:|/-–—")
    words = text.split()
    if len(words) > _LABEL_WORDS:
        text = " ".join(words[-_LABEL_WORDS:])
    for _ in range(2):
        stripped = _strip_connector(text)
        if stripped == text:
            break
        text = stripped.strip()
    text = _LOWER_TAIL.sub("", text)
    if len(text) > _LABEL_CHARS:
        text = text[-_LABEL_CHARS:]
        text = text[text.find(" ") + 1 :]
    return text.strip()


def _stops_of(region: str) -> int | None:
    count = _STOP_COUNT.search(region)
    if count:
        return int(count.group(1))
    return 0 if _NONSTOP.search(region) else None


def _duration_of(region: str) -> int | None:
    hours = _HOURS.search(region)
    minutes = _MINUTES.search(region)
    if not hours and not minutes:
        return None
    total = (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)
    return total or None


def _unit_word(unit: str, count: int) -> str:
    return unit if count == 1 else f"{unit}s"


def _spoken_price(price: float, currency: str) -> str:
    unit = _CURRENCY_WORDS.get(currency, "dollar")
    whole = int(price)
    cents = round((price - whole) * 100)
    if whole == 0 and cents == 0:
        return "0 dollars"
    if whole == 0:
        return f"{cents} {_unit_word('cent', cents)}"
    if cents:
        return f"{whole} {_unit_word(unit, whole)} and {cents} cents"
    return f"{whole} {_unit_word(unit, whole)}"


def _spoken_stops(stops: int | None) -> str:
    if stops is None:
        return ""
    if stops == 0:
        return "nonstop"
    return "1 stop" if stops == 1 else f"{stops} stops"


def _spoken_duration(minutes: int | None) -> str:
    if minutes is None:
        return ""
    hours, rest = divmod(minutes, 60)
    if hours and rest:
        return f"{hours} hours {rest} minutes" if hours > 1 else f"1 hour {rest} minutes"
    if hours:
        return "1 hour" if hours == 1 else f"{hours} hours"
    return "1 minute" if rest == 1 else f"{rest} minutes"


def spoken(entry: dict) -> str:
    """One result as a plain-speech fragment: label, stops, duration, price."""
    parts = [entry.get("label", "")]
    if entry.get("stops") is not None:
        parts.append(_spoken_stops(entry["stops"]))
    if entry.get("duration") is not None:
        parts.append(_spoken_duration(entry["duration"]))
    parts.append(_spoken_price(entry["price"], entry.get("currency", "")))
    return ", ".join(part for part in parts if part)


def compact(entry: dict) -> str:
    """The same result written short, for ``ToolResult.detail``."""
    parts = [entry.get("label", "")]
    if entry.get("stops") is not None:
        parts.append(_spoken_stops(entry["stops"]))
    if entry.get("duration") is not None:
        minutes = entry["duration"]
        hours, rest = divmod(minutes, 60)
        parts.append(f"{hours} h {rest} m" if hours and rest else f"{hours} h" if hours else f"{rest} m")
    parts.append(f"{entry.get('currency', '')}{entry['price']:g}")
    return ", ".join(part for part in parts if part)


def parse_results(text: str, kind: str, limit: int = 3) -> list[dict]:
    """Return at most ``limit`` results, cheapest first.

    Flights carry label, price, duration (minutes) and stops; hotel, car and
    product pages carry label and price, so the same call serves #49 and #51.
    An empty list means nothing on the page parsed.
    """
    if limit <= 0:
        return []
    page = " ".join(str(text).split())
    if not page:
        return []
    entries: list[dict] = []
    seen: set[tuple[str, float]] = set()
    previous_end = 0
    for match in _PRICE.finditer(page):
        head = page[previous_end:match.start()]
        tail = page[match.end():match.end() + _TAIL_CHARS]
        previous_end = match.end()
        if len(head) > _HEAD_CHARS:
            head = head[-_HEAD_CHARS:]
        price, currency = _price_of(match)
        if price <= 0:
            continue
        label = _clean_label(head) or _clean_label(tail)
        if not label or _CHROME.search(label):
            continue
        key = (label, price)
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {
            "label": label,
            "price": price,
            "currency": currency,
            "duration": None,
            "stops": None,
        }
        if kind == "flight":
            region = f"{head} {tail}"
            entry["stops"] = _stops_of(region)
            entry["duration"] = _duration_of(region)
        entries.append(entry)
    entries.sort(key=lambda entry: entry["price"])
    return entries[:limit]
