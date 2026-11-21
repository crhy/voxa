"""Read a product results page back aloud (crhy/voxa#51, first step).

The page arrives as the collapsed visible text of a ``BrowserSession.text()`` call,
or as the joined ``text`` fields of a ``BrowserSession.outline()`` result. Nothing here
touches the browser or the network: it is a deterministic stdlib parse, so shopping
pages are tested with canned text.
"""

from __future__ import annotations

import re

from voxa.agent.deals import DealRequest, build_url

__all__ = ["parse_products", "spoken_products", "product_line", "parse_results", "spoken", "compact"]

# A price: a currency sign in front of an amount, or an amount followed by a
# currency word. Amounts may carry thousands separators and cents.
_PRICE = re.compile(
    r"(?P<symbol>[$€£¢])\s*(?P<amount>\d[\d,]*(?:[.,]\d{1,2})?)"
    r"|(?P<bare>\d[\d,]*(?:\.\d{1,2})?)\s*(?:usd|dollars?|bucks)\b",
    re.IGNORECASE,
)

# Ratings in the shapes shopping pages write them: "4.6 out of 5", "4.6 stars",
# "Rated 4.6", "4.6 (1,204 reviews)", "customer rating: 4.6".
_RATING = re.compile(
    r"\b(?P<num>\d\.\d)\s*(?:out of\s*5|/\s*5|stars?\b|★)"
    r"|\b(?P<rev>\d(?:\.\d)?)\s*\(?\s*\d[\d,]*\s*(?:reviews?|ratings?|comments?)\b"
    r"|\b(?:rated|rating|customer rating|user rating|average rating)\b[^\d]{0,14}(?P<word>\d(?:\.\d)?)",
    re.IGNORECASE,
)

# Who sells it, when the page says so: "from Amazon", "on eBay", "sold by Best Buy".
# One word after the marker, plus a second only where shop names really pair up.
_SELLER_TAIL = r"Buy|Market|Marketplace|Store|Shop|Express|Direct|World|Plus|Com|Mall|Outlet|Depot|Warehouse"
_SELLER = re.compile(
    r"\b(?:from|on|at|via|sold (?:by|at|on)|available (?:at|on)|seller(?:s)?|store(?:s)?)\s*[:\-]?\s+"
    rf"(?P<name>[A-Za-z][\w.'+-]*)(?:\s+(?P<tail>{_SELLER_TAIL}))?\b",
    re.IGNORECASE,
)
_SELLER_STOP = {"google", "shop", "store", "seller", "price", "prices"}

# Page furniture that sits between two rows, and never belongs to a label.
_ROW_BREAK = re.compile(
    r"\b(?:sponsored|advertisement|ads\b|shopping results?|results? for|best (?:price|matches|value)"
    r"|lowest price (?:for|on)|prices?(?: from| only| starting)?|starting from|sorted by|sort by"
    r"|filters?|sign in|log in|see (?:all|details|more|how)|add to (?:cart|list)"
    r"|compare|reviews?|in stock|out of stock|save for later|about (?:this|our)|page \d+|results?)\b",
    re.IGNORECASE,
)

# A row's own tail ("with 3 USB ports") can sit in front of the next label.
_LOWER_TAIL = re.compile(r"^(?:[a-z'’.-]+ ){1,4}(?=[A-Z0-9$])")

_HEAD_CHARS = 140
_TAIL_CHARS = 100
_LABEL_WORDS = 8
_LABEL_CHARS = 90
# How each currency sign a page uses should be said.
_SYMBOLS = {"": "$", "$": "$", "¢": "$", "€": "€", "£": "£"}


def _price_of(match: re.Match) -> tuple[float, str]:
    raw = match.group("amount") or match.group("bare") or ""
    symbol = match.group("symbol") or ""
    if not raw:
        return 0.0, ""
    if re.fullmatch(r"\d+,\d{2}", raw):
        raw = raw.replace(",", ".")
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return 0.0, ""
    if symbol == "¢":
        value /= 100.0
    return value if value > 0 else 0.0, symbol


def _rating_of(region: str) -> float:
    for match in _RATING.finditer(region):
        for group in ("num", "rev", "word"):
            raw = match.group(group)
            if not raw:
                continue
            value = float(raw)
            if 0.0 < value <= 5.0:
                return value
    return 0.0


def _seller_of(region: str) -> str:
    match = _SELLER.search(region)
    if not match:
        return ""
    name = match.group("name")
    tail = match.group("tail")
    if tail:
        name = f"{name} {tail}"
    name = name.strip(" ,.;:|/-–—")
    if not name or name.lower() in _SELLER_STOP:
        return ""
    return name


def _clean_label(fragment: str) -> str:
    """Strip the prices, ratings and control words a result row carries around its label."""
    text = fragment
    cut = 0
    # Both page furniture and a "from <shop>" tail end a row: cut at whichever of
    # them sits furthest right, so the next row's label cannot be dragged along.
    for pattern in (_ROW_BREAK, _SELLER):
        for match in pattern.finditer(text):
            cut = max(cut, match.end())
    text = text[cut:]
    text = _SELLER.sub(" ", text)
    text = _RATING.sub(" ", text)
    text = _PRICE.sub(" ", text)
    text = re.sub(r"[$€£¢]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:|/-–—")
    words = text.split()
    if len(words) > _LABEL_WORDS:
        text = " ".join(words[-_LABEL_WORDS:])
    for _ in range(2):
        stripped = _LOWER_TAIL.sub("", text)
        if stripped == text:
            break
        text = stripped.strip()
    if len(text) > _LABEL_CHARS:
        text = text[-_LABEL_CHARS:]
        text = text[text.find(" ") + 1 :]
    return text.strip(" ,.;:|/-–—")


def parse_products(text: str, limit: int = 3) -> list[dict]:
    """Return at most ``limit`` products, cheapest first.

    Each entry carries ``label``, ``price`` and ``rating`` (0.0 when the page named
    none), plus ``seller`` when the page named one. An empty list means nothing on
    the page parsed.
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
        label = _clean_label(head)
        if not label and not head.strip():
            label = _clean_label(tail)
        if not label:
            continue
        key = (label, price)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "label": label,
                "price": price,
                "currency": currency,
                "rating": _rating_of(tail) or _rating_of(head),
                "seller": _seller_of(tail) or _seller_of(head),
            }
        )
    entries.sort(key=lambda entry: entry["price"])
    return entries[:limit]


def _site_of(req: DealRequest | None) -> str:
    """The shop a product request opens, as it should be spoken."""
    if req is None:
        return ""
    return build_url(req, "")[1]


def _spoken_price(entry: dict) -> str:
    price = float(entry.get("price") or 0.0)
    whole = int(price)
    cents = round((price - whole) * 100)
    symbol = _SYMBOLS.get(entry.get("currency", ""), "$")
    return f"{symbol}{whole}.{cents:02d}" if cents else f"{symbol}{whole}"


def _spoken_rating(entry: dict) -> str:
    rating = float(entry.get("rating") or 0.0)
    if rating <= 0:
        return ""
    return f"rated {rating:g} out of 5"


def product_line(entry: dict, req: DealRequest | None, lead: str = "") -> str:
    """One product as a plain-speech fragment: label, price, rating, where it is sold.

    Shaped like "Cheapest: Anker 5-in-1 hub, $31, rated 4.6 out of 5 on Google Shopping.";
    the rating part appears only when the page carried one.
    """
    site = _site_of(req)
    seller = str(entry.get("seller") or "")
    where = seller if seller and seller.lower() != site.lower() else site
    label = str(entry.get("label") or "")
    parts = [f"{lead}: {label}" if lead and label else label, _spoken_price(entry)]
    rating = _spoken_rating(entry)
    if rating and where:
        parts.append(f"{rating} on {where}")
    else:
        if rating:
            parts.append(rating)
        if where:
            parts.append(f"on {where}")
    return ", ".join(part for part in parts if part)


def spoken_products(items: list[dict], req: DealRequest | None) -> str:
    """The top products said aloud, cheapest first, one sentence per product.

    The rating is said only when the page carried one.
    """
    leads = ("Cheapest", "Second", "Third", "Fourth", "Fifth")
    sentences = []
    for index, entry in enumerate(items):
        lead = leads[index] if index < len(leads) else f"Number {index + 1}"
        line = product_line(entry, req, lead)
        if line:
            sentences.append(f"{line}.")
    return " ".join(sentences)


# --- the flight/price/booking reader (crhy/voxa#48) -------------------------------
# Lives beside the product reader from #51 in this one module; its private helpers
# are prefixed _deal_ so the two parsers keep their own tables.


__all__ = ["parse_results", "spoken", "compact"]

# A price: a currency sign in front of an amount, or an amount followed by a
# currency word. Amounts may carry thousands separators and cents.
_deal_PRICE = re.compile(
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
_deal_ROW_BREAK = re.compile(
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
_deal_LOWER_TAIL = re.compile(r"^(?:[a-z'’.-]+ ){1,3}(?=[A-Z0-9])")

# Page furniture that is never a result: ads, sort controls, fare legends.
_CHROME = re.compile(
    r"(?:starting (?:from|at)|prices? (?:from|starting|only)|cheapest (?:price|fare|airfare)"
    r"|sorted by|sort by|sponsored|advertisement|sign in|log in|cookie|select your"
    r"|round ?trip|one ?way|nonstop (?:only|flights only)|book (?:now|flights?))",
    re.IGNORECASE,
)

# A bare number with no currency sign is read as dollars.
_CURRENCY_WORDS = {"$": "dollar", "¢": "cent", "€": "euro", "£": "pound", "": "dollar"}

_deal_HEAD_CHARS = 120
_deal_TAIL_CHARS = 60
_deal_LABEL_CHARS = 80
_deal_LABEL_WORDS = 8


def _deal_price_of(match: re.Match) -> tuple[float, str]:
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


def _deal_clean_label(fragment: str) -> str:
    """Strip the numbers and control words a result row carries around its label."""
    text = fragment
    cut = 0
    for match in _deal_ROW_BREAK.finditer(text):
        cut = match.end()
    text = text[cut:]
    for pattern in (_HOURS, _MINUTES, _STOP_COUNT, _NONSTOP, _DATE, _TIME, _NIGHTS):
        text = pattern.sub(" ", text)
    text = re.sub(r"[$€£¢]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:|/-–—")
    words = text.split()
    if len(words) > _deal_LABEL_WORDS:
        text = " ".join(words[-_deal_LABEL_WORDS:])
    for _ in range(2):
        stripped = _strip_connector(text)
        if stripped == text:
            break
        text = stripped.strip()
    text = _deal_LOWER_TAIL.sub("", text)
    if len(text) > _deal_LABEL_CHARS:
        text = text[-_deal_LABEL_CHARS:]
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


def _deal_spoken_price(price: float, currency: str) -> str:
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
    parts.append(_deal_spoken_price(entry["price"], entry.get("currency", "")))
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
    for match in _deal_PRICE.finditer(page):
        head = page[previous_end:match.start()]
        tail = page[match.end():match.end() + _deal_TAIL_CHARS]
        previous_end = match.end()
        if len(head) > _deal_HEAD_CHARS:
            head = head[-_deal_HEAD_CHARS:]
        price, currency = _deal_price_of(match)
        if price <= 0:
            continue
        label = _deal_clean_label(head) or _deal_clean_label(tail)
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
