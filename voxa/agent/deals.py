from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

__all__ = ["DealRequest", "parse_request", "build_url", "describe", "detect_location"]


@dataclass(frozen=True, slots=True)
class DealRequest:
    kind: str
    what: str
    origin: str = ""
    depart: date | None = None
    ret: date | None = None
    max_price: int | None = None
    site: str = ""


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}
_MAKES = {
    "toyota", "ford", "honda", "chevrolet", "chevy", "nissan", "subaru", "tesla",
    "bmw", "mercedes", "audi", "kia", "hyundai", "mazda", "volkswagen", "vw",
    "jeep", "dodge", "ram", "cadillac", "buick", "gmc", "lexus", "infiniti",
    "acura", "volvo", "peugeot", "renault", "fiat", "opel", "skoda", "seat",
}
_VEHICLES = {
    "car", "cars", "truck", "trucks", "suv", "suv's", "sedan", "sedans",
    "van", "coupe", "minivan", "pickup", "wagon", "hatchback", "roadster",
    "tacoma", "f150", "f-150", "mustang", "corolla", "camry", "civic",
}
_SITES = {
    "expedia": "expedia", "kayak": "kayak", "amazon": "amazon",
    "ebay": "ebay", "google": "google", "google flights": "google",
    "hotels.com": "hotels.com", "booking": "booking", "booking.com": "booking",
}

_LEAD = re.compile(
    r"^(?:(find|get|show|look\s+up|search\s+for|book|buy)\s+)?"
    r"(?:me\s+)?"
    r"(?:(the\s+cheapest|a\s+cheap|the\s+best\s+price\s+on|a\s+deal\s+on|"
    r"the\s+lowest\s+price\s+for|cheapest|best\s+price\s+on|deal\s+on|"
    r"lowest\s+price\s+for)\s+)?",
    re.IGNORECASE,
)
_MONTH_WORDS = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEK_WORDS = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"

_DATE_PATTERNS = [
    (re.compile(rf"\bnext\s+({_WEEK_WORDS})\b", re.IGNORECASE), "weekday"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "tomorrow"),
    (re.compile(r"\bnext\s+week\b", re.IGNORECASE), "week"),
    (re.compile(r"\bnext\s+month\b", re.IGNORECASE), "month"),
    (re.compile(rf"\b(?:on|for|come|departing|leaving|starting|arriving)\s+(?:the\s+)?({_MONTH_WORDS})\s+(\d{{1,2}}){_ORD}\b", re.IGNORECASE), "md"),
    (re.compile(rf"\b({_MONTH_WORDS})\s+(\d{{1,2}}){_ORD}\b", re.IGNORECASE), "md"),
    (re.compile(rf"\bin\s+({_MONTH_WORDS})\b", re.IGNORECASE), "month-only"),
]


def _resolve_day(found: date, today: date) -> date:
    while found < today:
        found = found.replace(year=found.year + 1)
    return found


def find_date(s: str, today: date) -> tuple[date | None, str]:
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(s)
        if not m:
            continue
        rest = s[: m.start()] + s[m.end():]
        rest = re.sub(r"\s{2,}", " ", rest).strip()
        if kind == "weekday":
            target = _WEEKDAYS[m.group(1).lower()]
            found = today + timedelta(days=1)
            while found.weekday() != target:
                found += timedelta(days=1)
            return found, rest
        if kind == "tomorrow":
            return today + timedelta(days=1), rest
        if kind == "week":
            return today + timedelta(days=7), rest
        if kind == "month":
            return today + timedelta(days=30), rest
        month = _MONTHS[m.group(1).lower()]
        if kind == "month-only":
            found = date(today.year, month, 10)
        else:
            found = date(today.year, month, int(m.group(2)))
        return _resolve_day(found, today), rest
    return None, s


def parse_request(text: str, today: date) -> DealRequest | None:
    s = text.strip()
    s = re.sub(r"[?!]", "", s)
    s = re.sub(r"\s+", " ", s)
    lead = _LEAD.match(s)
    verb = (lead.group(1) or "").lower() if lead else ""
    price_phrase = bool(lead and lead.group(2))
    rest = s[lead.end():] if lead else s
    if not rest:
        return None

    site = ""
    m = re.search(
        r"\b(?:on|via|using|at)\s+(google flights|hotels\.com|booking\.com|expedia|kayak|amazon|ebay|google|booking)\b",
        rest,
        re.IGNORECASE,
    )
    if m:
        site = _SITES[m.group(1).lower()]
        rest = (rest[: m.start()] + rest[m.end():]).strip()

    max_price = None
    m = re.search(
        r"\b(?:under|less than|below|cheaper than|up to|for less than|maximum|max|priced)\s+\$?(\d[\d,]*)(k)?\b",
        rest,
        re.IGNORECASE,
    )
    if m:
        digits = int(m.group(1).replace(",", ""))
        max_price = digits * 1000 if m.group(2) else digits
        rest = (rest[: m.start()] + rest[m.end():]).strip()

    nights = None
    m = re.search(r"\bfor\s+(\d+)\s+nights?\b", rest, re.IGNORECASE)
    if m:
        nights = int(m.group(1))
        rest = (rest[: m.start()] + rest[m.end():]).strip()

    ret: date | None = None
    m = re.search(r"\b(?:returning|return|back)\s+(?:on\s+)?", rest, re.IGNORECASE)
    if m:
        tail = rest[m.end():]
        found, tail = find_date(tail, today)
        if found:
            ret = found
            rest = (rest[: m.start()] + " " + tail).strip()

    oneway = bool(re.search(r"\bone[-\s]way\b", rest, re.IGNORECASE))
    if oneway:
        rest = re.sub(r"\bone[-\s]way\b", "", rest, flags=re.IGNORECASE).strip()
    round_trip = bool(re.search(r"\bround[-\s]trip\b", rest, re.IGNORECASE))
    if round_trip:
        rest = re.sub(r"\bround[-\s]trip\b", "", rest, flags=re.IGNORECASE).strip()

    depart, rest = find_date(rest, today)

    origin = ""
    what = ""
    kind = ""

    m = re.search(r"\b(?:rental car|car rental|rent an? (?:electric )?car)\b", rest, re.IGNORECASE)
    if m:
        kind = "rental_car"
        m2 = re.search(r"\bin\s+(.+)$", rest, re.IGNORECASE)
        what = m2.group(1).strip() if m2 else ""

    if not kind:
        m = re.search(r"\b(?:plane |air )?(?:ticket|flight|airfare)s?\s+from\s+(.+?)\s+to\s+(.+)$", rest, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(?:fly to|(?:plane |air )?(?:ticket|flight|airfare)s? to)\s+(.+)$", rest, re.IGNORECASE)
        if m:
            kind = "flight"
            if m.lastindex == 2:
                origin, what = m.group(1), m.group(2)
            else:
                dest = m.group(1)
                if " from " in dest:
                    what, origin = dest.rsplit(" from ", 1)
                else:
                    what = dest

    if not kind:
        m = re.search(r"\b(?:hotel rooms?|hotels?|motels?|resorts?|place to stay)\s+in\s+(.+)$", rest, re.IGNORECASE)
        if m:
            kind = "hotel"
            what = m.group(1).strip()

    if not kind:
        buy_word = bool(re.search(r"\b(?:used|new|for sale|to buy|buying|buy)\b", rest, re.IGNORECASE))
        words = set(re.findall(r"[a-z0-9'-]+", rest.lower()))
        if buy_word and (words & _VEHICLES or words & _MAKES):
            kind = "car_for_sale"
            what = rest

    if not kind and (price_phrase or verb == "buy"):
        kind = "product"
        what = re.sub(r"^(?:a|an|the|some)\s+", "", rest, flags=re.IGNORECASE).strip()

    if not kind or not what:
        return None
    what = re.sub(
        r"^(?:a|an|the|some)\s+",
        "",
        re.sub(r"\s+(?:for sale|to buy)$", "", what.strip(" .,"), flags=re.IGNORECASE),
        flags=re.IGNORECASE,
    )

    if kind == "flight":
        if depart is None:
            depart = today + timedelta(days=14)
        if ret is None and not oneway:
            ret = depart + timedelta(days=7)
        if oneway:
            ret = None
    elif kind in ("hotel", "rental_car"):
        if depart is None:
            depart = today + timedelta(days=14)
        if ret is None:
            ret = depart + timedelta(days=nights if nights else 3)

    return DealRequest(
        kind=kind,
        what=what,
        origin=origin,
        depart=depart,
        ret=ret,
        max_price=max_price,
        site=site,
    )


def _q(value: str) -> str:
    return urllib.parse.quote(value)


def _slash(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _dash(d: date) -> str:
    return f"{d.year:02d}-{d.month:02d}-{d.day:02d}"


def build_url(req: DealRequest, location: str) -> tuple[str, str]:
    origin = req.origin or location
    if req.kind == "flight":
        if req.site in ("kayak", "google"):
            q = f"Flights to {req.what} from {origin} on {_dash(req.depart)} through {_dash(req.ret)}"
            return f"https://www.google.com/travel/flights?q={_q(q)}", "Google"
        trip = "roundtrip" if req.ret else "oneway"
        leg1 = f"from:{_q(origin)},to:{_q(req.what)},departure:{_slash(req.depart)}TANYT"
        leg2 = ""
        if req.ret:
            leg2 = f"&leg2=from:{_q(req.what)},to:{_q(origin)},departure:{_slash(req.ret)}TANYT"
        url = (
            f"https://www.expedia.com/Flights-Search?trip={trip}"
            f"&leg1={leg1}{leg2}&passengers=adults:1&mode=search&sort=PRICE_INCREASING"
        )
        return url, "Expedia"
    if req.kind == "hotel":
        url = (
            f"https://www.expedia.com/Hotel-Search?destination={_q(req.what)}"
            f"&startDate={_dash(req.depart)}&endDate={_dash(req.ret)}"
            f"&adults=2&sort=PRICE_LOW_TO_HIGH"
        )
        return url, "Expedia"
    if req.kind == "rental_car":
        url = (
            f"https://www.expedia.com/carsearch?locn={_q(req.what)}"
            f"&date1={_slash(req.depart)}&date2={_slash(req.ret)}&sort=price"
        )
        return url, "Expedia"
    if req.kind == "car_for_sale":
        url = (
            f"https://www.cars.com/shopping/results/?keyword={_q(req.what)}"
            f"&sort=list_price&zip=&maximum_distance=all"
        )
        if req.max_price is not None:
            url += f"&list_price_max={req.max_price}"
        return url, "cars.com"
    if req.site == "amazon":
        return f"https://www.amazon.com/s?k={_q(req.what)}&s=price-asc-rank", "Amazon"
    if req.site == "ebay":
        return f"https://www.ebay.com/sch/i.html?_nkw={_q(req.what)}&_sop=15", "eBay"
    return f"https://www.google.com/search?tbm=shop&q={_q(req.what)}&tbs=p_ord:p", "Google Shopping"


def _spoken_site(req: DealRequest, used: str) -> str:
    if used == "Expedia":
        return "Expedia"
    if req.site == "kayak":
        return "Kayak"
    if req.site == "google":
        return "Google"
    if req.site == "amazon":
        return "Amazon"
    if req.site == "ebay":
        return "eBay"
    return used


def _month_day(d: date) -> str:
    names = [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ]
    return f"{names[d.month - 1]} {d.day}"


def describe(req: DealRequest, location: str) -> str:
    origin = req.origin or location
    url, used = build_url(req, location)
    site = _spoken_site(req, used)
    if req.kind == "flight":
        plural = "flight" if req.what.lower() in ("it", "this") else "flights"
        text = f"Here are the cheapest {plural} from {origin} to {req.what}"
        if req.depart:
            text += f", leaving {_month_day(req.depart)}"
        if req.ret:
            text += f", returning {_month_day(req.ret)}"
        return f"{text}, on {site}."
    if req.kind == "hotel":
        text = f"Here are the cheapest hotels in {req.what}"
        if req.depart:
            text += f", checking in {_month_day(req.depart)}"
        if req.ret:
            text += f" for {(req.ret - req.depart).days} nights"
        return f"{text}, on {site}."
    if req.kind == "rental_car":
        text = f"Here are the cheapest rental cars in {req.what}"
        if req.depart:
            text += f", picking up {_month_day(req.depart)}"
        return f"{text}, on {site}."
    if req.kind == "car_for_sale":
        text = f"Here are the cheapest {req.what} matches"
        if req.max_price is not None:
            text += f" under {req.max_price} dollars"
        return f"{text}, on {site}."
    return f"Here are the cheapest {req.what} results, on {site}."


_LOCATION_CACHE: tuple[str, float] = ("", 0.0)
_CACHE_SECONDS = 30 * 60


def detect_location(fetch: Callable[[str], str] | None = None) -> str:
    global _LOCATION_CACHE
    now = time.monotonic()
    if _LOCATION_CACHE[0] and now - _LOCATION_CACHE[1] < _CACHE_SECONDS:
        return _LOCATION_CACHE[0]
    services = [
        ("https://ipapi.co/json", "city", "region"),
        ("https://ipinfo.io/json", "city", "region"),
        ("http://ip-api.com/json", "city", "regionName"),
    ]
    for url, city_key, region_key in services:
        try:
            if fetch is None:
                raw = urllib.request.urlopen(url, timeout=4).read()
            else:
                raw = fetch(url)
            data = json.loads(raw)
            city = str(data.get(city_key) or "").strip()
            if city:
                region = str(data.get(region_key) or "").strip()
                answer = f"{city}, {region}" if region else city
                _LOCATION_CACHE = (answer, now)
                return answer
        except Exception:
            continue
    return ""
