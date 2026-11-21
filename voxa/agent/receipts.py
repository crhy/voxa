"""Receipt text into ledger parts: the amount paid, what it was for, and the day.

A receipt photo or scan is turned into text on the host by tesseract (see
`voxa/agent/tools/receipts.py`); this module is pure and works on whatever text that
recognition returns, so the parsing is testable without an image. Nothing here invents an
amount: when no amount is printed the answer is None, and the tool then refuses to write a
ledger line.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from voxa.agent.expenses import parse_amount, spoken_money
from voxa.agent.extract import find_amounts, find_dates

# A receipt prints the amount paid on a line it labels; that line beats any guess.
_TOTAL_LABEL = re.compile(
    r"^(?:total|totals|total amount|amount(?: paid)?|net|charged|balance|you paid|i paid)\b",
    re.IGNORECASE,
)
# Lines that count things rather than money; their numbers must never become a total.
_NOT_MONEY = re.compile(
    r"\b(?:time|times|count|items?|guests?|calories?|miles?|minutes?|hours?|oz|lbs?|percent)\b",
    re.IGNORECASE,
)
# Lines that name a money figure rather than an item, so they never become the label.
_MONEY_LABEL = re.compile(
    r"^(?:sub-?total|total|totals|amount(?: paid)?|net|balance|charged|check amount|tax|tip"
    r"|change|cash|visa|mastercard|amex|gift (?:card|check)|register|table)\b",
    re.IGNORECASE,
)
# A bare amount with cents. extract.find_amounts leaves these alone: they carry no sign.
_BARE_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b")
# The lines a receipt ends with: neither the merchant nor an item.
_FOOTER = re.compile(
    r"thank (?:you|u)|have a (?:good|nice)|come back|visit|www\.|\.com|\.net|members? card"
    r"|loyalty|savings account|customer service|return policy|refund|warranty|carton|\bbag\b",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_LABEL_LIMIT = 60
_MAX_ITEMS = 12


def _money(token: str) -> float | None:
    """The number a money token stands for, with signs, commas and units removed."""
    digits = re.sub(r"[^\d.]", "", token)
    if not digits or not re.fullmatch(r"\d+(?:\.\d{1,2})?", digits):
        return None
    return round(float(digits), 2)


def _amounts_in(line: str) -> list[float]:
    """Amounts printed on one line: signed money first, then bare amounts with cents."""
    values = [value for value in map(_money, find_amounts(line)) if value is not None]
    if not values:
        values = [value for value in map(_money, _BARE_AMOUNT.findall(line)) if value is not None]
    return values


def _amount_on(line: str) -> float | None:
    """The amount on a labelled total line; the value is written after the label."""
    values = _amounts_in(line)
    if values:
        return values[-1]
    if _NOT_MONEY.search(line):
        return None
    return parse_amount(line)


def _clean(line: str) -> str:
    """A receipt line as words: dates, amounts and quantities dropped, lower case."""
    text = re.sub(r"[^\x20-\x7e]", " ", line)
    for token in find_dates(text):
        text = text.replace(token, " ")
    text = re.sub(_BARE_AMOUNT.pattern, " ", text)
    text = " ".join(text.split())
    text = re.sub(r"\s+(?:x\s*)?\d+(?:\.\d+)?$", "", text)
    return text.strip(" -,:;|/*").lower()


def _skipped(line: str) -> bool:
    """A line that is a money figure, a running total or a footer, not a merchant or an item."""
    return bool(_TOTAL_LABEL.match(line) or _MONEY_LABEL.match(line) or _FOOTER.search(line))


def _is_item(line: str) -> bool:
    if _skipped(line):
        return False
    return bool(re.search(r"[a-z]", _clean(line)))


def parse_receipt(text: str) -> tuple[float | None, str, list[str]]:
    """Return the paid total, the merchant or heading, and the item lines of a receipt.

    The total comes from a line the receipt labels (TOTAL, TOTALS, AMOUNT). Only when no
    such line exists does the largest amount in the text stand in for it, and None when the
    print carries no amount at all.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total: float | None = None
    for line in lines:
        if _TOTAL_LABEL.match(line):
            value = _amount_on(line)
            if value is not None:
                total = value
    if total is None:
        amounts = [value for line in lines for value in _amounts_in(line)]
        total = max(amounts) if amounts else None

    merchant = ""
    for line in lines:
        if _skipped(line):
            continue
        words = _clean(line)
        if not re.search(r"[a-z]", words):
            continue
        merchant = words
        break

    items: list[str] = []
    for line in lines:
        if not _is_item(line):
            continue
        words = _clean(line)
        if words and words != merchant and words not in items:
            items.append(words)
        if len(items) >= _MAX_ITEMS:
            break
    return (total, merchant, items)


def receipt_what(text: str) -> str:
    """The label for the ledger line, and the words expenses.category_for classifies.

    The merchant alone is usually unclassifiable ("accepco"), so the first item lines ride
    along with it: a deli receipt becomes food, a fuel receipt transport.
    """
    _total, merchant, items = parse_receipt(text)
    parts = [merchant] if merchant else []
    parts.extend(items[:3])
    label = " ".join(parts).strip()
    if len(label) > _LABEL_LIMIT:
        label = label[:_LABEL_LIMIT].rsplit(" ", 1)[0]
    return label


def _safe_date(year: int, month: int, day: int, today: date) -> date | None:
    try:
        found = date(year, month, day)
    except ValueError:
        return None
    if year < 1900 or year > today.year + 1 or found > today + timedelta(days=1):
        return None
    return found


def _to_date(token: str, today: date) -> date | None:
    """One printed date as a real day; None when it cannot be read or cannot be this receipt."""
    token = token.strip()
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", token)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        return _safe_date(year, month, day, today)
    slash = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", token)
    if slash:
        first, second, year = (int(part) for part in slash.groups())
        if year < 100:
            year += 2000
        month, day = first, second
        if month > 12 and second <= 12:  # a day-first print
            month, day = second, first
        return _safe_date(year, month, day, today)
    named = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", token.lower())
    if named and named.group(1) in _MONTHS:
        return _safe_date(int(named.group(3)), _MONTHS[named.group(1)], int(named.group(2)), today)
    named = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})", token.lower())
    if named and named.group(2) in _MONTHS:
        return _safe_date(int(named.group(3)), _MONTHS[named.group(2)], int(named.group(1)), today)
    return None


def receipt_date(text: str, today: date) -> date | None:
    """The day the receipt was printed; today when the print carries no readable date."""
    for token in find_dates(text):
        found = _to_date(token, today)
        if found is not None:
            return found
    return today


def spoken_receipt(amount: float | None, day: date, category: str) -> str:
    """The reply for a receipt that was read: the amount, its category and its day."""
    money = spoken_money(amount) if amount is not None else "nothing"
    return f"That receipt is {money}, filed under {category} for {day.isoformat()}."
