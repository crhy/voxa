"""Bank statements the user already has: read them, categorise them, compare months.

Standard library only, and nothing here talks to the network: the statement is read
from the user's own disk, the transactions go into Voxa's local spending log, and
the merchant corrections live in Voxa's own state file. No transaction data ever
leaves the machine.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from calendar import month_name
from datetime import date, timedelta
from statistics import mean

from voxa.agent import extract
from voxa.agent.expenses import CATEGORIES, category_for, spoken_money

# ---------------------------------------------------------------------------
# Merchants: the words that say which category a statement line belongs to.
# The spoken-word table in expenses.CATEGORIES stays the fallback, so a
# description the user dictated by hand keeps categorising as it always did.
# ---------------------------------------------------------------------------

MERCHANT_RULES: dict[str, tuple[str, ...]] = {
    "food": (
        "supermarket", "grocery", "groceries", "kroger", "safeway", "albertsons",
        "whole foods", "aldi", "tesco", "deli", "corner shop", "cafe", "coffee",
        "bakery", "butcher", "fishmonger",
    ),
    "transport": (
        "fuel", "petrol", "diesel", "gas station", "shell", "bp", "chevron", "exxon",
        "parking", "toll", "transit", "metro card", "ovoyo", "uber", "lyft", "taxi",
        "airline", "air lines", "airfare", "amtrak", "rail", "flight", "baggage",
    ),
    "home": (
        "utility", "utilities", "electricity", "electric", "water", "internet",
        "cable", "xfinity", "comcast", "verizon", "heating", "oil", "rent",
        "laundry", "cleaners", "hardware",
    ),
    "fun": (
        "cinema", "movie", "theater", "theatre", "concert", "festival", "museum",
        "netflix", "brewery", "winery", "distillery", "casino", "gaming", "lottery",
    ),
    "health": (
        "pharmacy", "drugstore", "drug store", "walgreens", "cvsmarket", "gym",
        "fitness", "yoga", "dentist", "dental", "optometry", "clinic", "hospital",
        "medicine", "pharma",
    ),
}

# ---------------------------------------------------------------------------
# Remembered corrections: "that merchant is food, not other". A correction the
# user makes once is kept, so the same merchant is never misfiled again.
# ---------------------------------------------------------------------------

CORRECTIONS_FILE = "statement-corrections.json"
_ONE_DAY = timedelta(days=1)
_MONTH_NUMBERS = {word.lower(): number for number, word in enumerate(month_name) if word}


def corrections_path() -> str:
    override = os.environ.get("VOXA_STATEMENT_CORRECTIONS")
    if override:
        return override
    state = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(state, "voxa", CORRECTIONS_FILE)


def correction_key(description: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", description.lower()).split())


def normalise_category(value: str) -> str:
    low = value.strip().lower()
    if low in CATEGORIES:
        return low
    return "other"


def load_corrections(path: str | None = None) -> dict[str, str]:
    try:
        with open(path or corrections_path()) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): normalise_category(value)
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def remember_correction(description: str, category: str, path: str | None = None) -> None:
    target = path or corrections_path()
    wanted = normalise_category(category)
    if not wanted:
        return
    stored = load_corrections(target)
    stored[correction_key(description)] = wanted
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as handle:
            json.dump(stored, handle)
    except OSError:
        # A correction we could not write is only lost for the next import.
        return


def remembered_correction(description: str, category: str, path: str | None = None) -> str:
    """The category the user corrected this merchant to, remembered for later imports."""
    wanted = normalise_category(category)
    remember_correction(description, wanted, path)
    return wanted


def categorise(description: str, corrections: dict[str, str] | None = None) -> str:
    """Category for one statement line: correction, then merchant, then the spoken words."""
    if corrections is None:
        corrections = load_corrections()
    wanted = corrections.get(correction_key(description))
    if wanted:
        return wanted
    low = description.lower()
    for category, words in MERCHANT_RULES.items():
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", low):
                return category
    return category_for(low)


# ---------------------------------------------------------------------------
# Reading the file: a CSV body with a header, or plain text lines.
# ---------------------------------------------------------------------------

_MONTH_WORDS = "|".join(word.lower() for word in month_name if word)

_DATE_HEADER = re.compile(r"\b(?:date|transaction date|posting date|posted|value date)\b", re.IGNORECASE)
_AMOUNT_HEADER = re.compile(r"\b(?:amount|usd|dollars|withdrawal|deposit|value amount)\b", re.IGNORECASE)
_DESCRIPTION_HEADER = re.compile(
    r"\b(?:description|details|detail|merchant|payee|counterparty|name|store|outlet)\b", re.IGNORECASE
)
# A summary row must never be counted twice, so these lines are dropped.
SUMMARY_LINE = re.compile(
    r"\b(?:total|totals|subtotal|balance|available|outstanding|overview|summary|"
    r"monthly|annual|range|card number|expiration|expiry|account number|online account|payments)\b",
    re.IGNORECASE,
)
_BARE_AMOUNT = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?![\d])")
_CURRENCY_CODE = re.compile(r"\s*(?:usd|eur|gbp|jpy|cad|aud|krw|inr)\b", re.IGNORECASE)


def _blank(text: str, value: str) -> str:
    index = text.find(value)
    if index < 0:
        return text
    return text[:index] + " " * len(value) + text[index + len(value):]


def _parse_day(value: str) -> date | None:
    text = value.strip().rstrip(".")
    try:
        return date.fromisoformat(text.split()[0])
    except ValueError:
        pass
    slashes = re.match(r"^(\d{1,2})[/./-](\d{1,2})[/./-](\d{2}|\d{4})$", text)
    if slashes:
        first, second, year_text = int(slashes.group(1)), int(slashes.group(2)), slashes.group(3)
        year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
        if first > 12:
            month, day = second, first
        else:
            # Bank statements from the user's bank put the month first.
            month, day = first, second
        try:
            return date(year, month, day)
        except ValueError:
            return None
    named = re.match(rf"^(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_WORDS})\s+(\d{{4}})$", text, re.IGNORECASE)
    if named:
        return _safe_date(int(named.group(3)), _MONTH_NUMBERS[named.group(2).lower()], int(named.group(1)))
    named = re.match(rf"^({_MONTH_WORDS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})$", text, re.IGNORECASE)
    if named:
        return _safe_date(int(named.group(3)), _MONTH_NUMBERS[named.group(1).lower()], int(named.group(2)))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_amount(value: str) -> float | None:
    # A statement writes an outflow as -45.99 or (45.99); spending is stored positive.
    text = _CURRENCY_CODE.sub("", value.strip())
    digits = re.sub(r"[^0-9.]", "", text)
    if not digits or digits.count(".") > 1:
        return None
    try:
        amount = float(digits)
    except ValueError:
        return None
    if amount == 0.0:
        return None
    return abs(round(amount, 2))


def _amount_tokens(line: str) -> list[str]:
    """Money on one line. extract.find_amounts first; bare decimals as well, because
    a statement column often carries no currency sign at all. Dates are blanked out
    first so the numbers inside a date are never read as an amount."""
    text = line
    for value in extract.find_dates(line):
        text = _blank(text, value)
    found = extract.find_amounts(text)
    if found:
        return found
    return [match.group(0) for match in _BARE_AMOUNT.finditer(text)]


def _description_from(line: str) -> str:
    """The merchant words left on a line once its date and money are removed."""
    text = line
    for value in extract.find_dates(line):
        text = _blank(text, value)
    for value in _amount_tokens(text):
        text = _blank(text, value)
    words = text.replace("|", " ").replace(",", " ").split()
    kept = [word for word in words if re.search(r"[A-Za-z]", word)]
    return " ".join(kept).strip(" -–*")


def _text_rows(text: str) -> list[tuple[date, float, str]]:
    out: list[tuple[date, float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or SUMMARY_LINE.search(line):
            continue
        dates = extract.find_dates(line)
        if not dates:
            continue
        day = _parse_day(dates[0])
        if day is None:
            continue
        amounts = _amount_tokens(line)
        if not amounts:
            continue
        amount = _parse_amount(amounts[0])
        if amount is None:
            continue
        description = _description_from(line)
        if not description:
            continue
        out.append((day, amount, description))
    return out


def _header_columns(row: list[str]) -> dict[str, list[int]]:
    columns: dict[str, list[int]] = {"date": [], "amount": [], "description": []}
    for index, cell in enumerate(row):
        cell = cell.strip()
        if not cell:
            continue
        for key, pattern in (("date", _DATE_HEADER), ("amount", _AMOUNT_HEADER), ("description", _DESCRIPTION_HEADER)):
            if pattern.search(cell):
                columns[key].append(index)
                break
    return columns


def _csv_rows(text: str) -> list[tuple[date, float, str]]:
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    header_at: int | None = None
    columns: dict[str, list[int]] = {}
    for index, row in enumerate(rows[:5]):
        found = _header_columns(row)
        if found.get("date") and found.get("amount"):
            header_at, columns = index, found
            break
    if header_at is None:
        return []
    out: list[tuple[date, float, str]] = []
    for row in rows[header_at + 1:]:
        if SUMMARY_LINE.search(" ".join(row)):
            continue
        day_cell = row[columns["date"][0]] if columns["date"][0] < len(row) else ""
        amount_cell = row[columns["amount"][0]] if columns["amount"][0] < len(row) else ""
        day = _parse_day(day_cell)
        amount = _parse_amount(amount_cell)
        if day is None or amount is None:
            continue
        parts: list[str] = []
        for index in columns["description"]:
            if index >= len(row):
                continue
            cell = row[index].strip()
            if cell and cell not in parts:
                parts.append(cell)
        description = " ".join(parts)
        if not description:
            continue
        out.append((day, amount, description))
    return out


def parse_statement(text: str) -> list[tuple[date, float, str]]:
    """Transactions in a statement: (day, amount, description), in file order.

    A CSV body with a date/amount/description header is read with the csv module;
    anything else is read line by line, pairing the dates and money on a line with
    the merchant words left over. Summary and balance lines are never counted.
    """
    rows = _csv_rows(text)
    if rows:
        return rows
    return _text_rows(text)


def transactions(text: str, corrections: dict[str, str] | None = None) -> list[tuple[date, float, str, str]]:
    """Statement rows as spending-log items: (day, amount, category, description)."""
    if corrections is None:
        corrections = load_corrections()
    return [
        (day, amount, categorise(description, corrections), description)
        for day, amount, description in parse_statement(text)
    ]


def candidate_lines(text: str) -> int:
    """Lines that look like transactions, so an import can say what it skipped."""
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if SUMMARY_LINE.search(line):
            continue
        if extract.find_dates(line) and _amount_tokens(line):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Reports over the spending log: items are (day, amount, category, description).
# ---------------------------------------------------------------------------


def month_start(day: date) -> date:
    return date(day.year, day.month, 1)


def next_month(start: date) -> date:
    return date(start.year + (start.month == 12), 1 if start.month == 12 else start.month + 1, 1)


def month_label(start: date) -> str:
    return f"{month_name[start.month]} {start.year}"


def _month_key(spoken: str, today: date) -> date:
    """Which month the user means: 'this month', 'last month', 'march 2026'."""
    text = spoken.strip().lower()
    if not text or "this month" in text or text in ("month", "now", "current"):
        return month_start(today)
    if "last month" in text or "previous month" in text or "past month" in text:
        return month_start(month_start(today) - _ONE_DAY)
    named = re.search(rf"\b({_MONTH_WORDS})\b(?:\s+(\d{{4}}))?", text, re.IGNORECASE)
    if named:
        month = _MONTH_NUMBERS[named.group(1).lower()]
        year = int(named.group(2)) if named.group(2) else today.year
        if month > today.month and year == today.year and "last" not in text:
            year -= 1
        return _safe_date(year, month, 1) or month_start(today)
    return month_start(today)


def compare_months(
    items: list[tuple[date, float, str, str]],
    month: date | str = "",
    today: date | None = None,
) -> dict:
    """This month against the month before it, per category, with the deltas."""
    today = today or date.today()
    current = month if isinstance(month, date) else _month_key(str(month), today)
    current = month_start(current)
    previous = month_start(current - _ONE_DAY)
    end = next_month(current)
    categories = sorted({cat for _day, _amount, cat, _what in items} | set(CATEGORIES))
    per: dict[str, tuple[float, float, float]] = {}
    for category in categories:
        this = round(sum(a for day, a, cat, _w in items if current <= day < end and cat == category), 2)
        last = round(sum(a for day, a, cat, _w in items if previous <= day < current and cat == category), 2)
        if not this and not last:
            continue
        per[category] = (this, last, round(this - last, 2))
    this_total = round(sum(a for day, a, _c, _w in items if current <= day < end), 2)
    last_total = round(sum(a for day, a, _c, _w in items if previous <= day < current), 2)
    return {
        "start": current,
        "previous_start": previous,
        "label": month_label(current),
        "previous_label": month_label(previous),
        "categories": per,
        "total": (this_total, last_total, round(this_total - last_total, 2)),
    }


def totals_by_category(items: list[tuple[date, float, str, str]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    for _day, amount, category, _what in items:
        sums[category] = round(sums.get(category, 0.0) + amount, 2)
    return dict(sorted(sums.items(), key=lambda kv: (-kv[1], kv[0])))


def largest_items(items: list[tuple[date, float, str, str]], count: int = 3) -> list[tuple[date, float, str, str]]:
    return sorted(items, key=lambda item: -item[1])[:count]


def recurring_merchants(items: list[tuple[date, float, str, str]], times: int = 2) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _day, _amount, _category, what in items:
        key = correction_key(what)
        counts[key] = counts.get(key, 0) + 1
    return sorted(
        [(what, n) for what, n in counts.items() if n >= times],
        key=lambda pair: (-pair[1], pair[0]),
    )


def unusual_charges(items: list[tuple[date, float, str, str]], factor: float = 2.0) -> list[tuple[date, float, str, str]]:
    """Charges far bigger than the rest of the statement, which is what the user wants
    to hear about: an unexpected purchase, a deposit, a mistake."""
    if len(items) < 3:
        return []
    typical = mean(amount for _day, amount, _category, _what in items)
    return sorted(
        [item for item in items if item[1] >= typical * factor],
        key=lambda item: -item[1],
    )


# ---------------------------------------------------------------------------
# Spoken reports.
# ---------------------------------------------------------------------------


def _join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def spoken_compare(result: dict) -> str:
    if not result["categories"]:
        return f"You have no spending recorded for {result['label']} or {result['previous_label']}."
    this_total, last_total, delta = result["total"]
    if this_total == 0:
        return f"You have nothing recorded for {result['label']}."
    lines = [f"In {result['label']} you spent {spoken_money(this_total)}, against {spoken_money(last_total)} in {result['previous_label']}."]
    moves = []
    for category, (_this, _last, change) in sorted(result["categories"].items(), key=lambda kv: -abs(kv[1][2])):
        if change == 0:
            continue
        moves.append(f"{category} is {'up' if change > 0 else 'down'} {spoken_money(abs(change))}")
    if moves:
        lines.append("The change: " + _join(moves) + ".")
    return " ".join(lines)


def spoken_category_compare(result: dict, category: str) -> str:
    """One category, this month against the month before it."""
    wanted = category.strip().lower()
    entry = result["categories"].get(wanted)
    if entry is None:
        return f"You have no {wanted or 'spending'} recorded in {result['label']} or {result['previous_label']}."
    this, last, change = entry
    if change == 0:
        return f"Your {wanted} spending is unchanged between {result['previous_label']} and {result['label']}: {spoken_money(this)}."
    return (
        f"In {result['label']} you spent {spoken_money(this)} on {wanted}, against "
        f"{spoken_money(last)} in {result['previous_label']}: {wanted} is "
        f"{'up' if change > 0 else 'down'} {spoken_money(abs(change))}."
    )


def spoken_largest(items: list[tuple[date, float, str, str]]) -> str:
    if not items:
        return "There is nothing to report."
    biggest = largest_items(items)
    parts = [f"{spoken_money(amount)} at {what.lower()}" for _day, amount, _category, what in biggest]
    return f"The biggest charges: {_join(parts)}."


def spoken_recurring(items: list[tuple[date, float, str, str]]) -> str:
    merchants = recurring_merchants(items)
    if not merchants:
        return "No merchant appears more than once."
    parts = [f"{what.lower()} {n} times" for what, n in merchants[:3]]
    return f"You pay most often at {_join(parts)}."


def spoken_unusual(items: list[tuple[date, float, str, str]]) -> str:
    charges = unusual_charges(items)
    if not charges:
        return "Nothing stands out as unusual."
    parts = [f"{spoken_money(amount)} at {what.lower()} on {day.isoformat()}" for day, amount, _category, what in charges[:3]]
    return f"Unusually large: {_join(parts)}."


def spoken_totals(items: list[tuple[date, float, str, str]]) -> str:
    sums = totals_by_category(items)
    if not sums:
        return "You have no spending recorded."
    parts = [f"{category} {spoken_money(value)}" for category, value in sums.items()]
    return f"By category: {_join(parts)}."


def report_body(items: list[tuple[date, float, str, str]], result: dict) -> str:
    """The text of the formatted document: totals, biggest, recurring, unusual, comparison."""
    lines = [f"Statement report: {len(items)} transactions"]
    sums = totals_by_category(items)
    if sums:
        lines.append("Totals by category")
        lines.extend(f"  {category}: {value:.2f}" for category, value in sums.items())
    lines.append("Largest charges")
    lines.extend(f"  {day.isoformat()}  {amount:.2f}  {what}  ({category})" for day, amount, category, what in largest_items(items))
    merchants = recurring_merchants(items)
    if merchants:
        lines.append("Recurring merchants")
        lines.extend(f"  {what}  {n} times" for what, n in merchants[:5])
    charges = unusual_charges(items)
    if charges:
        lines.append("Unusually large charges")
        lines.extend(f"  {day.isoformat()}  {amount:.2f}  {what}" for day, amount, _category, what in charges[:5])
    if result["categories"]:
        lines.append(f"{result['label']} against {result['previous_label']}")
        for category, (this, last, delta) in result["categories"].items():
            lines.append(f"  {category}: {this:.2f} against {last:.2f} ({delta:+.2f})")
        this_total, last_total, delta = result["total"]
        lines.append(f"  Total: {this_total:.2f} against {last_total:.2f} ({delta:+.2f})")
    return "\n".join(lines)
