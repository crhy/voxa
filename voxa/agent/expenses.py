from __future__ import annotations

import re
from datetime import date, timedelta

FILE_NAME = "Voxa Expenses.md"

CATEGORIES: dict[str, list[str]] = {
    "food": ["lunch", "dinner", "breakfast", "coffee", "groceries", "restaurant", "pizza", "snack"],
    "transport": ["gas", "fuel", "petrol", "bus", "train", "taxi", "uber", "lyft", "parking"],
    "home": ["rent", "electricity", "water", "internet", "utilities"],
    "fun": ["movie", "movies", "game", "games", "concert", "beer", "drinks"],
    "health": ["pharmacy", "doctor", "dentist", "medicine", "gym"],
    "other": [],
}

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_NUM_WORDS = {**_UNITS, **_TENS}


def parse_amount(spoken: str) -> float | None:
    text = spoken.lower()
    cents = re.search(r"(\d+(?:\.\d+)?)\s+dollars\s+and\s+(\d{1,2})\s+cents", text)
    if cents:
        return round(float(cents.group(1)) + int(cents.group(2)) / 100.0, 2)
    numeric = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars|bucks)?", text)
    if numeric:
        return round(float(numeric.group(1) or numeric.group(2)), 2)
    words = re.findall(r"[a-z]+", text)
    for i, word in enumerate(words):
        if word in _NUM_WORDS or word in _SCALES or word in ("a", "an"):
            value = _word_amount(words[i:])
            if value is not None:
                return value
    return None


def _word_amount(words: list[str]) -> float | None:
    total = 0.0
    current = 0.0
    seen = False
    for word in words:
        if word in ("a", "an"):
            continue
        if word in _NUM_WORDS:
            current += _NUM_WORDS[word]
            seen = True
        elif word in _SCALES:
            current = (current if current else 1) * _SCALES[word]
            seen = True
        else:
            break
    return (total + current) if seen else None


def category_for(what: str) -> str:
    low = what.lower()
    for category, words in CATEGORIES.items():
        if category == "other":
            continue
        for word in words:
            if re.search(rf"\b{word}\b", low):
                return category
    return "other"


def add(content: str, day: date, amount: float, what: str, tag: str = "") -> str:
    category = category_for(what)
    label = what.strip()
    if label:
        label = label[0].upper() + label[1:]
    line = f"- {day.isoformat()} | {amount:.2f} | {category} | {label}"
    if tag.strip():
        line += f" | trip:{tag.strip()}"
    line += "\n"
    if content and not content.endswith("\n"):
        content += "\n"
    return content + line


def entries(content: str) -> list[tuple[date, float, str, str]]:
    out: list[tuple[date, float, str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = line[2:].split("|")
        if len(parts) not in (4, 5):
            continue
        try:
            day = date.fromisoformat(parts[0].strip())
            amount = float(parts[1].strip())
        except ValueError:
            continue
        out.append((day, amount, parts[2].strip(), parts[3].strip()))
    return out


def tagged(content: str, tag: str) -> list[tuple[date, float, str, str]]:
    want = tag.strip().lower()
    out: list[tuple[date, float, str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = line[2:].split("|")
        if len(parts) != 5:
            continue
        trip = parts[4].strip()
        if not trip.lower().startswith("trip:"):
            continue
        if trip[5:].strip().lower() != want:
            continue
        try:
            day = date.fromisoformat(parts[0].strip())
            amount = float(parts[1].strip())
        except ValueError:
            continue
        out.append((day, amount, parts[2].strip(), parts[3].strip()))
    return out


def budgets(content: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- budget |"):
            continue
        parts = line[2:].split("|")
        if len(parts) != 3:
            continue
        try:
            amount = float(parts[2].strip())
        except ValueError:
            continue
        out[parts[1].strip()] = amount
    return out


def set_budget(content: str, category: str, amount: float) -> str:
    lines = content.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- budget |"):
            parts = stripped[2:].split("|")
            if len(parts) == 3 and parts[1].strip() == category:
                continue
        kept.append(line)
    new = "".join(f"{line}\n" for line in kept)
    if new and not new.endswith("\n"):
        new += "\n"
    return new + f"- budget | {category} | {amount:g}\n"


def remove_last(content: str) -> tuple[str, tuple | None]:
    lines = content.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("- "):
            removed = entries(lines[i] + "\n")
            if not removed:
                return (content, None)
            del lines[i]
            new = "".join(f"{line}\n" for line in lines)
            return (new, removed[0])
    return (content, None)


def total(items: list[tuple[date, float, str, str]], since: date, category: str | None = None) -> float:
    acc = 0.0
    for day, amount, cat, _what in items:
        if day >= since and (category is None or cat == category):
            acc += amount
    return round(acc, 2)


def period_start(spoken: str, today: date) -> tuple[date, str]:
    text = spoken.lower().strip()
    if "today" in text:
        return (today, "today")
    if "week" in text:
        monday = today - timedelta(days=today.weekday())
        return (monday, "this week")
    if "year" in text:
        return (date(today.year, 1, 1), "this year")
    return (date(today.year, today.month, 1), "this month")


def spoken_money(amount: float) -> str:
    amount = round(amount, 2)
    dollars = int(amount)
    cents = round((amount - dollars) * 100)
    dollar_word = "dollar" if dollars == 1 else "dollars"
    if cents == 0:
        if dollars == 0:
            return "0 dollars"
        return f"{dollars} {dollar_word}"
    if dollars == 0:
        return f"{cents} cents"
    return f"{dollars} {dollar_word} and {cents} cents"


def spoken_total(items: list[tuple[date, float, str, str]], today: date, period: str, category: str | None) -> str:
    start, words = period_start(period, today)
    amount = total(items, start, category)
    if amount == 0:
        return f"You have not spent anything {words}."
    money = spoken_money(amount)
    if category:
        return f"You have spent {money} on {category} {words}."
    return f"You have spent {money} {words}."


def spoken_breakdown(items: list[tuple[date, float, str, str]], today: date, period: str) -> str:
    start, words = period_start(period, today)
    sums: dict[str, float] = {}
    for day, amount, cat, _what in items:
        if day >= start:
            sums[cat] = round(sums.get(cat, 0.0) + amount, 2)
    sums = {cat: value for cat, value in sums.items() if value}
    if not sums:
        return f"You have not spent anything {words}."
    ordered = sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{cat} {spoken_money(value)}" for cat, value in ordered]
    label = words[0].upper() + words[1:]
    return f"{label}: " + ", ".join(parts) + "."


def spoken_budget(items: list[tuple[date, float, str, str]], budgets: dict[str, float], today: date, category: str | None = None) -> str:
    if not budgets:
        return "You have not set any budgets yet."
    start, words = period_start("", today)
    if category:
        if category not in budgets:
            return f"You have not set a {category} budget yet."
        budget = budgets[category]
        spent = total(items, start, category)
        if spent <= budget:
            left = round(budget - spent, 2)
            return (
                f"You have spent {spoken_money(spent)} of your {budget:g} dollar {category} budget {words}. "
                f"{spoken_money(left)} left."
            )
        over = round(spent - budget, 2)
        return f"You are {spoken_money(over)} over your {budget:g} dollar {category} budget {words}."
    parts = []
    for cat, budget in budgets.items():
        spent = total(items, start, cat)
        parts.append(f"{cat[0].upper() + cat[1:]}: {spent:g} of {budget:g} dollars.")
    return " ".join(parts)


def spoken_trip(items: list[tuple[date, float, str, str]], tag: str) -> str:
    name = tag.strip().title()
    if not items:
        return f"I have nothing logged for the {name} trip."
    acc = round(sum(amount for _day, amount, _cat, _what in items), 2)
    sums: dict[str, float] = {}
    for _day, amount, cat, _what in items:
        sums[cat] = round(sums.get(cat, 0.0) + amount, 2)
    parts = [f"{cat} {value:g}" for cat, value in sums.items()]
    return f"The {name} trip has cost {spoken_money(acc)} so far: " + ", ".join(parts) + "."


def to_csv(items: list[tuple[date, float, str, str]]) -> str:
    lines = ["Date,Amount,Category,What"]
    for day, amount, cat, what in items:
        lines.append(f"{day.isoformat()},{amount:.2f},{cat},{what}")
    return "\n".join(lines) + "\n"
