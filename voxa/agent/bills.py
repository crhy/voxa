"""Bills and subscriptions found in the spending log: pure Python, standard library only.

Issue #60 asks for recurring charges, the questions "what is due this week" and "what
subscriptions do I pay for", and a reminder before a due date. The statement importer
(#45) is not merged, so detection runs on the rows the spending store already keeps —
the ``expenses.entries`` output — and nothing here touches the network or the disk.
"""

from __future__ import annotations

import calendar
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

from voxa.agent.dates import spoken_date
from voxa.agent.expenses import spoken_money

__all__ = [
    "Subscription",
    "recurring_charges",
    "next_due",
    "due_within",
    "spoken_subscriptions",
    "spoken_due",
]

MONTHLY = "monthly"
_TOLERANCE = 0.10  # a charge may wobble by a tenth and still be the same bill
_MAX_STEP = 6  # months between sightings beyond this is not a bill you pay for

# Words that say "this is a bill" without naming what the bill is for.
_FILLER = {
    "bill", "bills", "charge", "charges", "charged", "invoice", "invoices",
    "payment", "payments", "paid", "pay", "receipt", "receipts", "statement",
    "statements", "account", "monthly", "recurring", "due", "the", "my", "a",
    "an", "and", "for", "of", "us", "per",
}


@dataclass(frozen=True, slots=True)
class Subscription:
    """One recurring charge: what it is, what it costs, when it comes due next."""

    label: str
    category: str
    amount: float
    cadence: str
    last_seen: date
    next_due: date


def _words(label: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", (label or "").lower()) if word not in _FILLER]


def _key(label: str) -> str:
    """"Electricity bill" and "electricity" are the same charge."""
    return " ".join(sorted(set(_words(label))))


def _month_index(day: date) -> int:
    return day.year * 12 + day.month - 1


def _add_months(day: date, count: int) -> date:
    """The same day-of-month ``count`` months later, clamped to months that lack it."""
    index = _month_index(day) + count
    year, month = divmod(index, 12)
    month += 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def next_due(last_seen: date, today: date, months: int = 1) -> date:
    """Roll the charge's day-of-month forward month by month until it is today or later.

    Never returns a date in the past: the first sighting's own month is already spent.
    """
    step = max(1, int(months))
    hops = 1
    while True:
        candidate = _add_months(last_seen, hops * step)
        if candidate >= today:
            return candidate
        hops += 1


def _cadence(months: list[int]) -> int:
    """Months between sightings: the single gap if it is steady, otherwise one month."""
    if len(months) < 2:
        return 1
    gaps = {later - earlier for earlier, later in zip(months, months[1:], strict=False)}
    if len(gaps) == 1:
        gap = gaps.pop()
        if 1 <= gap <= _MAX_STEP:
            return gap
    return 1


def recurring_charges(
    items: list[tuple[date, float, str, str]],
    today: date,
    min_times: int = 3,
) -> list[Subscription]:
    """Charges whose label repeats in at least ``min_times`` months at a steady amount.

    One sighting per month counts, so a bill logged twice in a month is still one bill.
    """
    groups: dict[str, dict[int, tuple[date, float, str, str]]] = {}
    for entry in items:
        day, amount, _category, label = entry
        key = _key(label)
        if not key or amount <= 0:
            continue
        seen = groups.setdefault(key, {})
        month = _month_index(day)
        if month not in seen or day > seen[month][0]:
            seen[month] = entry

    wanted = max(2, int(min_times))
    found: list[Subscription] = []
    for _label, seen in groups.items():
        months = sorted(seen)
        if len(months) < wanted:
            continue
        rows = [seen[month] for month in months]
        amounts = sorted(row[1] for row in rows)
        middle = amounts[len(amounts) // 2]
        if middle <= 0:
            continue
        if any(abs(amount - middle) > _TOLERANCE * middle for amount in amounts):
            continue  # the amount swings too much to be one bill
        step = _cadence(months)
        labels = Counter(row[3] for row in rows)
        categories = Counter(row[2] for row in rows)
        last_seen = max(row[0] for row in rows)
        found.append(
            Subscription(
                label=labels.most_common(1)[0][0],
                category=categories.most_common(1)[0][0],
                amount=round(middle, 2),
                cadence=MONTHLY if step == 1 else f"every {step} months",
                last_seen=last_seen,
                next_due=next_due(last_seen, today, step),
            )
        )
    found.sort(key=lambda sub: (sub.next_due, sub.label.lower()))
    return found


def due_within(subs: list[Subscription], today: date, days: int = 7) -> list[Subscription]:
    """The charges due from today up to ``days`` days ahead, soonest first."""
    horizon = today + timedelta(days=max(0, int(days)))
    return sorted(
        (sub for sub in subs if today <= sub.next_due <= horizon),
        key=lambda sub: (sub.next_due, sub.label.lower()),
    )


def _cadence_spoken(cadence: str) -> str:
    return "every month" if cadence == MONTHLY else cadence


def spoken_subscriptions(subs: list[Subscription]) -> str:
    """Answer to "what subscriptions do I pay for"."""
    if not subs:
        return "I have not found any subscriptions in your spending log."
    parts = [
        f"{sub.label.lower()} {spoken_money(sub.amount)} {_cadence_spoken(sub.cadence)}"
        for sub in subs
    ]
    word = "subscriptions" if len(subs) > 1 else "subscription"
    return f"You pay for {len(subs)} {word}: " + ", ".join(parts) + "."


def spoken_due(subs: list[Subscription], today: date, days: int = 7) -> str:
    """Answer to "what bills are due this week"."""
    window = "this week" if int(days) == 7 else f"in the next {days} days"
    ready = due_within(subs, today, days)
    if not ready:
        return f"Nothing you pay for is due {window}."
    parts = [
        f"your {sub.label.lower()} bill {spoken_date(sub.next_due, today)} for {spoken_money(sub.amount)}"
        for sub in ready
    ]
    if len(ready) == 1:
        return f"One bill is due {window}: {parts[0]}."
    return f"{len(ready)} bills are due {window}: " + ", ".join(parts) + "."
