"""One spoken sentence in, one numbered invoice out.

Pure standard library. The sentence is read with the same money and day helpers
the spending log uses (`expenses.parse_amount`, `extract.find_dates`,
`dates.parse_day`); the running invoice number lives in this module's own state
file beside the other Voxa state files; and the document itself is written by
`voxa/documents.py` as an ODT that the desktop's LibreOffice renders and prints.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from voxa.agent import dates
from voxa.agent.expenses import parse_amount, spoken_money
from voxa.agent.extract import find_dates

STATE_FILE = "invoices.json"

# "net 40 dollars freight 6 taxes 2": the qualifier decides which bucket an
# amount falls into, and total is always net + freight + taxes. An amount nobody
# qualified is the net.
_QUALIFIER_WORDS: dict[str, str] = {
    "net": "net",
    "freight": "freight",
    "taxes": "taxes",
    "tax": "taxes",
    "shipping": "freight",
    "shipping charge": "freight",
    "carriage": "freight",
    "carriage charge": "freight",
    "handling": "freight",
    "handling charge": "freight",
}
_QUALIFIER = re.compile(
    r"\b(?P<word>net|freight|taxes|tax|shipping(?: charge)?|carriage(?: charge)?"
    r"|handling(?: charge)?)\b",
    re.IGNORECASE,
)
# Where a clause stops: the next comma, or the next clause keyword.
_CLAUSE = re.compile(
    r"[,;]|\b(?:terms?|due|deliver(?:y|ing)?|ship|send|made|dated|reference|total|for|to"
    r"|net|freight|taxes|tax|shipping|carriage|handling)\b",
    re.IGNORECASE,
)
# An amount nobody qualified: it has to look like money, not like a quantity.
_BARE_AMOUNT = re.compile(
    r"[$\u00a3\u00a4\u00a5\u20ac]\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars|bucks|usd)\b",
    re.IGNORECASE,
)
_AMOUNT_TOKEN = (
    r"[$\u00a3\u00a4\u00a5\u20ac]\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s*(?:dollars|bucks|usd))?(?:\s+and\s+\d{1,2}\s+cents)?"
    r"|\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s*(?:dollars|bucks|usd))?(?:\s+and\s+\d{1,2}\s+cents)?"
)
# "freight 6" and "6 dollars freight" both mean the same thing.
_AFTER_AMOUNT = re.compile(
    rf"^\s*(?:(?:charge|amount|is|was|of|for|a|the|in|to|at)\s+)*(?P<amt>{_AMOUNT_TOKEN})",
    re.IGNORECASE,
)
_BEFORE_AMOUNT = re.compile(
    rf"(?P<amt>{_AMOUNT_TOKEN})\s*(?:(?:charge|amount|and|with|plus|another|more)\s*)*$",
    re.IGNORECASE,
)
_CURRENCY_WORDS = re.compile(r"(?:dollars?|bucks|usd|cents?)(?:\s+and\s+\d{1,2}\s+cents?)?", re.IGNORECASE)
_HEAD = re.compile(
    r"^\s*(?:(?:please|hey|hi|ok|okay)[,,:]?\s+)*"
    r"(?:(?:can|could|would)\s+you\s+)?"
    r"(?:(?:make|create|generate|write|draft|issue|build)\s+(?:me\s+)?(?:an?|the|one)?\s*(?:new\s+)?)"
    r"(?P<kind>invoice|delivery\s+slip)\b"
    r"[\s,:]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_INVOICE_WORD = re.compile(r"\b(?:invoice|delivery\s+slip)s?\b", re.IGNORECASE)
_TO = re.compile(
    r"\b(?:(?:deliver(?:y|ing)?|ship(?:ping)?|send(?:ing)?|order|package)\s+to|to)\s+(?P<client>[^,;]+)",
    re.IGNORECASE,
)
# "to Acme with 6 shipping": the client's name stops at a connector word.
_AFTER_CLIENT = re.compile(
    r"\b(?:with|plus|including|and|for|at|terms?|due|net|freight|taxes|tax|shipping|carriage|handling)\b",
    re.IGNORECASE,
)
_FOR = re.compile(r"\bfor\s+(?P<item>[^,;]+)", re.IGNORECASE)
_TERMS = re.compile(r"\b(?:terms?|delivery\s+terms?|delivery\s+date)\b(?P<when>[^,;]*)", re.IGNORECASE)
_DUE = re.compile(r"\bdue\s+(?:by|on|in|before)?\s*(?P<when>[^,;]+)", re.IGNORECASE)
_FROM_TO = re.compile(r"\bfrom\s+(?P<start>[^,;]+?)\s+to\s+(?P<due>[^,;]+)", re.IGNORECASE)
_COUNTED = re.compile(
    r"^(?P<count>\d+|[a-z]+(?:-[a-z]+)?(?:\s+(?:hundred|thousand))?)\s+(?P<rest>\S.*)$",
    re.IGNORECASE,
)
_COUNTED_DAYS = re.compile(
    r"^(?:in\s+)?(?P<count>\d+|[a-z]+)(?:\s*(?:st|nd|rd|th))?\s*"
    r"(?P<unit>business\s+days?|days?|weeks?|months?)s?$",
    re.IGNORECASE,
)
_MONTH_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


class Invoice(NamedTuple):
    """What one spoken sentence asked for, in the order the document reads it."""

    client: str
    quantity: int
    description: str
    terms: list[tuple[date, date]]
    net: float
    freight: float
    taxes: float
    total: float


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" ,;:.!?")


def _cut_clause(text: str, start: int) -> str:
    """The text from start up to the next comma or clause keyword."""
    match = _CLAUSE.search(text, start)
    return _clean(text[start : match.start() if match else len(text)])


def _amount_in(chunk: str) -> float | None:
    # "1,200" is one amount, not two: the thousands separator goes first.
    value = parse_amount(re.sub(r"(?<=\d),(?=\d{3}\b)", "", chunk))
    return round(value, 2) if value is not None else None


def _money(text: str) -> tuple[float, float, float]:
    """(net, freight, taxes): qualified amounts first, then one bare amount."""
    buckets: dict[str, float] = {}
    matches = list(_QUALIFIER.finditer(text))
    for index, match in enumerate(matches):
        limit = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        clause = _CLAUSE.search(text, match.end())
        if clause is not None and clause.start() < limit:
            limit = clause.start()
        bucket = _QUALIFIER_WORDS[match.group("word").lower()]
        if bucket in buckets:
            continue
        after = text[match.end() : limit]
        found = _AFTER_AMOUNT.match(after)
        if found is None:
            start = matches[index - 1].end() if index else 0
            start = max(start, text.rfind(",", 0, match.start()) + 1)
            found = _BEFORE_AMOUNT.search(text[start : match.start()])
        if found is not None:
            buckets[bucket] = _amount_in(found.group("amt"))
        else:
            buckets[bucket] = _amount_in(after)
    net = buckets.get("net") or 0.0
    freight = buckets.get("freight") or 0.0
    taxes = buckets.get("taxes") or 0.0
    if not (net or freight or taxes):
        bare = _BARE_AMOUNT.search(text)
        if bare is not None:
            net = _amount_in(bare.group(0)) or 0.0
    return (net, freight, taxes)


def _client_and_item(text: str) -> tuple[str, str]:
    """(client, what is being invoiced) out of "for <thing> to <client>"."""
    to = _TO.search(text)
    if to is not None:
        client = _cut_clause(to.group("client"), 0)
        stop = _AFTER_CLIENT.search(client)
        if stop is not None:
            client = _clean(client[: stop.start()])
        item = ""
        for match in _FOR.finditer(text):
            if match.start() < to.start():
                item = _cut_clause(match.group("item"), 0)
        return (client, item)
    item = ""
    for match in _FOR.finditer(text):
        clause = _cut_clause(match.group("item"), 0)
        if not clause:
            continue
        if _quantity_and_description(clause)[0]:
            item = clause  # "two office chairs" is what is being invoiced, not who
            continue
        return (clause, item)
    return ("", item)


def _quantity_and_description(item: str) -> tuple[int, str]:
    """Split "two office chairs" into (2, "office chairs")."""
    item = _clean(item)
    if not item or _BARE_AMOUNT.fullmatch(item):
        return (0, "")
    match = _COUNTED.match(item)
    if match is not None and not _CURRENCY_WORDS.match(match.group("rest")):
        value = parse_amount(match.group("count"))
        if value is not None and value > 0:
            return (int(value), _clean(match.group("rest")))
    return (0, item)


def _add_months(day: date, months: int) -> date:
    month = day.month + months
    year = day.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, min(day.day, _MONTH_DAYS[month]))


def _day_on(spoken: str, today: date) -> date | None:
    """A calendar date from spoken words, rolled into the future."""
    for candidate in (spoken, *find_dates(spoken)):
        parsed = dates.parse_day(candidate, today)
        if parsed is None:
            continue
        year, month, day = parsed
        if year is None:
            for guess in (today.year, today.year + 1):
                try:
                    found = date(guess, month, day)
                except ValueError:
                    continue
                if found >= today:
                    return found
        else:
            try:
                return date(year, month, day)
            except ValueError:
                continue
    return None


def _term_when(spoken: str, today: date) -> tuple[date, date] | None:
    """One (start, due) pair from "15 days", "2 weeks", "March 3rd" or "tomorrow"."""
    text = _clean(spoken)
    if not text:
        return None
    counted = _COUNTED_DAYS.match(text)
    if counted is not None:
        raw = counted.group("count")
        number = int(raw) if raw.isdigit() else dates.NUMBER_WORDS.get(raw.lower())
        if number:
            unit = counted.group("unit").lower()
            if unit.startswith("week"):
                return (today, today + timedelta(days=number * 7))
            if unit.startswith("month"):
                return (today, _add_months(today, number))
            return (today, today + timedelta(days=number))
    due = _day_on(text, today)
    return (today, due) if due is not None else None


def _terms(text: str, today: date) -> list[tuple[date, date]]:
    """The dated terms: "terms 15 days", "terms March 3rd", "due Friday"."""
    out: list[tuple[date, date]] = []
    terms = _TERMS.search(text)
    if terms is not None:
        when = terms.start("when")
        pair = _term_when(_cut_clause(text, when), today)
        if pair is None:
            span = _FROM_TO.search(text, when)
            if span is not None:
                start = _day_on(_clean(span.group("start")), today)
                due = _day_on(_clean(span.group("due")), today)
                if start is not None and due is not None and due >= start:
                    pair = (start, due)
        if pair is not None:
            out.append(pair)
    due_match = _DUE.search(text)
    if due_match is not None:
        pair = _term_when(_clean(due_match.group("when")), today)
        if pair is not None:
            out.append(pair)
    return out


def parse_invoice_line(text: str, today: date) -> Invoice | None:
    """Parse one spoken sentence into an invoice, or None when it names no client."""
    s = _clean(text)
    if not s:
        return None
    head = _HEAD.match(s)
    if head is not None:
        if not _INVOICE_WORD.search(head.group("rest")):
            s = _clean(head.group("rest"))
    elif not _INVOICE_WORD.search(s):
        return None
    client, item = _client_and_item(s)
    if not client:
        return None
    quantity, description = _quantity_and_description(item)
    net, freight, taxes = _money(s)
    return Invoice(
        client,
        quantity,
        description,
        _terms(s, today),
        net,
        freight,
        taxes,
        round(net + freight + taxes, 2),
    )


def state_path() -> Path:
    """This module's own state file, beside the other Voxa state files."""
    state = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(state) / "voxa" / STATE_FILE


def _read_state() -> dict:
    try:
        with open(state_path()) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(state, handle)


def next_reference() -> int:
    """The next invoice number; the counter is kept between runs."""
    state = _read_state()
    number = int(state.get("next") or 1)
    state["next"] = number + 1
    _write_state(state)
    return number


def remember(invoice: Invoice, reference: int, today: date) -> None:
    """Store the invoice just made so "read me the invoice back" can speak it."""
    state = _read_state()
    state["last"] = {
        "reference": reference,
        "made": today.isoformat(),
        "client": invoice.client,
        "quantity": invoice.quantity,
        "description": invoice.description,
        "terms": [[start.isoformat(), due.isoformat()] for start, due in invoice.terms],
        "net": invoice.net,
        "freight": invoice.freight,
        "taxes": invoice.taxes,
        "total": invoice.total,
    }
    _write_state(state)


def last_invoice(today: date) -> tuple[Invoice, int] | None:
    """The last invoice made, or None when there has not been one."""
    record = _read_state().get("last")
    if not isinstance(record, dict):
        return None
    try:
        terms = [(date.fromisoformat(str(start)), date.fromisoformat(str(due))) for start, due in record.get("terms", [])]
        invoice = Invoice(
            str(record.get("client", "")),
            int(record.get("quantity", 0) or 0),
            str(record.get("description", "")),
            terms,
            float(record.get("net", 0.0)),
            float(record.get("freight", 0.0)),
            float(record.get("taxes", 0.0)),
            float(record.get("total", 0.0)),
        )
    except (ValueError, TypeError):
        return None
    return (invoice, int(record.get("reference", 0) or 0))


def title(invoice: Invoice, reference: int) -> str:
    """The document title, which also becomes the file name."""
    return f"Invoice {reference} for {invoice.client or 'the client'}"


def body(invoice: Invoice, today: date, reference: int) -> str:
    """The document lines: client, dated terms, delivery, and the money."""
    lines = [
        f"Reference: VOXA-{reference:04d}",
        f"Client: {invoice.client}",
        f"Dated: {today.isoformat()}",
    ]
    if invoice.description or invoice.quantity:
        what = invoice.description or "the order"
        count = f"{invoice.quantity} " if invoice.quantity else ""
        lines.append(f"Delivery: {count}{what}")
    for start, due in invoice.terms:
        lines.append(f"Terms: from {start.isoformat()} to {due.isoformat()}")
    lines.extend(
        [
            f"Net: {invoice.net:.2f}",
            f"Freight: {invoice.freight:.2f}",
            f"Taxes: {invoice.taxes:.2f}",
            f"Total: {invoice.total:.2f}",
        ]
    )
    return "\n".join(lines) + "\n"


def spoken_due(terms: list[tuple[date, date]], today: date) -> str:
    """Spoken form of the terms: "on October 25th", or the dates joined."""
    if not terms:
        return ""
    parts = [dates.spoken_date(due, today) for _start, due in terms]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def spoken_invoice(invoice: Invoice, today: date, reference: int = 0) -> str:
    """Read the invoice back: client, what, the money, and when it is due."""
    if invoice.description and invoice.quantity:
        item = f"{invoice.quantity} {invoice.description}"
    elif invoice.description:
        item = invoice.description
    elif invoice.quantity:
        item = f"{invoice.quantity} items"
    else:
        item = "the order"
    label = f"Invoice {reference} for" if reference else "An invoice for"
    money = (
        f"net {spoken_money(invoice.net)}, freight {spoken_money(invoice.freight)} "
        f"and taxes {spoken_money(invoice.taxes)}, {spoken_money(invoice.total)} total"
    )
    due = f", due {spoken_due(invoice.terms, today)}" if invoice.terms else ""
    return f"{label} {invoice.client}: {item}, {money}{due}."
