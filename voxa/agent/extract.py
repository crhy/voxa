"""Pure helpers for pulling dates and money amounts out of document text."""

import re

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)

_DATE_PATTERNS = (
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{4}}\b",
)

_CURRENCY = r"[$\u00a3\u00a4\u00a5\u20ac]"

_AMOUNT_PATTERNS = (
    rf"{_CURRENCY}\s*\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{1,2}})?",
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:usd|eur|gbp|jpy|cad|aud|krw|inr)\b",
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b",
)


def _collect(text: str, patterns: tuple[str, ...]) -> list[str]:
    """Match each pattern in order, skipping consumed spans and repeated values."""
    taken = [False] * len(text)
    seen: set[str] = set()
    out: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if any(taken[m.start() : m.end()]):
                continue
            value = m.group(0)
            if value in seen:
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            seen.add(value)
            out.append(value)
    return out


def find_dates(text: str) -> list[str]:
    """Return the date strings found in text, in order, de-duplicated."""
    return _collect(text, _DATE_PATTERNS)


def find_amounts(text: str) -> list[str]:
    """Return the money-amount strings found in text, in order, de-duplicated."""
    return _collect(text, _AMOUNT_PATTERNS)


def spoken(name: str, dates: list[str], amounts: list[str], names: list[str]) -> str:
    """Build the reply line for the doc_extract tool."""

    def join(vals: list[str]) -> str:
        return ", ".join(vals) if vals else "none found"

    return f"In {name}: dates {join(dates)}; amounts {join(amounts)}; names {join(names)}."
