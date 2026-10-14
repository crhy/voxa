"""Pure text helpers for look-ups that open the right page (issue #102)."""

from __future__ import annotations

import re

AIRLINE_CODES = {
    "united": "UA",
    "delta": "DL",
    "american": "AA",
    "southwest": "WN",
    "alaska": "AS",
    "jetblue": "B6",
    "jet blue": "B6",
    "lufthansa": "LH",
    "british airways": "BA",
    "air france": "AF",
    "klm": "KL",
    "emirates": "EK",
    "air canada": "AC",
    "qantas": "QF",
    "ryanair": "FR",
    "easyjet": "U2",
    "easy jet": "U2",
}

DIGIT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

PERSON_STOP_WORDS = frozenset({"the", "a", "my", "your", "this", "that", "it", "he", "she", "they"})

CURRENCY_CODES = {
    "australian dollar": "AUD",
    "australian dollars": "AUD",
    "aud": "AUD",
    "british pound": "GBP",
    "british pounds": "GBP",
    "brittish pound": "GBP",
    "brittish pounds": "GBP",
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "cad": "CAD",
    "dollar": "USD",
    "dollars": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "gbp": "GBP",
    "japanese yen": "JPY",
    "jpy": "JPY",
    "pound": "GBP",
    "pound sterling": "GBP",
    "pounds": "GBP",
    "sterling": "GBP",
    "swiss franc": "CHF",
    "swiss francs": "CHF",
    "u s dollar": "USD",
    "u s dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "usd": "USD",
    "yen": "JPY",
}

STOCK_SYMBOLS = {
    "alphabet": ("GOOGL", "Alphabet"),
    "amazon": ("AMZN", "Amazon"),
    "apple": ("AAPL", "Apple"),
    "ford": ("F", "Ford"),
    "ford motor company": ("F", "Ford"),
    "general electric": ("GE", "GE Aerospace"),
    "google": ("GOOGL", "Alphabet"),
    "hewlett packard": ("HPQ", "HP"),
    "hp": ("HPQ", "HP"),
    "hp inc": ("HPQ", "HP"),
    "hpe": ("HPE", "Hewlett Packard Enterprise"),
    "hewlett packard enterprise": ("HPE", "Hewlett Packard Enterprise"),
    "mercedes": ("MBG.DE", "Mercedes-Benz"),
    "mercedes benz": ("MBG.DE", "Mercedes-Benz"),
    "meta": ("META", "Meta"),
    "microsoft": ("MSFT", "Microsoft"),
    "nvidia": ("NVDA", "Nvidia"),
    "tesla": ("TSLA", "Tesla"),
    "toyota": ("TM", "Toyota"),
}

PRIVATE_COMPANIES = {
    "general atomics": "General Atomics",
}


def flight_code(spoken: str) -> str | None:
    """"united 123", "UA 123", "delta flight 45" -> "UA123"; None when unparsable.

    Digits may be spoken as words ("one two three").
    """
    text = " ".join(spoken.lower().replace("-", " ").split())
    if not text:
        return None
    airline = None
    for name in sorted(AIRLINE_CODES, key=len, reverse=True):
        if text == name or text.startswith(name + " "):
            airline = AIRLINE_CODES[name]
            text = text[len(name):].strip()
            break
    if airline is None:
        code = re.match(r"([a-z][a-z0-9])\s+(.+)$", text)
        if code:
            airline = code.group(1).upper()
            text = code.group(2)
    text = re.sub(r"^flight\s+", "", text).strip()
    digits = ""
    for token in text.split():
        if token.isdigit():
            digits += token
        elif token in DIGIT_WORDS:
            digits += DIGIT_WORDS[token]
        else:
            return None
    if airline is None or not digits:
        return None
    return airline + digits


def tracking_number(spoken: str) -> str | None:
    """Letters and digits with spaces removed and upper-cased; None unless at least 8 of them."""
    cleaned = "".join(spoken.split()).upper()
    if len(cleaned) >= 8 and cleaned.isalnum():
        return cleaned
    return None


def market_target(what: str) -> str:
    """The thing to price, normalised; "" means the stock market in general."""
    text = re.sub(r"^the\s+", "", " ".join(what.lower().split()).strip())
    if text in ("", "market", "stock market", "stocks", "the stock market"):
        return ""
    return text


def currency_code(spoken: str) -> str | None:
    """Return an ISO currency code for a common spoken currency name."""
    text = " ".join(spoken.casefold().replace(".", " ").split()).strip(" ,?!.\"")
    text = re.sub(r"^(?:the|one|a)\s+", "", text)
    text = re.sub(r"\s+worth$", "", text)
    return CURRENCY_CODES.get(text)


def stock_name(what: str) -> str:
    """Remove quote wording and return the requested company name or ticker."""
    text = market_target(what)
    text = re.sub(
        r"^(?:(?:current )?stock price (?:of|for)|(?:current )?price (?:of|for))\s+",
        "",
        text,
    )
    # Natural dictation often puts the quantity before the company ("one
    # Toyota share") instead of saying "one share of Toyota".
    quantity_last = re.fullmatch(r"(?:one|a)\s+(.+?)\s+shares?", text)
    if quantity_last:
        text = quantity_last.group(1)
    text = re.sub(r"^(?:(?:one|a) shares?|shares?) (?:of|in)\s+", "", text)
    text = re.sub(r"\s+(?:right now|currently|today)$", "", text).strip()
    text = re.sub(r"(?:'s)?\s+(?:stock|stock price|share price|shares?)$", "", text).strip()
    if text in {"stock", "share", "shares", "stock price", "share price", "price"}:
        return ""
    return re.sub(r"'s$", "", text).strip()


def stock_target(what: str) -> tuple[str, str] | None:
    """Resolve a known company name or ticker into (ticker, spoken company name)."""
    text = stock_name(what)
    if text in STOCK_SYMBOLS:
        return STOCK_SYMBOLS[text]
    if re.fullmatch(r"[a-z]{1,5}", text) and text not in {
        "a", "an", "one", "price", "share", "shares", "stock", "today", "gold", "silver"
    }:
        return text.upper(), text.upper()
    return None


def private_company(what: str) -> str | None:
    """Return the display name when a requested company has no public shares."""
    return PRIVATE_COMPANIES.get(stock_name(what))


def person_words_ok(name: str) -> bool:
    """True when a "who is X" name looks like a person: 2-4 words, no small words."""
    words = name.split()
    return 2 <= len(words) <= 4 and not any(word.lower() in PERSON_STOP_WORDS for word in words)
