"""Look-ups that open the right page (issue #102): flights, packages, prices, people, markets, fact checks."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable

from voxa.agent import host, lookups
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

fetch: Callable[[str], str] | None = None


def _fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Voxa/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:  # noqa: S310 - fixed https hosts
        return response.read().decode("utf-8", "replace")


def _open(url: str) -> bool:
    return host.spawn(["xdg-open", url]) is not False


_COMPANY_FILLER = frozenset({
    "class", "co", "company", "corp", "corporation", "group", "holding", "holdings",
    "inc", "incorporated", "limited", "ltd", "motor", "motors", "plc", "the",
})


def _company_words(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", value.casefold())
        if word not in _COMPANY_FILLER
    }


def _find_stock(query: str) -> tuple[str, str] | None:
    """Resolve an unfamiliar company name with Yahoo's public symbol search."""
    url = (
        "https://query1.finance.yahoo.com/v1/finance/search?q="
        + urllib.parse.quote_plus(query)
        + "&quotesCount=6&newsCount=0"
    )
    payload = json.loads((fetch or _fetch)(url))
    wanted = _company_words(query)
    if not wanted:
        return None
    for item in payload.get("quotes") or []:
        if item.get("quoteType") != "EQUITY":
            continue
        symbol = str(item.get("symbol") or "").strip()
        name = str(item.get("longname") or item.get("shortname") or "").strip()
        if symbol and name and wanted <= _company_words(name):
            return symbol, name
    return None


def _yahoo_quote(symbol: str) -> tuple[float, str]:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(symbol, safe="")
        + "?interval=1d&range=1d"
    )
    payload = json.loads((fetch or _fetch)(url))
    meta = payload["chart"]["result"][0]["meta"]
    if meta.get("instrumentType") != "EQUITY":
        raise ValueError("not an equity")
    return float(meta["regularMarketPrice"]), str(meta.get("currency") or "USD").upper()


def _currency_for_speech(code: str) -> str:
    return {
        "AUD": "Australian dollars",
        "CAD": "Canadian dollars",
        "CHF": "Swiss francs",
        "EUR": "euros",
        "GBP": "British pounds",
        "GBX": "British pence",
        "JPY": "Japanese yen",
        "USD": "U.S. dollars",
    }.get(code, code)


def flight_status(args: dict[str, str]) -> ToolResult:
    code = lookups.flight_code(args.get("flight") or "")
    if code is None:
        return ToolResult.failure("Which flight? Say the airline and the number.")
    url = f"https://www.flightaware.com/live/flight/{code}"
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here is the status of flight {code[:2]} {code[2:]}.", detail=url)


def track_package(args: dict[str, str]) -> ToolResult:
    number = lookups.tracking_number(args.get("tracking") or "")
    if number is None:
        return ToolResult.failure("I need the tracking number.")
    url = f"https://parcelsapp.com/en/tracking/{number}"
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success("Here is the tracking for that package.", detail=url)


def compare_prices(args: dict[str, str]) -> ToolResult:
    item = (args.get("item") or "").strip()
    if not item:
        return ToolResult.failure("I need to know what to compare.")
    url = "https://www.google.com/search?tbm=shop&q=" + urllib.parse.quote_plus(item)
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here are prices for {item} from several shops.", detail=url)


def look_up(args: dict[str, str]) -> ToolResult:
    name = (args.get("name") or "").strip()
    if not name:
        return ToolResult.failure("I need a name.")
    kind = (args.get("kind") or "").strip().lower()
    if kind == "company":
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(f"{name} company")
    else:
        url = "https://en.wikipedia.org/w/index.php?search=" + urllib.parse.quote_plus(name)
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(f"Here is what I found about {name}.", detail=url)


def market(args: dict[str, str]) -> ToolResult:
    what = lookups.market_target(args.get("what") or "")
    name = lookups.stock_name(what)
    if what and not name:
        return ToolResult.failure("Which company? Say its name or stock ticker.")
    private = lookups.private_company(what)
    if private is not None:
        return ToolResult.success(
            f"{private} is privately held, so it does not have a public stock price."
        )
    stock = lookups.stock_target(what)
    if stock is None and what and name not in {"bitcoin", "ethereum", "gold", "silver"}:
        try:
            stock = _find_stock(name)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, OSError):
            stock = None
    if stock is not None:
        symbol, name = stock
        url = f"https://api.nasdaq.com/api/quote/{urllib.parse.quote(symbol)}/info?assetclass=stocks"
        try:
            payload = json.loads((fetch or _fetch)(url))
            data = payload["data"]
            quote = data["primaryData"]
            price = float(str(quote["lastSalePrice"]).replace("$", "").replace(",", ""))
            status = str(data.get("marketStatus") or "").casefold()
            session = " in after-hours trading" if "after" in status else ""
            speech = f"{name} stock is {price:,.2f} U.S. dollars per share{session}."
            return ToolResult.success(speech, detail=url)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            try:
                price, currency = _yahoo_quote(symbol)
                speech = f"{name} stock is {price:,.2f} {_currency_for_speech(currency)} per share."
                return ToolResult.success(speech, detail=url)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, OSError):
                return ToolResult.failure(
                    f"I couldn't retrieve a current quote for {name} right now.", detail=url
                )
    if re.search(r"\b(?:stock|shares?)\b", what, re.IGNORECASE):
        return ToolResult.failure(
            f"I couldn't find a publicly traded company matching {name}."
        )
    if not what:
        url = "https://www.google.com/finance"
        speech = "Here is the market today."
    else:
        url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(f"{what} price")
        speech = f"Here is the price of {what}."
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success(speech, detail=url)


def exchange_rate(args: dict[str, str]) -> ToolResult:
    base = lookups.currency_code(args.get("base") or "")
    quote = lookups.currency_code(args.get("quote") or "")
    if base is None or quote is None:
        return ToolResult.failure("Tell me which two currencies to compare.")
    url = f"https://api.frankfurter.dev/v2/rate/{base.lower()}/{quote.lower()}"
    try:
        payload = json.loads((fetch or _fetch)(url))
        rate = float(payload["rate"])
        date = str(payload.get("date") or "")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return ToolResult.failure("I couldn't retrieve the exchange rate right now.", detail=url)
    names = {
        "AUD": ("Australian dollar", "Australian dollars"),
        "CAD": ("Canadian dollar", "Canadian dollars"),
        "CHF": ("Swiss franc", "Swiss francs"),
        "EUR": ("euro", "euros"),
        "GBP": ("British pound", "British pounds"),
        "JPY": ("Japanese yen", "Japanese yen"),
        "USD": ("U.S. dollar", "U.S. dollars"),
    }
    amount = f"{rate:.4f}".rstrip("0").rstrip(".")
    source_date = f", using the reference rate for {date}" if date else ""
    speech = f"One {names[base][0]} is {amount} {names[quote][1]}{source_date}."
    return ToolResult.success(speech, detail=url)


def fact_check(args: dict[str, str]) -> ToolResult:
    claim = (args.get("claim") or "").strip()
    if not claim:
        return ToolResult.failure("Which claim? Say what to check.")
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(f"fact check {claim}")
    if not _open(url):
        return ToolResult.failure("I couldn't open the browser.", detail=url)
    return ToolResult.success("Here is what fact checkers say about that.", detail=url)


def lookups_tools() -> list[Tool]:
    return [
        Tool(name="flight_status", description="Open the live status page for a flight.",
             parameters={"flight": "the airline and number, e.g. united 123 or UA 123"},
             risk=RiskLevel.REVERSIBLE, handler=flight_status),
        Tool(name="track_package", description="Open the tracking page for a parcel.",
             parameters={"tracking": "the tracking number: letters and digits"},
             risk=RiskLevel.REVERSIBLE, handler=track_package),
        Tool(name="compare_prices", description="Show prices for an item from several shops.",
             parameters={"item": "what to compare, e.g. a cordless drill"},
             risk=RiskLevel.REVERSIBLE, handler=compare_prices),
        Tool(name="look_up", description="Look up a company or a person.",
             parameters={"name": "the company or person", "kind": "company or person"},
             risk=RiskLevel.REVERSIBLE, handler=look_up),
        Tool(name="market", description="Show the stock market or the price of something.",
             parameters={"what": "bitcoin, gold, apple stock; empty for the market in general"},
             risk=RiskLevel.REVERSIBLE, handler=market),
        Tool(name="exchange_rate", description="Give the latest reference exchange rate between two currencies.",
             parameters={"base": "currency to convert from", "quote": "currency to convert to"},
             risk=RiskLevel.READ_ONLY, handler=exchange_rate),
        Tool(name="fact_check", description="Open fact checkers for a claim.",
             parameters={"claim": "the claim to check"},
             risk=RiskLevel.REVERSIBLE, handler=fact_check),
    ]
