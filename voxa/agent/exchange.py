"""Bitcoin price, balance and recent orders, read from the user's own exchange (issue #46).

What this slice is
------------------
The first step of issue #46's plan: price, balance and recent orders through one
exchange API. ``price`` is the exchange's public ticker and needs no key at all;
``balance`` and ``recent_orders`` are signed requests made with the key kept in
``Settings.exchange_key`` / ``exchange_secret``.

What this slice never is
------------------------
It never trades, never withdraws, never sends funds, and never touches a wallet,
a card or a payment provider: there is no buy, no sell, no deposit, no
withdrawal and no payment call anywhere in this module or in
``voxa/agent/tools/exchange.py``. ``bitcoin_buy`` is registered only so the
phrase is recognised; its handler answers ``BUY_REFUSAL`` without touching the
network, because how much may be spent is the decision issue #43 records and
rhY has not made it yet - so there is nothing here to invent a limit from, and
no order log either (the issue logs an order id, amount, price and fee only
once a buy exists).

Key setup - issue #46's "documented setup"
------------------------------------------
1. Create the key on the exchange with trading permission only and
   **withdrawals disabled** (Kraken: no Withdraw permission; Coinbase: no
   "account:withdraw" / no wallet-withdraw scope; Bitstamp: an instant-settlement
   key with no withdrawal permission).
2. Keep it in the secrets store: ``Settings.exchange_name``,
   ``exchange_key`` and ``exchange_secret`` (Preferences on Spaced Linux).
   ``backup.py SECRET_KEY_BASES`` redacts both values in backups.
3. A key that *can* withdraw is not supported. The tools never ask it to, but
   the key's own restriction - not this code - is what protects the funds.

Endpoints used (the exchanges' own documented paths)
----------------------------------------------------
kraken:    GET  0/public/Ticker?pair=XBT<QUOTE>   price
           POST  0/private/Balance                balance
           POST  0/private/Query2                 recent orders
coinbase:  GET  /v2/prices/BTC-<QUOTE>/spot       price
           GET  /v2/accounts/BTC/balance          balance
           GET  /v2/transactions                  recent orders
bitstamp:  GET  /api/v2/ticker/btc<quote>/        price
           GET  /api/v2/instantsettlement/balance/ balance
           GET  /api/v2/instantsettlement/order_list/ recent orders

Coinbase keys are used without a passphrase (the ``CB-ACCESS-PASSPHRASE`` header
is sent empty); a key created with a passphrase is not supported by this slice.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from voxa.config import EXCHANGE_NAMES

__all__ = [
    "BUY_REFUSAL",
    "NOT_CONFIGURED",
    "EXCHANGES",
    "Balance",
    "Order",
    "Quote",
    "balance",
    "client_settings",
    "display_name",
    "price",
    "recent_orders",
    "signed_settings",
    "spoken_balance",
    "spoken_orders",
    "spoken_price",
]

log = logging.getLogger("voxa.agent.exchange")

EXCHANGES = EXCHANGE_NAMES
DEFAULT_CURRENCY = "USD"
TIMEOUT_SECONDS = 6.0

BASE_URLS = {
    "kraken": "https://api.kraken.com",
    "coinbase": "https://api.coinbase.com",
    "bitstamp": "https://www.bitstamp.net",
}

# The buy half of issue #46 stays closed until issue #43 decides the spending rule.
BUY_REFUSAL = (
    "I won't buy Bitcoin yet. Spending money needs the rule from issue #43, "
    "which isn't decided; this only reads price, balance and recent orders."
)
NOT_CONFIGURED = (
    "Your exchange isn't set up. Choose Kraken, Coinbase or Bitstamp in Preferences "
    "and add a trade-only key with withdrawals disabled."
)

_CURRENCY_WORDS = {
    "AUD": "Australian dollars",
    "CAD": "Canadian dollars",
    "CHF": "Swiss francs",
    "EUR": "euros",
    "GBP": "British pounds",
    "JPY": "Japanese yen",
    "USD": "U.S. dollars",
}
# Quote currencies the three exchanges all offer a Bitcoin ticker for.
CURRENCIES = frozenset({"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "USD"})

# Injectable exactly like tools/lookups.py's ``fetch``: tests point it at a fake
# server, and the same exception set as homeassistant.py is swallowed around it.
fetch_json: Callable[[str, dict], dict] | None = None

_NETWORK_ERRORS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    TimeoutError,
    ValueError,
    json.JSONDecodeError,
)


@dataclass(frozen=True, slots=True)
class ExchangeSettings:
    name: str
    key: str = ""
    secret: str = ""

    def signed(self) -> bool:
        """True when a key and secret are present (balance and orders need them)."""
        return bool(self.key and self.secret)


@dataclass(frozen=True, slots=True)
class Quote:
    exchange: str
    currency: str
    price: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Balance:
    exchange: str
    amount: float
    unit: str = "BTC"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Order:
    id: str
    side: str
    pair: str
    volume: float
    price: float
    when: str = ""
    exchange: str = ""


def client_settings() -> ExchangeSettings | None:
    """The configured exchange, or None when nothing is set up."""
    from voxa.config import ConfigStore

    settings = ConfigStore().load()
    name = (settings.exchange_name or "").strip().casefold()
    if name not in EXCHANGE_NAMES:
        return None
    return ExchangeSettings(name, (settings.exchange_key or "").strip(),
                             (settings.exchange_secret or "").strip())


def signed_settings() -> ExchangeSettings | None:
    """The configured exchange only when its key and secret are present."""
    settings = client_settings()
    return settings if settings is not None and settings.signed() else None


def _fetch(url: str, request: dict) -> dict:
    """Run ``request`` and return the parsed JSON object, or {} on any failure."""
    try:
        payload = (fetch_json or _fetch_json)(url, request)
    except _NETWORK_ERRORS as exc:
        log.info("exchange request to %s failed: %s", url, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _fetch_json(url: str, request: dict) -> dict:
    """The real network call: fixed https exchange hosts only."""
    data = request.get("data")
    http = urllib.request.Request(
        url,
        data=data.encode("utf-8") if data else None,
        headers=dict(request.get("headers") or {}),
        method=str(request.get("method") or "GET"),
    )
    with urllib.request.urlopen(http, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed exchange hosts
        return json.loads(response.read().decode("utf-8", "replace"))


def _number(value: object) -> float:
    try:
        return float(str(value).replace(",", "").split(" ")[0])
    except (TypeError, ValueError):
        return 0.0


# --- signatures (each exchange's documented scheme) ---

def _kraken_signature(secret: str, path: str, nonce: str) -> str:
    """base64 HMAC-SHA256 with the base64-decoded secret over path + nonce."""
    key = base64.b64decode(secret)
    digest = hmac.new(key, path.encode() + nonce.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def _coinbase_signature(secret: str, timestamp: str, path: str) -> str:
    """hex HMAC-SHA256 of the secret over timestamp + path."""
    return hmac.new(secret.encode(), timestamp.encode() + path.encode(),
                    hashlib.sha256).hexdigest()


def _bitstamp_signature(secret: str, path: str, nonce: str) -> str:
    """hex HMAC-SHA256 of the secret over path + nonce."""
    return hmac.new(secret.encode(), path.encode() + nonce.encode(),
                    hashlib.sha256).hexdigest()


def _kraken_private(settings: ExchangeSettings, path: str) -> tuple[str, dict]:
    nonce = str(int(time.time() * 1000))
    signature = _kraken_signature(settings.secret, path, nonce)
    body = urllib.parse.urlencode({"key": settings.key, "signature": signature,
                                   "nonce": nonce})
    return (f"{BASE_URLS['kraken']}{path}",
            {"method": "POST",
             "headers": {"Content-Type": "application/x-www-form-urlencoded"},
             "data": body})


def _coinbase_get(settings: ExchangeSettings, path: str) -> tuple[str, dict]:
    timestamp = str(int(time.time()))
    return (f"{BASE_URLS['coinbase']}{path}",
            {"method": "GET",
             "headers": {
                 "CB-ACCESS-KEY": settings.key,
                 "CB-ACCESS-SIGN": _coinbase_signature(settings.secret, timestamp, path),
                 "CB-ACCESS-TIMESTAMP": timestamp,
                 "CB-ACCESS-PASSPHRASE": "",
             }})


def _bitstamp_get(settings: ExchangeSettings, path: str) -> tuple[str, dict]:
    nonce = str(int(time.time() * 1000))
    return (f"{BASE_URLS['bitstamp']}{path}",
            {"method": "GET",
             "headers": {
                 "Bitstamp-ApiKey": settings.key,
                 "Bitstamp-Nonce": nonce,
                 "Bitstamp-Signature": _bitstamp_signature(settings.secret, path, nonce),
             }})


def _signed(settings: ExchangeSettings, path: str) -> tuple[str, dict]:
    if settings.name == "kraken":
        return _kraken_private(settings, path)
    if settings.name == "coinbase":
        return _coinbase_get(settings, path)
    return _bitstamp_get(settings, path)


# --- price: the exchange's public ticker, no key needed ---

def _ticker_request(name: str, currency: str) -> tuple[str, dict]:
    code = currency.upper()
    if name == "kraken":
        path = f"/0/public/Ticker?pair=XBT{code}"
    elif name == "coinbase":
        path = f"/v2/prices/BTC-{code}/spot"
    else:
        path = f"/api/v2/ticker/btc{code.lower()}/"
    return f"{BASE_URLS[name]}{path}", {"method": "GET",
                                        "headers": {"Accept": "application/json"}}


def _ticker_price(name: str, payload: dict) -> float:
    if name == "kraken":
        result = payload.get("result")
        if isinstance(result, dict):
            for pair in result.values():
                if not isinstance(pair, dict):
                    continue
                for field in ("b", "c"):
                    rows = pair.get(field) or []
                    if rows:
                        return _number(str(rows[0]).split(" - ")[0])
    elif name == "coinbase":
        data = payload.get("data") or {}
        if isinstance(data, dict):
            return _number(data.get("amount") or data.get("spot"))
    else:
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if isinstance(data, dict):
            return _number(data.get("last"))
    return 0.0


def price(currency: str = DEFAULT_CURRENCY) -> Quote | None:
    """The current Bitcoin price in ``currency`` from the configured exchange."""
    settings = client_settings()
    if settings is None:
        return None
    code = currency.upper() if currency.upper() in CURRENCIES else DEFAULT_CURRENCY
    url, request = _ticker_request(settings.name, code)
    payload = _fetch(url, request)
    if not payload:
        return None
    value = _ticker_price(settings.name, payload)
    if value <= 0:
        return None
    return Quote(settings.name, code, value, url)


# --- balance: signed, read-only ---

BALANCE_PATHS = {
    "kraken": "/0/private/Balance",
    "coinbase": "/v2/accounts/BTC/balance",
    "bitstamp": "/api/v2/instantsettlement/balance/",
}


def _balance_amount(name: str, payload: dict) -> float:
    if name == "kraken":
        result = payload.get("result")
        if isinstance(result, dict):
            value = result.get("XBT") or result.get("BTC") or next(iter(result.values()), 0)
            return _number(value)
        return 0.0
    if name == "coinbase":
        data = payload.get("data") or {}
        return _number(data.get("amount")) if isinstance(data, dict) else 0.0
    return _number(payload.get("balance"))


def balance() -> Balance | None:
    """How much Bitcoin the user holds at their exchange."""
    settings = signed_settings()
    if settings is None:
        return None
    url, request = _signed(settings, BALANCE_PATHS[settings.name])
    payload = _fetch(url, request)
    if not payload:
        return None
    return Balance(settings.name, _balance_amount(settings.name, payload), "BTC", url)


# --- recent orders: signed, read-only ---

ORDER_PATHS = {
    "kraken": "/0/private/Query2",
    "coinbase": "/v2/transactions",
    "bitstamp": "/api/v2/instantsettlement/order_list/",
}


def _order_rows(name: str, payload: dict) -> list[dict]:
    if name == "kraken":
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        rows = result.get("query2") or result.get("trades") or {}
        if not isinstance(rows, dict):
            return [row for row in rows if isinstance(row, dict)]
        out: list[dict] = []
        for key, row in rows.items():
            if isinstance(row, dict):
                out.append({**row, "id": str(row.get("ordertxid") or key)})
        return out
    if name == "coinbase":
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]
    return [row for row in (payload.get("orders") or payload.get("data") or [])
            if isinstance(row, dict)]


def _order(name: str, row: dict) -> Order:
    amount = row.get("amount")
    if isinstance(amount, dict):
        amount = amount.get("amount")
    pair = str(row.get("pair") or row.get("currency") or "")
    if not pair and isinstance(row.get("amount"), dict):
        pair = str(row["amount"].get("currency") or "")
    return Order(
        exchange=name,
        id=str(row.get("id") or row.get("txid") or row.get("ordertxid")
                or row.get("order_id") or ""),
        side=str(row.get("side") or row.get("type") or "").casefold(),
        pair=pair or "XBT",
        volume=_number(row.get("volume") or row.get("amount_total") or amount),
        price=_number(row.get("price") or row.get("value")),
        when=str(row.get("when") or row.get("timestamp") or row.get("time") or ""),
    )


def recent_orders(limit: int = 5) -> list[Order] | None:
    """The most recent orders at the exchange, newest first, capped at ``limit``."""
    settings = signed_settings()
    if settings is None:
        return None
    url, request = _signed(settings, ORDER_PATHS[settings.name])
    payload = _fetch(url, request)
    if not payload:
        return None
    rows = _order_rows(settings.name, payload)
    orders = [_order(settings.name, row) for row in rows]
    return orders[: max(1, int(limit))]


# --- spoken forms ---

def display_name(name: str) -> str:
    return {"kraken": "Kraken", "coinbase": "Coinbase", "bitstamp": "Bitstamp"}.get(name, name)


def _currency_for_speech(code: str) -> str:
    return _CURRENCY_WORDS.get(code.upper(), code.upper())


def spoken_price(quote: Quote) -> str:
    return (f"Bitcoin is {_number(quote.price):,.2f} "
            f"{_currency_for_speech(quote.currency)} at {display_name(quote.exchange)}.")


def spoken_balance(record: Balance) -> str:
    amount = f"{record.amount:,.8f}".rstrip("0").rstrip(".")
    return (f"You hold {amount or '0'} Bitcoin at "
            f"{display_name(record.exchange)}.")


def spoken_orders(orders: list[Order]) -> str:
    if not orders:
        return "You have no recent orders at your exchange."
    parts = []
    for order in orders:
        volume = f"{order.volume:,.8f}".rstrip("0").rstrip(".")
        piece = f"{order.side or 'order'} {volume or '0'} {order.pair}"
        if order.price > 0:
            piece += f" at {order.price:,.2f}"
        parts.append(piece)
    where = display_name(orders[0].exchange)
    head = ("Your most recent order" if len(orders) == 1
            else f"Your {len(orders)} most recent orders")
    return f"{head} at {where}: " + "; ".join(parts) + "."
