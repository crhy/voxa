"""The exchange client itself: URLs, signatures, parsing, spoken forms, the refusal.

Everything runs against a fake server: ``voxa.agent.exchange.fetch_json`` is the
injectable transport (same idea as tools/lookups.py's ``fetch``), so no test here
opens a socket.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import urllib.error
import urllib.parse
from contextlib import contextmanager
from unittest.mock import patch

from voxa.agent import exchange
from voxa.config import Settings

SECRET = base64.b64encode(b"topsecret").decode()


class FakeServer:
    """Answers each known path with a canned payload and records the request."""

    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, request: dict) -> dict:
        self.calls.append((url, request))
        for path, payload in self.payloads.items():
            if path in url:
                return payload
        raise AssertionError(f"fake server got an unexpected url: {url}")


@contextmanager
def fake_server(payloads: dict[str, dict]):
    server = FakeServer(payloads)
    with patch("voxa.agent.exchange.fetch_json", server):
        yield server


@contextmanager
def configured(name: str = "kraken", key: str = "PUBKEY", secret: str = SECRET):
    """Patch the settings store so the exchange looks set up."""
    settings = Settings(exchange_name=name, exchange_key=key, exchange_secret=secret)
    with patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = settings
        yield


@contextmanager
def unconfigured():
    """Patch the settings store so nothing is set up."""
    with patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = Settings()
        yield


KRAKEN_TICKER = {"error": [], "result": {"XBTUSD": {
    "a": ["134.00000000 - Bitcoin / USD - Daily"],
    "b": ["134.56000000 - Bitcoin / USD - Daily"],
    "c": ["134.55000000 - Bitcoin / USD - Daily"],
}}}
COINBASE_SPOT = {"data": {"amount": "134.560000", "base": "BTC", "symbol": "USD"}}
BITSTAMP_TICKER = {"high": "135.00", "low": "134.00", "last": "134.56", "volume": "12.00"}

KRAKEN_BALANCE = {"error": [], "result": {"XBT": "1.23456789"}}
COINBASE_BALANCE = {"data": {"amount": "2.50000000", "account": "BTC"}}
BITSTAMP_BALANCE = {"balance": "0.50000000", "account_handle": "handle-1"}

KRAKEN_ORDERS = {"error": [], "result": {"query2": {
    "OTR6D-AVXQZ-OYXKT2": {
        "ordertxid": "OTR6D-AVXQZ-OYXKT2",
        "postxid": "73575766",
        "time": "1760000000.0000",
        "type": "buy",
        "pair": "XBTUSD",
        "price": "134.56000000",
        "volume": "0.50000000",
    },
}}}
COINBASE_ORDERS = {"data": [
    {"id": "tx-1", "type": "send", "input": False,
     "amount": {"amount": "0.25000000", "currency": "BTC"}},
]}
BITSTAMP_ORDERS = {"orders": [
    {"id": "order-1", "amount": "0.10000000", "currency": "BTC",
     "status": "accepted", "account_handle": "handle-1"},
]}


# --- configuration ---

def test_client_settings_reads_the_profile() -> None:
    with configured("bitstamp"):
        settings = exchange.client_settings()
    assert settings == exchange.ExchangeSettings("bitstamp", "PUBKEY", SECRET)


def test_price_and_balance_are_none_when_unconfigured() -> None:
    with unconfigured():
        assert exchange.client_settings() is None
        assert exchange.signed_settings() is None
        assert exchange.price() is None
        assert exchange.balance() is None
        assert exchange.recent_orders() is None


def test_an_exchange_the_code_does_not_know_is_not_configured() -> None:
    with configured("binance"):
        assert exchange.client_settings() is None


def test_signed_reads_need_the_key() -> None:
    with configured("kraken", key="", secret=""):
        assert exchange.client_settings() is not None
        assert exchange.signed_settings() is None


# --- price: the public ticker, no key needed ---

def test_price_uses_the_kraken_public_ticker() -> None:
    with fake_server({"/0/public/Ticker": KRAKEN_TICKER}) as server, \
            configured("kraken", key="", secret=""):
        quote = exchange.price()
    assert quote == exchange.Quote(
        "kraken", "USD", 134.56, "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
    )
    assert server.calls[0][1]["method"] == "GET"


def test_price_asks_the_ticker_for_the_spoken_currency() -> None:
    with fake_server({"/api/v2/ticker/": BITSTAMP_TICKER}) as server, \
            configured("bitstamp"):
        quote = exchange.price("GBP")
    assert quote.currency == "GBP"
    assert quote.price == 134.56
    assert server.calls[0][0] == "https://www.bitstamp.net/api/v2/ticker/btcgbp/"


def test_price_falls_back_to_usd_for_a_currency_the_tickers_lack() -> None:
    with fake_server({"/v2/prices/": COINBASE_SPOT}) as server, \
            configured("coinbase"):
        quote = exchange.price("SGK")
    assert quote.currency == "USD"
    assert server.calls[0][0] == "https://api.coinbase.com/v2/prices/BTC-USD/spot"


def test_price_parses_the_coinbase_spot_answer() -> None:
    with fake_server({"/v2/prices/": {"data": {"amount": "123.450000"}}}), \
            configured("coinbase"):
        assert exchange.price("EUR").price == 123.45


# --- balance: signed, read-only ---

def test_balance_posts_a_signed_kraken_request() -> None:
    with fake_server({"/0/private/Balance": KRAKEN_BALANCE}) as server, \
            configured("kraken"):
        record = exchange.balance()
    url, request = server.calls[0]
    assert url == "https://api.kraken.com/0/private/Balance"
    assert request["method"] == "POST"
    form = dict(urllib.parse.parse_qsl(request["data"]))
    assert form["key"] == "PUBKEY"
    nonce = form["nonce"]
    expected = base64.b64encode(
        hmac.new(base64.b64decode(SECRET),
                 b"/0/private/Balance" + nonce.encode(), hashlib.sha256).digest()
    ).decode()
    assert form["signature"] == expected
    assert record == exchange.Balance("kraken", 1.23456789, "BTC", url)


def test_balance_signs_a_coinbase_get() -> None:
    with fake_server({"/v2/accounts/BTC/balance": COINBASE_BALANCE}) as server, \
            configured("coinbase", key="CBKEY", secret="cbsecret"):
        record = exchange.balance()
    url, request = server.calls[0]
    headers = request["headers"]
    assert url == "https://api.coinbase.com/v2/accounts/BTC/balance"
    assert headers["CB-ACCESS-KEY"] == "CBKEY"
    timestamp = headers["CB-ACCESS-TIMESTAMP"]
    expected = hmac.new(b"cbsecret",
                        timestamp.encode() + b"/v2/accounts/BTC/balance",
                        hashlib.sha256).hexdigest()
    assert headers["CB-ACCESS-SIGN"] == expected
    assert record.amount == 2.5


def test_balance_signs_a_bitstamp_get() -> None:
    with fake_server({"/api/v2/instantsettlement/balance/": BITSTAMP_BALANCE}) as server, \
            configured("bitstamp", key="BSKEY", secret="bssecret"):
        record = exchange.balance()
    url, request = server.calls[0]
    headers = request["headers"]
    assert url == "https://www.bitstamp.net/api/v2/instantsettlement/balance/"
    nonce = headers["Bitstamp-Nonce"]
    expected = hmac.new(b"bssecret",
                        b"/api/v2/instantsettlement/balance/" + nonce.encode(),
                        hashlib.sha256).hexdigest()
    assert headers["Bitstamp-Signature"] == expected
    assert record.amount == 0.5


def test_balance_never_calls_the_exchange_without_a_key() -> None:
    with fake_server({"/0/private/Balance": KRAKEN_BALANCE}) as server, \
            configured("kraken", key="", secret=""):
        assert exchange.balance() is None
    assert server.calls == []


# --- recent orders: signed, read-only ---

def test_recent_orders_parses_kraken_query2() -> None:
    with fake_server({"/0/private/Query2": KRAKEN_ORDERS}), configured("kraken"):
        orders = exchange.recent_orders()
    assert orders == [exchange.Order(
        id="OTR6D-AVXQZ-OYXKT2", side="buy", pair="XBTUSD", volume=0.5,
        price=134.56, when="1760000000.0000", exchange="kraken",
    )]


def test_recent_orders_parses_coinbase_transactions() -> None:
    with fake_server({"/v2/transactions": COINBASE_ORDERS}), configured("coinbase"):
        orders = exchange.recent_orders()
    assert orders[0].id == "tx-1"
    assert orders[0].volume == 0.25
    assert orders[0].pair == "BTC"


def test_recent_orders_parses_bitstamp_order_list() -> None:
    with fake_server({"/api/v2/instantsettlement/order_list/": BITSTAMP_ORDERS}), \
            configured("bitstamp"):
        orders = exchange.recent_orders()
    assert orders[0].id == "order-1"
    assert orders[0].volume == 0.1


def test_recent_orders_caps_at_the_limit() -> None:
    many = {"orders": [{"id": f"order-{i}", "amount": "1.0", "currency": "BTC"}
                       for i in range(9)]}
    with fake_server({"/api/v2/instantsettlement/order_list/": many}), \
            configured("bitstamp"):
        assert len(exchange.recent_orders(3)) == 3


def test_an_empty_order_list_is_not_a_failure() -> None:
    with fake_server({"/0/private/Query2": {"error": [], "result": {"query2": {}}}}), \
            configured("kraken"):
        assert exchange.recent_orders() == []


# --- failures are swallowed, never raised ---

def test_an_unreachable_exchange_returns_none() -> None:
    def down(url: str, request: dict) -> dict:
        raise urllib.error.URLError("no route")

    with patch("voxa.agent.exchange.fetch_json", down), configured("kraken"):
        assert exchange.price() is None
        assert exchange.balance() is None
        assert exchange.recent_orders() is None


def test_a_malformed_answer_returns_none() -> None:
    with fake_server({"/0/public/Ticker": {"error": ["EInvalid"]}}), configured("kraken"):
        assert exchange.price() is None


def test_a_non_dict_answer_is_ignored() -> None:
    with fake_server({"/0/private/Balance": []}), configured("kraken"):
        assert exchange.balance() is None


# --- spoken forms ---

def test_spoken_price() -> None:
    quote = exchange.Quote("kraken", "GBP", 134.56)
    assert exchange.spoken_price(quote) == "Bitcoin is 134.56 British pounds at Kraken."


def test_spoken_balance_drops_trailing_zeros() -> None:
    assert exchange.spoken_balance(
        exchange.Balance("kraken", 1.23456789, "BTC")
    ) == "You hold 1.23456789 Bitcoin at Kraken."
    assert exchange.spoken_balance(
        exchange.Balance("coinbase", 0.0, "BTC")
    ) == "You hold 0 Bitcoin at Coinbase."


def test_spoken_orders() -> None:
    orders = [exchange.Order("OTR6D-AVXQZ-OYXKT2", "buy", "XBTUSD", 0.5, 134.56,
                             "", "kraken")]
    assert exchange.spoken_orders(orders) == (
        "Your most recent order at Kraken: buy 0.5 XBTUSD at 134.56."
    )
    assert exchange.spoken_orders([]) == "You have no recent orders at your exchange."
    two = orders + [exchange.Order("OTR6D-AVXQZ-OYXKT3", "sell", "XBTUSD", 1, 130, "", "kraken")]
    assert exchange.spoken_orders(two) == (
        "Your 2 most recent orders at Kraken: buy 0.5 XBTUSD at 134.56; "
        "sell 1 XBTUSD at 130.00."
    )


# --- the buy refusal ---

def test_the_buy_refusal_names_issue_43() -> None:
    assert "#43" in exchange.BUY_REFUSAL
    assert "won't buy" in exchange.BUY_REFUSAL


def test_the_not_configured_message_names_the_key_restrictions() -> None:
    assert "withdrawals disabled" in exchange.NOT_CONFIGURED
    for name in ("Kraken", "Coinbase", "Bitstamp"):
        assert name in exchange.NOT_CONFIGURED
