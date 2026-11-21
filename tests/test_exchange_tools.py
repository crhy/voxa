"""The Bitcoin tools: routing, handlers against a fake server, and the guards
that keep this slice read-only.

The fake server is the same trick tests/test_homeassistant.py uses with ``open_url``:
``voxa.agent.exchange.fetch_json`` is patched, so no test reaches the network.
"""

from __future__ import annotations

import ast
import base64
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from voxa.agent import exchange
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.exchange import (
    bitcoin_balance,
    bitcoin_buy,
    bitcoin_orders,
    bitcoin_price,
)
from voxa.config import Settings

SECRET = base64.b64encode(b"topsecret").decode()

KRAKEN_TICKER = {"error": [], "result": {"XBTUSD": {"b": ["134.56000000 - Bitcoin / USD"]}}}
KRAKEN_BALANCE = {"error": [], "result": {"XBT": "1.23456789"}}
KRAKEN_ORDERS = {"error": [], "result": {"query2": {
    "OTR6D-AVXQZ-OYXKT2": {"ordertxid": "OTR6D-AVXQZ-OYXKT2", "type": "buy",
                           "pair": "XBTUSD", "price": "134.56000000",
                           "volume": "0.50000000"},
}}}
ALL_PAYLOADS = {
    "/0/public/Ticker": KRAKEN_TICKER,
    "/0/private/Balance": KRAKEN_BALANCE,
    "/0/private/Query2": KRAKEN_ORDERS,
}


class FakeServer:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def __call__(self, url: str, request: dict) -> dict:
        self.calls.append(url)
        for path, payload in self.payloads.items():
            if path in url:
                return payload
        raise AssertionError(f"fake server got an unexpected url: {url}")


@contextmanager
def fake_server(payloads: dict[str, dict] | None = None):
    server = FakeServer(ALL_PAYLOADS if payloads is None else payloads)
    with patch("voxa.agent.exchange.fetch_json", server):
        yield server


@contextmanager
def configured(name: str = "kraken", key: str = "PUBKEY", secret: str = SECRET):
    settings = Settings(exchange_name=name, exchange_key=key, exchange_secret=secret)
    with patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = settings
        yield


@contextmanager
def unconfigured():
    with patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = Settings()
        yield


# --- routing: the exchange answers Bitcoin, the market rule keeps everything else ---

def test_router_bitcoin_price() -> None:
    assert route("Voxa, what's Bitcoin at?") == ToolCall("bitcoin_price", {"currency": ""})
    assert route("how much is bitcoin") == ToolCall("bitcoin_price", {"currency": ""})
    assert route("what is the price of Bitcoin") == ToolCall("bitcoin_price", {"currency": ""})


def test_router_bitcoin_price_in_a_spoken_currency() -> None:
    assert route("what's Bitcoin at in euros") == ToolCall("bitcoin_price", {"currency": "EUR"})
    assert route("how much is bitcoin in British pounds") == ToolCall(
        "bitcoin_price", {"currency": "GBP"}
    )


def test_router_bitcoin_balance() -> None:
    assert route("Voxa, how much Bitcoin do I have?") == ToolCall("bitcoin_balance", {})
    assert route("what's my Bitcoin balance") == ToolCall("bitcoin_balance", {})


def test_router_bitcoin_orders() -> None:
    assert route("show my recent exchange orders") == ToolCall("bitcoin_orders", {})
    assert route("show recent bitcoin trades") == ToolCall("bitcoin_orders", {})


def test_router_bitcoin_buy_parses_the_amount_and_still_routes_to_the_refusal() -> None:
    assert route("Voxa, buy fifty dollars of Bitcoin") == ToolCall("bitcoin_buy", {"amount": "50"})
    assert route("buy 10 dollars of bitcoin") == ToolCall("bitcoin_buy", {"amount": "10"})


def test_the_market_rule_still_gets_gold_silver_and_companies() -> None:
    assert route("how much is gold").tool == "market"
    assert route("check the stock market").tool == "market"
    assert route("what is Tesla stock price").tool == "market"
    assert route("what is the British pound in USD").tool == "exchange_rate"


# --- handlers ---

def test_bitcoin_price_speech() -> None:
    with fake_server() as server, configured():
        result = bitcoin_price({"currency": ""})
    assert result.ok
    assert result.speech == "Bitcoin is 134.56 U.S. dollars at Kraken."
    assert result.detail == "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
    assert server.calls == ["https://api.kraken.com/0/public/Ticker?pair=XBTUSD"]


def test_bitcoin_balance_speech() -> None:
    with fake_server(), configured():
        result = bitcoin_balance({})
    assert result.ok
    assert result.speech == "You hold 1.23456789 Bitcoin at Kraken."


def test_bitcoin_orders_speech() -> None:
    with fake_server(), configured():
        result = bitcoin_orders({"limit": "3"})
    assert result.ok
    assert result.speech == "Your most recent order at Kraken: buy 0.5 XBTUSD at 134.56."


def test_bitcoin_price_not_configured() -> None:
    with unconfigured(), fake_server() as server:
        result = bitcoin_price({})
    assert not result.ok
    assert result.speech == exchange.NOT_CONFIGURED
    assert server.calls == []


def test_bitcoin_balance_and_orders_not_configured_without_a_key() -> None:
    with configured("kraken", key="", secret=""), fake_server() as server:
        assert bitcoin_balance({}).speech == exchange.NOT_CONFIGURED
        assert bitcoin_orders({}).speech == exchange.NOT_CONFIGURED
    assert server.calls == []


def test_bitcoin_price_unreachable_exchange() -> None:
    def down(url: str, request: dict) -> dict:
        raise urllib.error.URLError("no route")

    with patch("voxa.agent.exchange.fetch_json", down), configured():
        result = bitcoin_price({})
    assert not result.ok
    assert result.speech == "I couldn't reach Kraken for the Bitcoin price right now."


def test_the_key_never_appears_in_any_answer() -> None:
    with fake_server(), configured("kraken", key="SECRETKEY", secret=SECRET):
        for handler, args in ((bitcoin_price, {"currency": ""}),
                              (bitcoin_balance, {}),
                              (bitcoin_orders, {"limit": ""})):
            result = handler(args)
            assert "SECRETKEY" not in result.speech + result.detail
            assert SECRET not in result.speech + result.detail


def test_the_registry_exposes_the_tools_with_their_risk_levels() -> None:
    registry = default_registry()
    for name in ("bitcoin_price", "bitcoin_balance", "bitcoin_orders"):
        assert registry.get(name).risk.name == "READ_ONLY"
    assert registry.get("bitcoin_buy").risk.name == "EXTERNAL_WRITE"


def test_the_registry_refuses_a_buy_too() -> None:
    with fake_server(), configured():
        result = default_registry().call("bitcoin_buy", {"amount": "50"})
    assert not result.ok
    assert result.speech == exchange.BUY_REFUSAL


# --- guards: this slice never trades, withdraws or sends funds ---

def test_bitcoin_buy_makes_no_network_call_at_all() -> None:
    server = FakeServer(ALL_PAYLOADS)
    with patch("voxa.agent.exchange.fetch_json", server), \
            patch("urllib.request.urlopen", side_effect=AssertionError("network")), \
            configured("kraken", key="SECRETKEY", secret=SECRET):
        result = bitcoin_buy({"amount": "50"})
    assert result.speech == exchange.BUY_REFUSAL
    assert server.calls == []


def test_the_buy_refusal_reaches_the_user_through_the_registry() -> None:
    result = default_registry().call("bitcoin_buy", {"amount": "50"})
    assert not result.ok
    assert "#43" in result.speech


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def _attributes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]


# Every endpoint the slice is able to ask for, spelled out so a future edit that
# adds a trading, withdrawal or payment path fails here instead of silently shipping.
READ_ONLY_PATHS = {
    "/0/public/Ticker?pair=XBT", "/v2/prices/BTC-", "/spot", "/api/v2/ticker/btc",
    "/0/private/Balance", "/v2/accounts/BTC/balance",
    "/api/v2/instantsettlement/balance/",
    "/0/private/Query2", "/v2/transactions",
    "/api/v2/instantsettlement/order_list/",
}
SPEND_WORDS = ("withdraw", "deposit", "buy", "sell", "transfer", "payment",
               "checkout", "invoice", "send", "charge", "card", "wallet", "micro")


def test_the_only_paths_this_slice_can_request_are_read_only() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("voxa/agent/exchange.py", "voxa/agent/tools/exchange.py"):
        for constant in _string_constants(root / name):
            if not constant.startswith("/") or len(constant) < 2:
                continue  # a lone "/" is a path separator inside an f-string, not an endpoint
            assert constant in READ_ONLY_PATHS, f"{name} requests {constant}"
            for word in SPEND_WORDS:
                assert word not in constant.lower(), f"{name} mentions {word}"


def test_no_wallet_or_withdraw_api_is_reachable() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("voxa/agent/exchange.py", "voxa/agent/tools/exchange.py"):
        for attribute in _attributes(root / name):
            lowered = attribute.lower()
            for word in ("withdraw", "deposit", "transfer", "payment", "checkout",
                         "invoice", "charge", "card", "wallet", "micro"):
                assert word not in lowered, f"{name} calls {attribute}"


def test_the_slice_imports_nothing_outside_the_exchange_client() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("voxa/agent/exchange.py", "voxa/agent/tools/exchange.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module in {"__future__", "collections.abc", "dataclasses",
                                       "voxa.config", "voxa.agent",
                                       "voxa.agent.policy", "voxa.agent.registry",
                                       "voxa.agent.result"}, node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name in {"base64", "binascii", "hashlib", "hmac",
                                          "json", "logging", "time",
                                          "urllib.error", "urllib.parse",
                                          "urllib.request"}, alias.name

