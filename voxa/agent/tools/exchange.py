"""Bitcoin tools on top of the user's own exchange (issue #46, read-only slice).

Everything here reads: the price comes from the exchange's public ticker, the
balance and the recent orders are signed read requests. Nothing in this module
trades, withdraws, deposits or sends funds, and it never will: ``bitcoin_buy``
exists only so the phrase is recognised, and it answers
``voxa.agent.exchange.BUY_REFUSAL`` without a network call, because the rule
that decides how much may be spent is the open decision recorded in issue #43.
The key in ``Settings.exchange_key`` is required to be trade-only with
withdrawals disabled; see the docstring in voxa/agent/exchange.py.
"""

from __future__ import annotations

from voxa.agent import exchange
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

NOT_CONFIGURED = exchange.NOT_CONFIGURED


def _name() -> str:
    settings = exchange.client_settings()
    return exchange.display_name(settings.name) if settings is not None else "your exchange"


def _unreachable(what: str) -> str:
    return f"I couldn't reach {_name()} for the Bitcoin {what} right now."


def bitcoin_price(args: dict[str, str]) -> ToolResult:
    if exchange.client_settings() is None:
        return ToolResult.failure(NOT_CONFIGURED)
    quote = exchange.price(args.get("currency") or "")
    if quote is None:
        return ToolResult.failure(_unreachable("price"), detail=_name())
    return ToolResult.success(exchange.spoken_price(quote), detail=quote.detail)


def bitcoin_balance(args: dict[str, str]) -> ToolResult:
    if exchange.signed_settings() is None:
        return ToolResult.failure(NOT_CONFIGURED)
    record = exchange.balance()
    if record is None:
        return ToolResult.failure(_unreachable("balance"), detail=_name())
    return ToolResult.success(exchange.spoken_balance(record), detail=record.detail)


def bitcoin_orders(args: dict[str, str]) -> ToolResult:
    if exchange.signed_settings() is None:
        return ToolResult.failure(NOT_CONFIGURED)
    limit = args.get("limit") or ""
    try:
        count = max(1, int(limit)) if limit else 5
    except ValueError:
        count = 5
    orders = exchange.recent_orders(count)
    if orders is None:
        return ToolResult.failure(_unreachable("orders"), detail=_name())
    return ToolResult.success(exchange.spoken_orders(orders), detail=_name())


def bitcoin_buy(args: dict[str, str]) -> ToolResult:
    """Refuses, and always will until issue #43 decides the spending rule.

    Reads no settings and makes no network call: the buy path is closed, not
    limited, so there is no per-order limit to apply and no order to log.
    """
    return ToolResult.failure(exchange.BUY_REFUSAL)


def exchange_tools() -> list[Tool]:
    return [
        Tool(
            name="bitcoin_price",
            description="Speak the current Bitcoin price from your exchange's public ticker.",
            parameters={"currency": "USD, CAD, GBP, EUR, JPY, AUD or CHF; empty for your usual one"},
            risk=RiskLevel.READ_ONLY,
            handler=bitcoin_price,
        ),
        Tool(
            name="bitcoin_balance",
            description="Speak how much Bitcoin you hold at your exchange.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=bitcoin_balance,
        ),
        Tool(
            name="bitcoin_orders",
            description="Speak your most recent orders at your exchange.",
            parameters={"limit": "how many orders to read out; empty for five"},
            risk=RiskLevel.READ_ONLY,
            handler=bitcoin_orders,
        ),
        Tool(
            name="bitcoin_buy",
            description="Refused: buying Bitcoin waits for the spending rule in issue #43.",
            parameters={"amount": "the currency amount you asked to buy"},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=bitcoin_buy,
        ),
    ]
