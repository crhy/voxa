from __future__ import annotations

import json

from voxa.agent import lookups
from voxa.agent.tools import lookups as lookup_tools


def test_currency_names_and_stock_names_are_resolved():
    assert lookups.currency_code("the British pound") == "GBP"
    assert lookups.currency_code("brittish pounds") == "GBP"
    assert lookups.currency_code("U.S. dollars") == "USD"
    assert lookups.stock_target("Tesla stock") == ("TSLA", "Tesla")
    assert lookups.stock_target("Tesla's") == ("TSLA", "Tesla")
    assert lookups.stock_target("TSLA") == ("TSLA", "TSLA")
    assert lookups.stock_target("one share of Tesla") == ("TSLA", "Tesla")
    assert lookups.stock_target("one share of General Electric") == ("GE", "GE Aerospace")
    assert lookups.stock_target("one share of Toyota") == ("TM", "Toyota")
    assert lookups.stock_target("one share of Ford Motor Company") == ("F", "Ford")
    assert lookups.stock_target("one share of Mercedes") == ("MBG.DE", "Mercedes-Benz")
    assert lookups.stock_target("one Toyota share") == ("TM", "Toyota")
    assert lookups.stock_target("a Toyota share") == ("TM", "Toyota")
    assert lookups.stock_target("one share") is None
    assert lookups.stock_name("share price right now") == ""


def test_market_prompts_for_company_instead_of_quoting_one(monkeypatch):
    monkeypatch.setattr(lookup_tools, "fetch", lambda _url: (_ for _ in ()).throw(AssertionError("no fetch")))

    result = lookup_tools.market({"what": "share price"})

    assert not result.ok
    assert result.speech == "Which company? Say its name or stock ticker."


def test_market_speaks_numeric_tesla_quote(monkeypatch):
    payload = {
        "data": {
            "marketStatus": "After-Hours",
            "primaryData": {"lastSalePrice": "$374.6588"},
        }
    }
    monkeypatch.setattr(lookup_tools, "fetch", lambda url: json.dumps(payload))

    result = lookup_tools.market({"what": "Tesla stock"})

    assert result.ok
    assert result.speech == "Tesla stock is 374.66 U.S. dollars per share in after-hours trading."
    assert "TSLA" in result.detail


def test_exchange_rate_speaks_numeric_rate(monkeypatch):
    payload = {"date": "2026-10-13", "base": "GBP", "quote": "USD", "rate": 1.3245}
    monkeypatch.setattr(lookup_tools, "fetch", lambda url: json.dumps(payload))

    result = lookup_tools.exchange_rate({"base": "GBP", "quote": "USD"})

    assert result.ok
    assert result.speech == (
        "One British pound is 1.3245 U.S. dollars, using the reference rate for 2026-10-13."
    )
    assert "/gbp/usd" in result.detail


def test_quote_failures_are_spoken_without_crashing(monkeypatch):
    def unavailable(url):
        raise OSError("offline")

    monkeypatch.setattr(lookup_tools, "fetch", unavailable)

    assert not lookup_tools.market({"what": "Tesla stock"}).ok
    assert not lookup_tools.exchange_rate({"base": "GBP", "quote": "USD"}).ok


def test_market_resolves_an_unfamiliar_company_then_speaks_its_quote(monkeypatch):
    responses = iter([
        json.dumps({"quotes": [{
            "quoteType": "EQUITY",
            "symbol": "ORCL",
            "longname": "Oracle Corporation",
        }]}),
        "not Nasdaq JSON",
        json.dumps({"chart": {"result": [{"meta": {
            "instrumentType": "EQUITY",
            "regularMarketPrice": 281.125,
            "currency": "USD",
        }}]}}),
    ])
    monkeypatch.setattr(lookup_tools, "fetch", lambda url: next(responses))

    result = lookup_tools.market({"what": "one share of Oracle"})

    assert result.ok
    assert result.speech == "Oracle Corporation stock is 281.12 U.S. dollars per share."


def test_market_explains_that_private_company_has_no_stock_price(monkeypatch):
    monkeypatch.setattr(
        lookup_tools,
        "fetch",
        lambda url: (_ for _ in ()).throw(AssertionError("private companies need no lookup")),
    )

    result = lookup_tools.market({"what": "one share of General Atomics"})

    assert result.ok
    assert result.speech == (
        "General Atomics is privately held, so it does not have a public stock price."
    )
