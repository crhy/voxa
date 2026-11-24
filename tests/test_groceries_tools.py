"""Groceries/food delivery routes to grocery_order and returns an order read-back."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.groceries import _grocery_order_handler


def test_order_and_delivery_phrases_route() -> None:
    assert route("order milk and eggs for delivery") == ToolCall(
        "grocery_order", {"items": "milk and eggs", "store": ""}
    )
    assert route("get my groceries delivered from instacart") == ToolCall(
        "grocery_order", {"items": "groceries", "store": "instacart"}
    )
    assert route("order milk from instacart") == ToolCall(
        "grocery_order", {"items": "milk", "store": "instacart"}
    )


def test_grocery_order_does_not_steal_list_or_travel_intents() -> None:
    assert route("add milk to my shopping list") == ToolCall(
        "list_add", {"item": "milk", "list": "shopping"}
    )
    assert route("add milk to my grocery list") == ToolCall(
        "list_add", {"item": "milk", "list": "grocery"}
    )
    assert route("what's on my shopping list") == ToolCall("list_read", {"list": "shopping"})
    assert route("my flight to Denver was cancelled") == ToolCall(
        "travel_options", {"what": "flight", "place": "Denver", "problem": "cancelled"}
    )


def test_grocery_order_handler_with_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.groceries.websearch.search",
        lambda q: [
            websearch_result("Instacart", "Grocery delivery from local stores"),
            websearch_result("Amazon Fresh", "Same-day grocery delivery"),
        ],
    )
    monkeypatch.setattr(
        "voxa.agent.tools.groceries.ask_model",
        lambda messages: (
            "1. Order from Instacart.\n2. Use Amazon Fresh for same-day delivery.\n3. Pick up curbside."
        ),
    )
    result = _grocery_order_handler({"items": "milk, eggs and bread", "store": "instacart"})
    assert result.ok
    assert "milk" in result.speech
    assert "eggs" in result.speech
    assert "bread" in result.speech
    assert "Instacart" in result.speech
    assert "Estimated total" in result.speech


def test_grocery_order_handler_fallback_when_search_fails(monkeypatch) -> None:
    def boom(q: str):
        raise RuntimeError("no network")

    monkeypatch.setattr("voxa.agent.tools.groceries.websearch.search", boom)
    monkeypatch.setattr("voxa.agent.tools.groceries.ask_model", None)
    result = _grocery_order_handler({"items": "milk and eggs", "store": ""})
    assert result.ok
    assert "milk" in result.speech
    assert "eggs" in result.speech
    assert "Estimated total" in result.speech
    assert "grocery-delivery" in result.speech or "delivery" in result.speech


def test_grocery_order_handler_rejects_missing_items() -> None:
    result = _grocery_order_handler({"items": "", "store": "instacart"})
    assert not result.ok


def test_registry_has_grocery_order() -> None:
    assert "grocery_order" in default_registry().names()


def websearch_result(title: str, snippet: str):
    from voxa.websearch import SearchResult

    return SearchResult(title=title, url="https://example.com", snippet=snippet)
