"""Personal knowledge base: remember, recall, list and forget facts end to end."""

from __future__ import annotations

from datetime import date

from voxa.agent.intents import ToolCall, route
from voxa.agent.memory_facts import add, clean_fact, forget, search, spoken_all, spoken_recall
from voxa.agent.tools import default_registry
from voxa.agent.tools.memory_facts import (
    _forget_fact_handler,
    _list_facts_handler,
    _recall_fact_handler,
    _remember_fact_handler,
)


def test_clean_fact() -> None:
    assert clean_fact("  that the wifi password is on the fridge. ") == "The wifi password is on the fridge"
    assert clean_fact("my car registration expires in May") == "My car registration expires in May"


def test_add_dedupes() -> None:
    content = "- 2026-10-10 | The spare key is under the blue pot\n"
    again = add(content, date(2026, 10, 11), "the spare key is under the blue pot")
    assert again == content
    added = add(content, date(2026, 10, 11), "my car registration expires in May")
    assert added == content + "- 2026-10-11 | My car registration expires in May\n"


def test_search_ranking() -> None:
    items = [
        "The spare key is under the blue pot",
        "My car registration expires in May",
        "The wifi password is on the fridge",
    ]
    assert search(items, "the spare key") == ["The spare key is under the blue pot"]
    assert search(items, "blue pot key") == ["The spare key is under the blue pot"]
    assert search(items, "what about the car and may") == ["My car registration expires in May"]
    assert search(items, "nothing relevant") == []


def test_spoken_recall() -> None:
    assert spoken_recall([], "the spare key") == "I don't have anything about the spare key."
    assert spoken_recall(["The spare key is under the blue pot"], "key") == "You told me: The spare key is under the blue pot."
    assert (
        spoken_recall(["A fact", "B fact"], "topic") == "You told me: A fact. Also: B fact."
    )


def test_spoken_all() -> None:
    assert spoken_all([]) == "I don't remember anything yet."
    assert spoken_all(["One", "Two", "Three"]) == "I remember 3 things. The latest: Three."


def test_forget() -> None:
    content = "- 2026-10-10 | The spare key is under the blue pot\n- 2026-10-11 | My car registration expires in May\n"
    new, removed = forget(content, "the spare key")
    assert removed == "The spare key is under the blue pot"
    assert new == "- 2026-10-11 | My car registration expires in May\n"
    same, none = forget(content, "nothing here")
    assert none is None and same == content


def test_memory_phrases_route() -> None:
    assert route("remember that the spare key is under the blue pot") == ToolCall(
        "remember_fact", {"fact": "the spare key is under the blue pot"}
    )
    assert route("don't forget the milk is cold") == ToolCall("remember_fact", {"fact": "the milk is cold"})
    assert route("keep in mind that the gate is locked") == ToolCall("remember_fact", {"fact": "the gate is locked"})
    assert route("what do you remember about the spare key") == ToolCall("recall_fact", {"topic": "the spare key"})
    assert route("what do you know about my car") == ToolCall("recall_fact", {"topic": "my car"})
    assert route("do you remember anything about the wifi") == ToolCall("recall_fact", {"topic": "the wifi"})
    assert route("what did i tell you about the pot") == ToolCall("recall_fact", {"topic": "the pot"})
    assert route("what do you remember") == ToolCall("list_facts", {})
    assert route("what have i told you") == ToolCall("list_facts", {})
    assert route("show me your memory") == ToolCall("list_facts", {})
    assert route("forget about the spare key") == ToolCall("forget_fact", {"topic": "the spare key"})
    assert route("forget that the gate is locked") == ToolCall("forget_fact", {"topic": "the gate is locked"})


def test_memory_rules_do_not_steal_other_commands() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_remember_to_is_left_unrouted() -> None:
    assert route("remember to buy milk") is None


def test_anniversary_still_routes_to_dates() -> None:
    assert route("remember that Sam's anniversary is June 12") == ToolCall(
        "date_remember", {"kind": "anniversary", "name": "Sam", "when": "June 12"}
    )


def test_memory_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.memory_facts._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.memory_facts._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _remember_fact_handler({"fact": "the spare key is under the blue pot"})
    assert added.ok and added.speech == "I'll remember that."
    assert "spare key" in store["content"]
    _remember_fact_handler({"fact": "my car registration expires in May"})
    recalled = _recall_fact_handler({"topic": "the spare key"})
    assert recalled.ok and "blue" in recalled.speech
    listed = _list_facts_handler({})
    assert listed.ok and listed.speech == "I remember 2 things. The latest: My car registration expires in May."
    forgotten = _forget_fact_handler({"topic": "the spare key"})
    assert forgotten.ok and forgotten.speech == "Forgotten: The spare key is under the blue pot."
    assert "spare key" not in store["content"]
    missing = _forget_fact_handler({"topic": "the moon"})
    assert not missing.ok and missing.speech == "I don't have anything about the moon."


def test_registry_has_memory_tools() -> None:
    names = default_registry().names()
    for name in ("remember_fact", "recall_fact", "list_facts", "forget_fact"):
        assert name in names
