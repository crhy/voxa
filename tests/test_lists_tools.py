"""Voice commands for to-do and shopping lists route and work end to end."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.lists import (
    _list_add_handler,
    _list_clear_handler,
    _list_done_handler,
    _list_read_handler,
    _list_remove_handler,
)


def test_list_phrases_route() -> None:
    assert route("add milk to my shopping list") == ToolCall("list_add", {"item": "milk", "list": "shopping"})
    assert route("put eggs on the shopping list") == ToolCall("list_add", {"item": "eggs", "list": "shopping"})
    assert route("add call the vet to my to-do list") == ToolCall("list_add", {"item": "call the vet", "list": "to-do"})
    assert route("what's on my shopping list") == ToolCall("list_read", {"list": "shopping"})
    assert route("read my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("check off milk on my shopping list") == ToolCall("list_done", {"item": "milk", "list": "shopping"})
    assert route("mark call the vet as done") == ToolCall("list_done", {"item": "call the vet", "list": ""})
    assert route("remove eggs from my shopping list") == ToolCall("list_remove", {"item": "eggs", "list": "shopping"})
    assert route("clear my shopping list") == ToolCall("list_clear", {"list": "shopping"})


def test_list_rules_do_not_steal_notes_or_desktop() -> None:
    assert route("read my notes") == ToolCall("read_notes", {})
    assert route("show the desktop") == ToolCall("minimize_all", {})


def test_list_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.lists._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _list_add_handler({"item": "milk", "list": "shopping"})
    assert added.ok and added.speech == "I added Milk to your shopping list."
    assert _list_add_handler({"item": "eggs", "list": "shopping"}).ok
    read = _list_read_handler({"list": "shopping"})
    assert read.speech == "You have 2 things on your shopping list: Milk and Eggs."
    done = _list_done_handler({"item": "milk", "list": "shopping"})
    assert done.ok and done.speech == "Done: Milk."
    removed = _list_remove_handler({"item": "eggs", "list": "shopping"})
    assert removed.ok and removed.speech == "I removed Eggs from your shopping list."
    cleared = _list_clear_handler({"list": "shopping"})
    assert cleared.ok and cleared.speech == "I cleared your shopping list."


def test_list_add_rejects_empty_item() -> None:
    result = _list_add_handler({"item": "  ", "list": "shopping"})
    assert not result.ok


def test_registry_has_list_tools() -> None:
    names = default_registry().names()
    for name in ("list_add", "list_read", "list_done", "list_remove", "list_clear"):
        assert name in names


def test_done_without_a_list_name_looks_on_every_list(monkeypatch):
    from voxa.agent.tools import lists as list_tools_module

    store = {"text": "## Shopping\n- [ ] Milk\n"}
    monkeypatch.setattr(list_tools_module, "_read", lambda: store["text"])
    monkeypatch.setattr(list_tools_module, "_write", lambda content: store.__setitem__("text", content))
    result = list_tools_module._list_done_handler({"item": "milk", "list": ""})
    assert result.ok and result.speech == "Done: Milk."
    assert store["text"] == "## Shopping\n- [x] Milk\n"


def test_only_real_list_names_are_lists():
    from voxa.agent.intents import route

    for said in ("what's on tv", "what's on my calendar", "put the kettle on the stove"):
        call = route(said)
        assert call is None or not call.tool.startswith("list_"), said
