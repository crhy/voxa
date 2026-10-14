"""Progress reports, pending counts and next-task suggestions route and speak correctly."""

from __future__ import annotations

from datetime import date

from voxa.agent.intents import ToolCall, route
from voxa.agent.lists import spoken_next, spoken_pending, spoken_progress
from voxa.agent.tools import default_registry
from voxa.agent.tools.lists import (
    _list_next_handler,
    _list_pending_handler,
    _list_progress_handler,
)


def test_spoken_progress_exact_strings() -> None:
    lists = {"To-do": [(True, "A"), (True, "B"), (False, "C"), (False, "D"), (False, "E")]}
    assert spoken_progress(lists) == "You have finished 2 of 5 things on your to-do list. Still open: C, D and E."
    assert spoken_progress({"To-do": [(True, "A"), (True, "B"), (True, "C")]}) == "Everything on your to-do list is done: 3 of 3."
    assert spoken_progress({}) == "Your to-do list is empty."
    assert spoken_progress({"To-do": []}) == "Your to-do list is empty."


def test_spoken_pending_exact_strings() -> None:
    lists = {"To-do": [(False, "A"), (False, "B"), (False, "C"), (True, "D")], "Shopping": [(False, "Milk")]}
    assert spoken_pending(lists) == "You have 4 things open: 3 on your to-do list and 1 on your shopping list."
    assert spoken_pending({}) == "Nothing is open. Well done."
    assert spoken_pending({"To-do": [(True, "A")]}) == "Nothing is open. Well done."


def test_spoken_next_exact_strings() -> None:
    lists = {"To-do": [(False, "Call the vet"), (False, "The report")]}
    assert spoken_next(lists, ["Report"]) == "Start with the report: it is due soon. After that: Call the vet."
    assert spoken_next(lists, []) == "Next on your list: Call the vet."
    assert spoken_next({"To-do": [(True, "A")]}, ["a"]) == "Nothing is waiting. Your to-do list is clear."
    assert spoken_next({}, []) == "Nothing is waiting. Your to-do list is clear."


def test_progress_phrases_route() -> None:
    assert route("how am i doing") == ToolCall("list_progress", {"list": ""})
    assert route("how am i doing on my to-do list") == ToolCall("list_progress", {"list": "to-do"})
    assert route("give me a progress report") == ToolCall("list_progress", {"list": ""})
    assert route("my progress report") == ToolCall("list_progress", {"list": ""})
    assert route("what have i finished today") == ToolCall("list_progress", {"list": ""})
    assert route("what's still open") == ToolCall("list_pending", {})
    assert route("what do i still have to do") == ToolCall("list_pending", {})
    assert route("what's left to do") == ToolCall("list_pending", {})
    assert route("what should i do next") == ToolCall("list_next", {})
    assert route("what's my next task") == ToolCall("list_next", {})
    assert route("what do i do next") == ToolCall("list_next", {})


def test_guard_phrases_stay_put() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
    assert route("what's my next appointment") == ToolCall("calendar_next", {})


def test_list_progress_handlers(monkeypatch) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 10, 10)

    monkeypatch.setattr("voxa.agent.tools.lists.date", FixedDate)
    store = {"content": "## To-do\n- [x] Call the vet\n- [ ] The report\n- [ ] Water the plants\n"}
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.lists._write",
        lambda content: store.__setitem__("content", content),
    )
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: "- 10-12 | deadline | Report\n")

    progress = _list_progress_handler({"list": ""})
    assert progress.ok and progress.speech == "You have finished 1 of 3 things on your to-do list. Still open: The report and Water the plants."
    pending = _list_pending_handler({})
    assert pending.ok and pending.speech == "You have 2 things open: 2 on your to-do list."
    nxt = _list_next_handler({})
    assert nxt.ok and nxt.speech == "Start with the report: it is due soon. After that: Water the plants."


def test_list_next_survives_dates_error(monkeypatch) -> None:
    store = {"content": "## To-do\n- [ ] Water the plants\n"}
    monkeypatch.setattr("voxa.agent.tools.lists._read", lambda: store["content"])

    def boom() -> str:
        raise OSError("gone")

    monkeypatch.setattr("voxa.agent.tools.dates._read", boom)
    result = _list_next_handler({})
    assert result.ok and result.speech == "Next on your list: Water the plants."


def test_registry_has_new_tools() -> None:
    names = default_registry().names()
    for name in ("list_progress", "list_pending", "list_next"):
        assert name in names
