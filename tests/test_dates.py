"""Birthdays, anniversaries and deadlines route and work end to end."""

from __future__ import annotations

from datetime import date

from voxa.agent.dates import add, find, next_occurrence, parse_day, remove, spoken_upcoming, spoken_when
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.dates import (
    _date_forget_handler,
    _date_remember_handler,
    _date_upcoming_handler,
    _date_when_handler,
)

TODAY = date(2026, 10, 10)


def test_parse_day_forms() -> None:
    assert parse_day("March 3rd", TODAY) == (None, 3, 3)
    assert parse_day("3rd of March", TODAY) == (None, 3, 3)
    assert parse_day("March 3", TODAY) == (None, 3, 3)
    assert parse_day("3 March", TODAY) == (None, 3, 3)
    assert parse_day("october 16 2026", TODAY) == (2026, 10, 16)
    assert parse_day("tomorrow", TODAY) == (2026, 10, 11)
    assert parse_day("today", TODAY) == (2026, 10, 10)
    assert parse_day("friday", TODAY) == (2026, 10, 16)
    assert parse_day("next friday", TODAY) == (2026, 10, 16)
    assert parse_day("this friday", TODAY) == (2026, 10, 16)
    assert parse_day("in 5 days", TODAY) == (2026, 10, 15)
    assert parse_day("in two weeks", TODAY) == (2026, 10, 24)
    assert parse_day("banana", TODAY) is None


def test_add_replace_remove() -> None:
    content = add("", "birthday", "mom", (None, 3, 3))
    assert content == "- 03-03 | birthday | Mom\n"
    again = add(content, "birthday", "Mom", (None, 4, 4))
    assert again == "- 04-04 | birthday | Mom\n"
    both = add(again, "deadline", "the report", (2026, 10, 16))
    assert both == "- 04-04 | birthday | Mom\n- 2026-10-16 | deadline | The report\n"
    new_content, removed = remove(both, "mom")
    assert removed == "Mom"
    assert new_content == "- 2026-10-16 | deadline | The report\n"
    missing, none = remove(new_content, "nobody")
    assert none is None
    assert missing == new_content


def test_find() -> None:
    content = "- 03-03 | birthday | Mom\n- 2026-10-16 | deadline | The report\n"
    assert find(content, "mom") == ("03-03", "birthday", "Mom")
    assert find(content, "mom's") == ("03-03", "birthday", "Mom")
    assert find(content, "report") == ("2026-10-16", "deadline", "The report")
    assert find(content, "nobody") is None


def test_next_occurrence() -> None:
    assert next_occurrence("03-03", TODAY) == date(2027, 3, 3)
    assert next_occurrence("10-16", TODAY) == date(2026, 10, 16)
    assert next_occurrence("2026-10-16", TODAY) == date(2026, 10, 16)
    assert next_occurrence("13-01", TODAY) is None


def test_spoken_when_exact_strings() -> None:
    assert spoken_when(("10-16", "birthday", "Mom"), TODAY) == "Mom's birthday is on Friday, in 6 days."
    assert spoken_when(("2026-10-16", "deadline", "The report"), TODAY) == "The report is due on Friday, in 6 days."
    assert spoken_when(("10-10", "birthday", "Mom"), TODAY) == "Mom's birthday is today."
    assert spoken_when(("10-11", "birthday", "Mom"), TODAY) == "Mom's birthday is tomorrow."
    assert spoken_when(("2026-10-02", "deadline", "The report"), TODAY) == "The report was due on October 2nd."
    assert spoken_when(("03-03", "birthday", "Mom"), TODAY) == "Mom's birthday is on March 3rd 2027, in 144 days."


def test_spoken_upcoming_exact_strings() -> None:
    content = "- 10-16 | birthday | Mom\n- 2026-10-16 | deadline | The report\n"
    assert (
        spoken_upcoming(content, TODAY)
        == "Coming up: Mom's birthday on Friday, and The report due on Friday."
    )
    assert spoken_upcoming("", TODAY) == "Nothing is coming up in the next 30 days."
    assert spoken_upcoming("- 03-03 | birthday | Mom\n", TODAY) == "Nothing is coming up in the next 30 days."


def test_date_phrases_route() -> None:
    assert route("my mom's birthday is March 3rd") == ToolCall(
        "date_remember", {"kind": "birthday", "name": "mom", "when": "March 3rd"}
    )
    assert route("remember that Sam's anniversary is June 12") == ToolCall(
        "date_remember", {"kind": "anniversary", "name": "Sam", "when": "June 12"}
    )
    assert route("the report is due Friday") == ToolCall("date_remember", {"kind": "deadline", "name": "report", "when": "Friday"})
    assert route("when is mom's birthday") == ToolCall("date_when", {"name": "mom"})
    assert route("when is the report due") == ToolCall("date_when", {"name": "report"})
    assert route("any birthdays coming up") == ToolCall("date_upcoming", {})
    assert route("what's due this week") == ToolCall("date_upcoming", {})
    assert route("forget mom's birthday") == ToolCall("date_forget", {"name": "mom"})


def test_misheard_due_still_routes_to_the_date_tools() -> None:
    """Voxa's own voice says "due"; the ear sometimes hears "do" or "2"."""
    assert route("when is the report 2") == ToolCall("date_when", {"name": "report"})
    assert route("Hey when is the report do please") == ToolCall("date_when", {"name": "report"})
    assert route("what's do soon") == ToolCall("date_upcoming", {})


def test_a_verb_in_front_of_the_name_is_not_part_of_the_name() -> None:
    assert route("please remember the report deadline is friday") == ToolCall(
        "date_remember", {"kind": "deadline", "name": "report", "when": "friday"}
    )
    assert route("Remember the report is due Friday") == ToolCall(
        "date_remember", {"kind": "deadline", "name": "report", "when": "Friday"}
    )
    assert route("Please note the report deadline is Friday") == ToolCall(
        "date_remember", {"kind": "deadline", "name": "report", "when": "Friday"}
    )


def test_forget_removes_every_entry_about_the_thing() -> None:
    content = "- 2026-10-16 | deadline | Report\n- 2026-10-16 | deadline | Remember the report\n"
    new_content, removed = remove(content, "report")
    assert removed == "Report"
    assert new_content == ""


def test_deadline_chain_end_to_end(monkeypatch) -> None:
    """Remember, ask, list, forget: the file must really lose the deadline."""
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: store["content"])
    monkeypatch.setattr("voxa.agent.tools.dates._write", lambda content: store.__setitem__("content", content))
    for prompt in ("the report is due friday", "please remember the report deadline is friday"):
        result = _date_remember_handler(route(prompt).args)
        assert result.ok and "report" in result.speech.lower()
    assert "deadline" in store["content"]
    assert _date_when_handler(route("when is the report due").args).ok
    soon = _date_upcoming_handler({})
    assert soon.ok and "report" in soon.speech.lower()
    forgot = _date_forget_handler(route("forget the report deadline").args)
    assert forgot.ok and "forgot" in forgot.speech.lower()
    assert "deadline" not in store["content"]


def test_date_rules_do_not_steal_lists_or_questions() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what is the capital of France") is None


def test_date_handlers_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.dates._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.dates._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _date_remember_handler({"kind": "birthday", "name": "mom", "when": "March 3rd"})
    assert added.ok and added.speech.startswith("Got it.")
    assert store["content"] == "- 03-03 | birthday | Mom\n"
    replaced = _date_remember_handler({"kind": "birthday", "name": "Mom", "when": "April 4th"})
    assert replaced.ok
    assert store["content"] == "- 04-04 | birthday | Mom\n"
    looked = _date_when_handler({"name": "mom"})
    assert looked.ok
    missing = _date_when_handler({"name": "nobody"})
    assert not missing.ok and missing.speech == "I don't have a date for nobody."
    soon = _date_upcoming_handler({})
    assert soon.ok
    forgot = _date_forget_handler({"name": "mom"})
    assert forgot.ok and forgot.speech == "I forgot Mom."
    assert store["content"] == ""
    again = _date_forget_handler({"name": "mom"})
    assert not again.ok and again.speech == "I don't have a date for mom."


def test_date_remember_rejects_unparsable_date() -> None:
    result = _date_remember_handler({"kind": "birthday", "name": "mom", "when": "banana"})
    assert not result.ok and result.speech == "I didn't catch that date."


def test_registry_has_date_tools() -> None:
    names = default_registry().names()
    for name in ("date_remember", "date_when", "date_upcoming", "date_forget"):
        assert name in names
