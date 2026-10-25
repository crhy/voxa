"""Capability 21 (to-do lists) with the words the microphone really heard."""

from __future__ import annotations

from pathlib import Path

from voxa.agent.intents import ToolCall, route
from voxa.agent.lists import FILE_NAME, clear, parse

SAMPLE = Path(__file__).parent.parent / "voxatest" / "data" / "samples" / "lists-sample.md"


def test_misheard_buy_still_adds_the_right_item() -> None:
    assert route("Add buy stamps to my to-do list") == ToolCall(
        "list_add", {"item": "buy stamps", "list": "to-do"}
    )
    for heard in ("add by stamps to my to-do list", "hey add by stamps to my to-do list",
                  "can you add by stamps to my to-do list please"):
        assert route(heard) == ToolCall("list_add", {"item": "buy stamps", "list": "to-do"}), heard


def test_marked_done_survives_a_glued_wake_word() -> None:
    assert route("Mark buy stamps as done") == ToolCall("list_done", {"item": "buy stamps", "list": ""})
    for heard in ("marked by stamps is done", "Heymark by Stamps is done",
                  "can you check off by stamps from my to-do list"):
        call = route(heard)
        assert call is not None and call.tool == "list_done", heard
        assert call.args["item"].casefold() == "buy stamps", heard


def test_clear_paraphrases_route_to_list_clear() -> None:
    assert route("Clear my to-do list") == ToolCall("list_clear", {"list": "to-do"})
    assert route("hey clear my to-do list") == ToolCall("list_clear", {"list": "to-do"})
    assert route("Please empty my to-do list") == ToolCall("list_clear", {"list": "to-do"})


def test_the_lists_sample_holds_a_list_the_clear_can_empty() -> None:
    content = SAMPLE.read_text(encoding="utf-8")
    assert FILE_NAME == "Voxa Lists.md"
    assert parse(content)["To-do"] == [(False, "Buy stamps"), (False, "Pay the rent")]
    new_content, removed = clear(content, "To-do")
    assert removed == 2
    assert "pay the rent" not in new_content.casefold()


def test_ordinary_speech_is_not_cut_into_a_command() -> None:
    assert route("what's on my to-do list") is not None
    assert route("his story was long") is None
    assert route("what is the capital of France") is None
