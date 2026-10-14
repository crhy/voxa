from __future__ import annotations

from voxa.agent.intents import ToolCall, route


def test_focus_session_hyphenated() -> None:
    assert route("Start a 25-minute focus session") == ToolCall(
        "focus_start", {"duration": "25 minute", "what": ""}
    )


def test_focus_session_spaced() -> None:
    assert route("start a 25 minute focus session") == ToolCall(
        "focus_start", {"duration": "25 minute", "what": ""}
    )


def test_timer_hyphenated() -> None:
    assert route("set a 5-minute timer") == ToolCall("set_timer", {"duration": "5 minute"})


def test_county_later_is_calculator() -> None:
    assert route("Start the county later") == ToolCall("open_app", {"name": "calculator"})


def test_guard_phrases_unchanged() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
