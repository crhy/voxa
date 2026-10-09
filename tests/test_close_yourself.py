from __future__ import annotations

from voxa.agent.intents import ToolCall, route

CLOSE_CASES = [
    "close yourself",
    "close voxa",
    "quit voxa",
    "exit voxa",
    "shut yourself down",
    "goodbye voxa",
    "please close yourself",
    "close voxa please",
]

NEGATIVE_CASES = [
    "close the window",
    "close firefox",
]


def test_close_yourself_phrases_route_to_the_tool() -> None:
    for phrase in CLOSE_CASES:
        assert route(phrase) == ToolCall("close_yourself", {}), phrase


def test_close_yourself_does_not_capture_other_close_phrases() -> None:
    for phrase in NEGATIVE_CASES:
        assert route(phrase) != ToolCall("close_yourself", {}), phrase
