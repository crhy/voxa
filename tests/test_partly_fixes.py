from __future__ import annotations

import pytest

from voxa.agent.intents import route


def test_focus_hyphenated_duration_routes_to_focus_start():
    call = route("Start a 25-minute focus session")
    assert call is not None and call.tool == "focus_start"


def test_focus_plain_duration_still_routes():
    call = route("Start a 25 minute focus session")
    assert call is not None and call.tool == "focus_start"


@pytest.mark.parametrize(
    "heard",
    [
        "Start a pomodoro",
        "Start a Paladoro",
        "Start a Palidoro",
        "Start a Pomadoro",
        "Start a Pamodoro",
        "begin a palidoro timer",
    ],
)
def test_pomodoro_sound_alikes_route_to_focus_start(heard):
    call = route(heard)
    assert call is not None and call.tool == "focus_start", heard


def test_fused_contact_email_is_rebuilt():
    call = route("Sam's email is samanexample.com")
    assert call is not None and call.tool == "set_contact_email"
    assert call.args.get("address") == "sam@example.com"


def test_spoken_contact_email_still_works():
    call = route("Alice's email is alice at gmail dot com")
    assert call is not None and call.tool == "set_contact_email"
    assert call.args.get("name").lower() == "alice"


@pytest.mark.parametrize(
    "text,tool",
    [
        ("what's on my to-do list", "list_read"),
        ("what time is it", "tell_time"),
        ("what is the capital of France", None),
        ("open the calculator", "open_app"),
    ],
)
def test_ordinary_phrases_still_route_as_before(text, tool):
    call = route(text)
    expected = tool
    if expected is None:
        assert call is None, text
    else:
        assert call is not None and call.tool == expected, text
