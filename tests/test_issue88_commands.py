from __future__ import annotations

from voxa.agent.intents import route
from voxa.agent.issueflow import parse_post_issue
from voxa.agent.spoken_text import dictation_prefix, parse_dictation_control
from voxa.ui.avatars import describe_characters, find_character

ROUTE_CASES = [
    ("clean up that text", ("cleanup_text", {})),
    ("fix that document", ("cleanup_text", {})),
    ("make it full screen", ("maximize_app", {"name": ""})),
    ("fullscreen", ("maximize_app", {"name": ""})),
    ("minimize yourself", ("minimize_self", {})),
    ("hide yourself", ("minimize_self", {})),
    ("change character Charlotte", ("change_character", {"name": "charlotte"})),
    ("list characters", ("list_characters", {})),
    ("who can you be", ("list_characters", {})),
]


def test_route_commands() -> None:
    for text, (tool, args) in ROUTE_CASES:
        call = route(text)
        assert call is not None, f"{text!r} should route"
        assert call.tool == tool
        assert call.args == args


def test_hide_window_not_self() -> None:
    call = route("hide the window")
    assert call is not None
    assert call.tool == "minimize_app"
    assert call.args == {"name": ""}


CONTROL_CASES = [
    ("stop dictating", "stop"),
    ("stop dictating stop dictating", "stop"),
    ("hello stop dictating", "stop"),
    ("scratch that", "undo"),
    ("send it", "send"),
    ("keep typing", None),
]


def test_dictation_control() -> None:
    for text, action in CONTROL_CASES:
        assert parse_dictation_control(text) == action


def test_dictation_prefix() -> None:
    assert dictation_prefix("hello stop dictating") == "hello "
    assert dictation_prefix("stop dictating") == ""
    assert dictation_prefix("stop dictating stop dictating") == ""


POST_ISSUE_CASES = [
    ("I need to post a GitHub issue for", ""),
    ("Let's post a get-hub issue about", ""),
    ("GitHub issue", ""),
    ("I need to post a GitHub issue about Spaced Linux", "spaced linux"),
    ("Let's post a bug report about Spaced Linux on GitHub", "spaced linux"),
    ("Open the GitHub issues for and Brave", "and brave"),
]


def test_parse_post_issue() -> None:
    for text, expected in POST_ISSUE_CASES:
        assert parse_post_issue(text) == expected


def test_parse_post_issue_rejects() -> None:
    for text in ("what is a github issue", "open github", "post this to github"):
        assert parse_post_issue(text) is None


def test_find_character() -> None:
    assert find_character("Charlotte") is not None
    assert find_character("charlotte") == find_character("Charlotte")
    assert find_character("no such person") is None


def test_describe_characters() -> None:
    lines = describe_characters()
    assert isinstance(lines, list)
    assert all(isinstance(line, str) for line in lines)
