from __future__ import annotations

import pytest

from voxa.agent.spoken_text import format_dictation, parse_dictation_control

FORMAT_CASES = [
    ("hello comma world period new line thanks", "hello, world.\nthanks "),
    ("new line", "\n"),
    ("new paragraph", "\n\n"),
    ("hello", "hello "),
    ("", ""),
    ("period", ". "),
    ("Question Mark please", "? please "),
    ("one two three", "one two three "),
    ("open quote hello close quote period", '"hello". '),
    ("wait colon one semicolon two", "wait: one; two "),
    ("full stop new paragraph thanks", ".\n\nthanks "),
    ("exclamation point new line exclamation mark", "!\n! "),
    ("keep the CAPS and commas", "keep the CAPS and commas "),
    ("new line new line spaced", "\n\nspaced "),
]

CONTROL_CASES = [
    ("stop dictating", "stop"),
    ("Stop Dictation!", "stop"),
    ("end dictation.", "stop"),
    ("that's all", "stop"),
    ("scratch that", "undo"),
    ("Undo that!", "undo"),
    ("send it", "send"),
    ("Send the email.", "send"),
    ("hello there", None),
    ("keep typing", None),
    ("", None),
    ("Vox A Stopdictation", "stop"),
    ("Voxa, stop dictation.", "stop"),
    ("Voxa.", "stop"),
    ("Boxer", "stop"),
    ("All right, so are we done dictating?", None),
]


@pytest.mark.parametrize(("spoken", "expected"), FORMAT_CASES)
def test_format_dictation(spoken: str, expected: str) -> None:
    assert format_dictation(spoken) == expected


@pytest.mark.parametrize(("spoken", "expected"), CONTROL_CASES)
def test_parse_dictation_control(spoken: str, expected: str | None) -> None:
    assert parse_dictation_control(spoken) == expected
