from __future__ import annotations

import inspect

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from voxa.agent import repeat  # noqa: E402
from voxa.window import MainWindow  # noqa: E402

REPEATABLE_PHRASES = [
    "again",
    "do that again",
    "do it again",
    "do that once more",
    "once more",
    "one more time",
    "same again",
    "and again",
    "repeat that action",
    "do the same again",
    "more",
]

NOT_REPEATABLE = [
    "say that again",
    "repeat that",
    "play it again sam",
    "again and again",
]


def test_repeat_phrases():
    for phrase in REPEATABLE_PHRASES:
        assert repeat.is_repeat_action(phrase)
        assert repeat.is_repeat_action(f"Voxa, {phrase}")
        assert repeat.is_repeat_action(f"{phrase} please")
        assert repeat.is_repeat_action(f"Voxa, {phrase} please")


def test_not_repeat_phrases():
    for phrase in NOT_REPEATABLE:
        assert not repeat.is_repeat_action(phrase)
        assert not repeat.is_repeat_action(f"Voxa, {phrase}")
        assert not repeat.is_repeat_action(f"{phrase} please")
        assert not repeat.is_repeat_action(f"Voxa, {phrase} please")


def test_repeatable_table():
    for tool in sorted(repeat.NEVER_REPEAT):
        assert not repeat.repeatable(tool)
    for tool in ("volume_up", "zoom_in", "open_app", "play_music", "take_screenshot"):
        assert repeat.repeatable(tool)


def test_window_bookkeeping():
    src = inspect.getsource(MainWindow.ask_ai)
    assert "repeat.is_repeat_action(prompt)" in src
    assert src.index("repeat.is_repeat_action(prompt)") > src.index("if self._external_dictation:")
    run_src = inspect.getsource(MainWindow._run_tool)
    assert "self._last_tool_call = call" in run_src


def test_dictation_wins_over_starting_an_issue_or_an_email():
    import inspect

    from voxa.window import MainWindow

    source = inspect.getsource(MainWindow.ask_ai)
    dictating = source.index("if self._external_dictation:")
    assert dictating < source.index("issueflow.parse_post_issue(prompt)")
    assert dictating < source.index("mail.parse_email_command(prompt)")
    assert dictating < source.index("repeat.is_repeat_action(prompt)")
