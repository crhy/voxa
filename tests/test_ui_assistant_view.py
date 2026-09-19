from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.assistant_view import AssistantView  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402


def test_caption_per_state() -> None:
    view = AssistantView()
    expected = {
        AssistantState.OFFLINE: "Offline",
        AssistantState.READY: "Ready",
        AssistantState.LISTENING: "Listening…",
        AssistantState.THINKING: "Thinking…",
        AssistantState.SPEAKING: "Speaking…",
        AssistantState.WORKING: "Working…",
        AssistantState.WAITING: "Waiting for you…",
        AssistantState.ERROR: "Something went wrong",
    }
    for state, text in expected.items():
        view.set_state(state)
        assert view.caption.get_text() == text


def test_detail_replaces_caption() -> None:
    view = AssistantView()
    view.set_state(AssistantState.WORKING, detail="Step 3 of 6")
    assert view.caption.get_text() == "Step 3 of 6"
    view.set_state(AssistantState.WORKING)
    assert view.caption.get_text() == "Working…"


def test_activity_classes_toggle() -> None:
    view = AssistantView()
    view.set_listening(True)
    assert view.has_css_class("listening")
    view.set_listening(False)
    assert not view.has_css_class("listening")
    view.set_thinking(True)
    view.set_speaking(True)
    assert view.has_css_class("thinking")
    assert view.has_css_class("speaking")


def test_audio_level_clamped() -> None:
    view = AssistantView()
    view.set_audio_level(0.5)
    assert view.audio_level == 0.5
    view.set_audio_level(1.7)
    assert view.audio_level == 1.0
    view.set_audio_level(-0.4)
    assert view.audio_level == 0.0
