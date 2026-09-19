from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.legacy_view import LegacyCallbacks, LegacyView  # noqa: E402


def _view(events: list[str]) -> LegacyView:
    def name_for(name: str) -> object:
        return lambda *args: events.append(name)

    return LegacyView(
        LegacyCallbacks(
            record_toggled=name_for("record"),
            conversation_toggled=name_for("conversation"),
            copy=name_for("copy"),
            copy_reply=name_for("copy_reply"),
            clear=name_for("clear"),
            ask=name_for("ask"),
            speak=name_for("speak"),
            stop=name_for("stop"),
            model_selected=name_for("model"),
        )
    )


def test_every_widget_is_exposed() -> None:
    view = LegacyView(
        LegacyCallbacks(
            record_toggled=lambda *_: None,
            conversation_toggled=lambda *_: None,
            copy=lambda: None,
            copy_reply=lambda: None,
            clear=lambda: None,
            ask=lambda: None,
            speak=lambda: None,
            stop=lambda: None,
            model_selected=lambda *_: None,
        )
    )
    for name in LegacyView.WIDGET_NAMES:
        assert getattr(view, name) is not None
    assert view.view is not None


def test_header_and_action_bar_widgets_match_the_legacy_names() -> None:
    view = _view([])
    assert view.record_button.get_label() == "Dictate"
    assert view.conversation_button.get_label() == "Conversation"
    assert view.ask_button.get_label() == "Ask AI"
    assert view.stop_button.get_label() == "Stop"
    assert view.status_label.get_text() == "Starting…"
    assert view.status_box.has_css_class("status-strip")
    assert view.model_combo.has_css_class("model-select")
    assert not view.progress.get_visible()
    assert not view.gpu_box.get_visible()


def test_each_button_fires_its_callback_exactly_once() -> None:
    events: list[str] = []
    view = _view(events)
    view.record_button.set_active(True)
    view.conversation_button.set_active(True)
    view.copy_button.emit("clicked")
    view.copy_response_button.emit("clicked")
    view.clear_button.emit("clicked")
    view.ask_button.emit("clicked")
    view.speak_button.emit("clicked")
    view.stop_button.emit("clicked")
    assert events == [
        "record",
        "conversation",
        "copy",
        "copy_reply",
        "clear",
        "ask",
        "speak",
        "stop",
    ]


def test_transcript_is_editable_and_response_is_not() -> None:
    view = _view([])
    assert view.transcript_view.get_editable()
    assert view.transcript_view.get_cursor_visible()
    assert not view.response_view.get_editable()
    assert not view.response_view.get_cursor_visible()
    assert view.transcript_view.has_css_class("document")
    assert view.response_view.has_css_class("card")
