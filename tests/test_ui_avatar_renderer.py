from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.assistant_view import AssistantView, StaticAssistantRenderer  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402


def test_default_renderer_is_static() -> None:
    view = AssistantView()
    assert isinstance(view.renderer, StaticAssistantRenderer)
    assert view.renderer.widget.get_css_classes()


def test_public_api_forwards_to_renderer() -> None:
    view = AssistantView()
    view.set_state(AssistantState.READY)
    assert view.caption.get_text() == "Ready"
    view.set_listening(True)
    view.set_thinking(True)
    view.set_speaking(True)
    view.set_audio_level(0.8)
    assert view.audio_level == 0.8
    assert view.has_css_class("listening")
    assert view.has_css_class("thinking")
    assert view.has_css_class("speaking")
    assert view.renderer._avatar.has_css_class("audio-high")

    view.set_audio_level(-1.2)
    assert view.audio_level == 0.0
    assert view.renderer._avatar.has_css_class("audio-low") is False


def test_future_renderer_stub_defaults_are_noops() -> None:
    view = AssistantView()
    view.set_emotion("curious")
    view.set_viseme("M")
    view.set_gaze_target(0.2, -0.1)
    view.set_activity_intensity(0.7)
    assert view.audio_level == 0.0


def test_3d_opt_in_falls_back_when_renderer_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    view = AssistantView()
    assert isinstance(view.renderer, StaticAssistantRenderer)


def test_set_character_empty_keeps_static_renderer(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    view = AssistantView()
    view.set_character("")
    assert isinstance(view.renderer, StaticAssistantRenderer)
    assert view._character_id == ""


def test_avatar_widget_is_inside_assistant_view() -> None:
    view = AssistantView()
    child = view.get_first_child()
    assert child is view.renderer.widget
    avatar = child.get_first_child()
    assert avatar.has_css_class("voxa-avatar")
    assert view.caption.get_parent() is child
