from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.assistant_view import AssistantView  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402


def test_caption_per_state() -> None:
    view = AssistantView()
    expected = {
        AssistantState.OFFLINE: "Offline",
        AssistantState.READY: "Ready",
        AssistantState.LISTENING: "Listening",
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
    assert view.hint.get_text() == ""
    assert view.hint.get_visible() is False
    view.set_state(AssistantState.WORKING)
    assert view.caption.get_text() == "Working…"


def test_hint_per_state() -> None:
    view = AssistantView()
    expected = {
        AssistantState.OFFLINE: "Press ACTIVE to start listening",
        AssistantState.READY: "Say “Voxa”, then your request",
        AssistantState.LISTENING: "Go ahead",
        AssistantState.THINKING: "",
        AssistantState.SPEAKING: "Say “Voxa” to interrupt",
        AssistantState.WORKING: "",
        AssistantState.WAITING: "Answer to continue",
        AssistantState.ERROR: "Try again",
    }
    for state, text in expected.items():
        view.set_state(state)
        assert view.hint.get_text() == text
        assert view.hint.get_visible() is bool(text)


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


def test_artwork_has_a_fixed_size_and_leaves_negative_space() -> None:
    view = AssistantView()
    window = Gtk.Window()
    window.set_default_size(1200, 760)
    window.set_child(view)
    window.present()
    ctx = GLib.MainContext.default()
    for _ in range(100):
        ctx.iteration(False)
    _minimum, natural, _mb, _nb = view.measure(Gtk.Orientation.VERTICAL, -1)
    assert natural <= 400
    window.destroy()


def test_character_without_portrait_keeps_static_renderer() -> None:
    view = AssistantView("no-such-character")
    assert view.renderer is view._static_renderer


def test_empty_character_uses_static_badge() -> None:
    view = AssistantView("")
    assert view.renderer is view._static_renderer


def test_portrait_character_uses_photo_renderer() -> None:
    view = AssistantView("aoife")
    assert type(view.renderer).__name__ == "PhotoFaceRenderer"
    assert type(view.renderer.widget).__name__ == "FaceCanvas"


def test_face_mode_reaches_renderer() -> None:
    view = AssistantView("aoife")
    view.set_face_mode("still")
    assert view.renderer.get_mode() == "still"


def test_face_mode_survives_set_character() -> None:
    view = AssistantView("aoife")
    view.set_face_mode("still")
    view.set_character("chidi")
    assert view.renderer.get_mode() == "still"


def test_speaking_reaches_photo_renderer(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "voxa.ui.photo_face.PhotoFaceRenderer.set_speaking",
        lambda self, active: calls.append(active),
    )
    view = AssistantView("aoife")
    view.set_speaking(True)
    assert calls == [True]


def test_word_timeline_and_clock_forward_to_active_renderer(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "voxa.ui.photo_face.PhotoFaceRenderer.set_word_timeline",
        lambda self, words: calls.append(("words", words)),
    )
    monkeypatch.setattr(
        "voxa.ui.photo_face.PhotoFaceRenderer.set_speech_clock",
        lambda self, clock: calls.append(("clock", clock)),
    )
    view = AssistantView("aoife")
    clock = object()
    view.set_word_timeline([("hi", 0.0, 0.2)])
    view.set_speech_clock(clock)
    assert calls == [("words", [("hi", 0.0, 0.2)]), ("clock", clock)]


def test_no_3d_renderer_is_constructed(monkeypatch) -> None:
    def _boom(*_args, **_kwargs) -> None:
        raise AssertionError("the 3D renderer must not be built")

    monkeypatch.setattr("voxa.ui.avatar_3d.Gl3DFaceRenderer.__init__", _boom)
    view = AssistantView("aoife")
    assert view.renderer is not None


def test_apply_face_mode_pushes_setting() -> None:
    from voxa.window import MainWindow

    class _Settings:
        face_mode = "still"

    class _Shell:
        assistant_view = None

    view = AssistantView("aoife")
    shell = _Shell()
    shell.assistant_view = view
    holder = type("Holder", (), {})()
    holder.shell = shell
    holder.settings = _Settings()
    holder._refresh_live_available = lambda: None  # the face-server check is not part of this test
    MainWindow._apply_face_mode(holder)
    assert view.get_face_mode() == "still"
