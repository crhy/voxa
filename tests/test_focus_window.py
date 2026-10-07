from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from gi.repository import GLib  # noqa: E402

from voxa.ui import x11hints  # noqa: E402
from voxa.ui.focus_window import FACE_SIZE, FocusWindow  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402


def _pump(iterations: int = 60) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def test_corner_position_top_right():
    assert x11hints.corner_position((0, 0, 1920, 1080), (184, 168)) == (1712, 24)


def test_corner_position_second_monitor():
    assert x11hints.corner_position((1920, 0, 2560, 1440), (184, 168)) == (4272, 24)


def test_corner_position_wider_than_monitor():
    x, _y = x11hints.corner_position((100, 50, 200, 300), (500, 10))
    assert x == 100


def test_window_defaults():
    window = FocusWindow()
    assert not window.get_visible()
    assert not window.get_decorated()
    assert not window.get_focusable()
    assert window.has_css_class("voxa-focus-window")
    window.destroy()


def test_canvas_sizing():
    window = FocusWindow()
    assert window.canvas.natural_size == FACE_SIZE
    assert window.canvas.clip_radius == FACE_SIZE / 2
    assert window.renderer.get_mode() == "prerendered"
    window.destroy()


def test_caption_ready():
    window = FocusWindow()
    window.set_state(AssistantState.READY)
    assert window.caption.get_text() == "Ready"
    window.destroy()


def test_caption_detail_wins():
    window = FocusWindow()
    window.set_state(AssistantState.WORKING, "Searching Amazon")
    assert window.caption.get_text() == "Searching Amazon"
    window.destroy()


def test_speaking_not_restarted():
    window = FocusWindow()
    window.set_state(AssistantState.SPEAKING)
    since = window.renderer._speaking_since
    assert since is not None
    window.set_state(AssistantState.SPEAKING)
    assert window.renderer._speaking_since == since
    window.set_state(AssistantState.READY)
    assert window.renderer._speaking_since is None
    window.destroy()


def test_set_character_once():
    window = FocusWindow()
    calls = []
    window.renderer.set_character = lambda cid: calls.append(cid)
    window.set_character("grace")
    window.set_character("grace")
    assert len(calls) == 1
    window.destroy()


def test_set_character_swallows_errors():
    window = FocusWindow()

    def boom(cid):
        raise ValueError("bad pack")

    window.renderer.set_character = boom
    window.set_character("grace")
    window.destroy()


def test_show_at_corner():
    window = FocusWindow()
    window.show_at_corner((0, 0, 1920, 1080))
    _pump()
    assert window.get_visible()
    assert window._position == x11hints.corner_position(
        (0, 0, 1920, 1080), (FACE_SIZE + 16, FACE_SIZE)
    )
    window.destroy()


def test_hide_window():
    window = FocusWindow()
    window.show_at_corner((0, 0, 1920, 1080))
    _pump()
    window.hide_window()
    assert not window.get_visible()
    window.destroy()


def test_keep_on_top_hidden_no_raise():
    window = FocusWindow()
    window.keep_on_top()
    window.destroy()


def test_activate_main_callback():
    calls = []
    window = FocusWindow(on_activate=lambda: calls.append(1))
    window.activate_main()
    assert calls == [1]
    window.destroy()


def test_activate_main_none():
    window = FocusWindow()
    window.activate_main()
    window.destroy()
