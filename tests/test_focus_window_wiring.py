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

from voxa.ui.shell import AssistantShell  # noqa: E402
from voxa.ui.state import AssistantModel, AssistantState  # noqa: E402


def _pump(iterations: int = 200) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def test_fallback_without_focus_window():
    shell = AssistantShell(AssistantModel())
    assert shell.focus_window is None
    shell.set_window_focus(False)
    _pump()
    assert shell.focus_popup.get_visible()
    shell.close_focus_window()


def test_enable_shows_floating_head():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    shell.set_window_focus(False)
    _pump()
    assert shell.focus_window.get_visible()
    assert not shell.focus_popup.get_visible()
    shell.close_focus_window()


def test_focus_hides_floating_head():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    shell.set_window_focus(False)
    _pump()
    assert shell.focus_window.get_visible()
    shell.set_window_focus(True)
    _pump()
    assert not shell.focus_window.get_visible()
    shell.close_focus_window()


def test_keep_on_top_when_already_visible():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    shell.set_window_focus(False)
    _pump()
    assert shell.focus_window.get_visible()
    fw = shell.focus_window
    keep = []
    corner = []
    fw.keep_on_top = lambda: keep.append(1)
    fw.show_at_corner = lambda *a: corner.append(1)
    shell.set_window_focus(False)
    assert keep
    assert not corner
    shell.close_focus_window()


def test_apply_state_sets_caption():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    shell._apply_state(AssistantState.THINKING, "")
    assert shell.focus_window.caption.get_text() == "Thinking…"
    shell.close_focus_window()


def test_forwarding_reaches_focus_window():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    fw = shell.focus_window
    calls = []
    fw.set_word_timeline = lambda words: calls.append(("wt", words))
    fw.set_speech_clock = lambda clock: calls.append(("clock", clock))
    fw.set_audio_level = lambda level: calls.append(("level", level))
    fw.reset_word_timeline = lambda: calls.append(("reset",))
    shell.set_word_timeline([("hi", 0.0, 0.1)])
    shell.set_speech_clock(0.5)
    shell.set_audio_level(0.3)
    shell.reset_word_timeline()
    assert ("wt", [("hi", 0.0, 0.1)]) in calls
    assert ("clock", 0.5) in calls
    assert ("level", 0.3) in calls
    assert ("reset",) in calls
    shell.close_focus_window()


def test_activate_callback():
    shell = AssistantShell(AssistantModel())
    called = []
    shell.enable_focus_window(on_activate=lambda: called.append(1))
    shell.focus_window.activate_main()
    assert called
    shell.close_focus_window()


def test_close_falls_back_to_badge():
    shell = AssistantShell(AssistantModel())
    shell.enable_focus_window()
    shell.close_focus_window()
    assert shell.focus_window is None
    shell.set_window_focus(False)
    _pump()
    assert shell.focus_popup.get_visible()
