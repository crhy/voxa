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

from voxa.ui.focus_popup import POPUP_SIZE, FocusPopup  # noqa: E402
from voxa.ui.shell import AssistantShell  # noqa: E402
from voxa.ui.state import AssistantModel, AssistantState  # noqa: E402


def _pump(iterations: int = 200) -> None:
    """Push the main context until widgets realize, with a hard bound."""
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        if not ctx.iteration(False):
            continue
    for _ in range(20):
        ctx.iteration(False)


def _present(shell: AssistantShell, width: int, height: int) -> Gtk.Window:
    window = Gtk.Window()
    window.set_default_size(width, height)
    window.set_child(shell)
    window.present()
    _pump()
    return window


def test_popup_starts_hidden() -> None:
    popup = FocusPopup()
    assert not popup.get_visible()


def test_popup_shows_only_when_window_unfocused() -> None:
    popup = FocusPopup()
    popup.set_window_focus(True)
    assert not popup.get_visible()
    popup.set_window_focus(False)
    assert popup.get_visible()
    popup.set_window_focus(True)
    assert not popup.get_visible()


def test_caption_follows_state() -> None:
    popup = FocusPopup()
    popup.set_state(AssistantState.READY)
    assert popup.caption.get_text() == "Ready"
    popup.set_state(AssistantState.SPEAKING)
    assert popup.caption.get_text() == "Speaking…"
    popup.set_state(AssistantState.WORKING, "Step 3 of 6")
    assert popup.caption.get_text() == "Step 3 of 6"


def test_circle_holds_the_avatar() -> None:
    popup = FocusPopup()
    circle = popup.get_first_child()
    assert circle.has_css_class("voxa-focus-circle")
    assert circle.get_size_request()[0] == POPUP_SIZE
    avatar = circle.get_first_child()
    assert isinstance(avatar, Gtk.Image)
    assert avatar.has_css_class("voxa-focus-avatar")


def test_shell_wires_focus_popup() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    _present(shell, 1200, 760)

    shell.set_window_focus(True)
    assert not shell.focus_popup.get_visible()
    shell.set_window_focus(False)
    assert shell.focus_popup.get_visible()
