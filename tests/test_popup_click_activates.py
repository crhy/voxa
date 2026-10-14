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

from voxa.ui.focus_popup import FocusPopup  # noqa: E402
from voxa.ui.shell import AssistantShell  # noqa: E402
from voxa.ui.state import AssistantModel  # noqa: E402


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


def test_click_fires_callback_once_per_click() -> None:
    popup = FocusPopup()
    fired: list[int] = []
    popup.set_click_callback(lambda: fired.append(1))
    popup._click.emit("released", 0, 0.0, 0.0)
    assert fired == [1]
    popup._click.emit("released", 0, 0.0, 0.0)
    assert fired == [1, 1]


def test_popup_click_presents_and_activates() -> None:
    shell = AssistantShell(AssistantModel())
    fired: list[str] = []
    shell.on_present = lambda: fired.append("present")
    shell.on_active = lambda: fired.append("active")
    _present(shell, 1200, 760)

    shell.focus_popup._click.emit("released", 0, 0.0, 0.0)
    assert fired == ["present", "active"]


def test_insensitive_active_button_blocks_activation() -> None:
    shell = AssistantShell(AssistantModel())
    fired: list[str] = []
    shell.on_present = lambda: fired.append("present")
    shell.on_active = lambda: fired.append("active")
    _present(shell, 1200, 760)

    shell.status_controls.active_button.set_sensitive(False)
    shell.focus_popup._click.emit("released", 0, 0.0, 0.0)
    assert fired == ["present"]
