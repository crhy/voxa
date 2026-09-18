"""ACTIVE / OFFLINE controls (issue #5).

Two large, equally sized buttons bound to real application state:
ACTIVE enables the persistent wake-word mode, OFFLINE stops listening,
capture, speech, and task execution. Not cosmetic.
"""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Gtk


class ActiveOfflineControls(Gtk.Box):
    """Green ACTIVE + red OFFLINE buttons."""

    def __init__(self) -> None:
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8, homogeneous=True
        )
        self.active_button = Gtk.Button(label="ACTIVE")
        self.active_button.set_size_request(140, 48)
        self.active_button.add_css_class("voxa-mode-button")
        self.active_button.set_tooltip_text("Enable the assistant (wake-word listening)")
        self.offline_button = Gtk.Button(label="OFFLINE")
        self.offline_button.set_size_request(140, 48)
        self.offline_button.add_css_class("voxa-mode-button")
        self.offline_button.set_tooltip_text("Stop listening, speech, and tasks")
        self.append(self.active_button)
        self.append(self.offline_button)
        self.set_agent_state(running=False, offline=True)

    def set_agent_state(self, *, running: bool, offline: bool) -> None:
        """Reflect backend state: green ACTIVE while listening, red when offline."""
        if running:
            self.active_button.add_css_class("voxa-active-on")
        else:
            self.active_button.remove_css_class("voxa-active-on")
        if offline:
            self.offline_button.add_css_class("voxa-offline-on")
        else:
            self.offline_button.remove_css_class("voxa-offline-on")

    def connect_active(self, callback: Callable[[], None]) -> None:
        self.active_button.connect("clicked", lambda _b: callback())

    def connect_offline(self, callback: Callable[[], None]) -> None:
        self.offline_button.connect("clicked", lambda _b: callback())
