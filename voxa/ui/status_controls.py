"""ACTIVE / OFFLINE status controls (bottom-right of the window)."""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from .state import AssistantState  # noqa: E402


class StatusControls(Gtk.Box):
    """Two large, equal buttons whose selected state mirrors the assistant state."""

    def __init__(
        self,
        on_active: Callable[[], None] | None = None,
        on_offline: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.on_active = on_active
        self.on_offline = on_offline

        self.active_button = Gtk.Button(label="ACTIVE")
        self.active_button.add_css_class("voxa-active")
        self.active_button.update_property([Gtk.AccessibleProperty.LABEL], ["Activate assistant"])
        self.active_button.connect("clicked", lambda *_: self._fire(self.on_active))

        self.offline_button = Gtk.Button(label="OFFLINE")
        self.offline_button.add_css_class("voxa-offline")
        self.offline_button.update_property([Gtk.AccessibleProperty.LABEL], ["Take assistant offline"])
        self.offline_button.connect("clicked", lambda *_: self._fire(self.on_offline))

        self.append(self.active_button)
        self.append(self.offline_button)
        self.set_state(AssistantState.OFFLINE)

    def set_state(self, state: AssistantState) -> None:
        """OFFLINE selects the offline button; every other state selects ACTIVE."""
        if state is AssistantState.OFFLINE:
            self._select(self.offline_button, self.active_button)
        else:
            self._select(self.active_button, self.offline_button)

    @staticmethod
    def _select(selected: Gtk.Button, other: Gtk.Button) -> None:
        selected.add_css_class("selected")
        selected.remove_css_class("dimmed")
        other.add_css_class("dimmed")
        other.remove_css_class("selected")

    @staticmethod
    def _fire(callback: Callable[[], None] | None) -> None:
        if callback is not None:
            callback()
