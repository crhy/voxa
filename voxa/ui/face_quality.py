"""A clearly labelled "Facial Quality" switch: Low / Medium / High.

Low = a still portrait, Medium = pre-rendered lip movement, High = the full
neural face. The three levels are toggle buttons kept mutually exclusive by
hand (this GTK build has no linked-toggle enum); clicking one fires
``on_selected(mode)`` once, and clicking the already-active one is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

QUALITY_LEVELS = (
    ("still", "Low", "A still portrait"),
    ("prerendered", "Medium", "Pre-rendered lip movement, works on any computer"),
    ("live", "High", "Full neural face, needs a graphics card"),
)

MODE_TO_LABEL = {mode: label for mode, label, _desc in QUALITY_LEVELS}
LABEL_TO_MODE = {label: mode for mode, label, _desc in QUALITY_LEVELS}
DESCRIPTIONS = {mode: desc for mode, _label, desc in QUALITY_LEVELS}

DEFAULT_MODE = "prerendered"
LIVE_UNAVAILABLE_REASON = "The full neural face is not available yet"


def quality_label(mode: str) -> str:
    """Map a face mode to its quality label; unknown modes read as Medium."""
    return MODE_TO_LABEL.get(mode, "Medium")


def mode_for_label(label: str) -> str:
    """Map a quality label back to a face mode; unknown labels read as Medium."""
    return LABEL_TO_MODE.get(label, DEFAULT_MODE)


class FaceQualitySwitch(Gtk.Box):
    """A caption plus three linked toggle buttons (Low / Medium / High)."""

    def __init__(self, on_selected: Callable[[str], None] | None = None) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.on_selected = on_selected

        caption = Gtk.Label(label="FACIAL QUALITY")
        caption.set_xalign(0)
        caption.add_css_class("voxa-control-caption")
        self._caption = caption
        self.append(caption)

        self._buttons: dict[str, Gtk.ToggleButton] = {}
        self._mode = DEFAULT_MODE
        self._updating = False
        for mode, label, desc in QUALITY_LEVELS:
            button = Gtk.ToggleButton(label=label)
            button.add_css_class("linked")
            button.set_tooltip_text(desc)
            button.set_active(mode == self._mode)
            button.connect("clicked", self._make_click_handler(mode))
            self._buttons[mode] = button
            self.append(button)

    def _make_click_handler(self, mode: str):
        def _handler(_button: Gtk.ToggleButton, mode: str = mode) -> None:
            if self._updating:
                return
            if mode == self._mode:
                self._buttons[mode].set_active(True)
                return
            for other, button in self._buttons.items():
                if other != mode:
                    button.set_active(False)
            self._mode = mode
            self._fire()

        return _handler

    def set_mode(self, mode: str) -> None:
        """Select a level by value without firing the callback; unknown ignored."""
        if mode not in self._buttons:
            return
        self._updating = True
        try:
            self._mode = mode
            for other, button in self._buttons.items():
                button.set_active(other == mode)
        finally:
            self._updating = False

    def get_mode(self) -> str:
        return self._mode

    def set_live_available(self, available: bool, reason: str = "") -> None:
        """Enable or disable the High (live) button.

        When disabled the High button becomes insensitive and its tooltip is
        ``reason``; if "live" was selected it falls back to "prerendered" and
        fires the callback once.
        """
        high = self._buttons["live"]
        if not available:
            high.set_sensitive(False)
            high.set_tooltip_text(reason or LIVE_UNAVAILABLE_REASON)
            if self._mode == "live":
                self._mode = DEFAULT_MODE
                self._buttons[DEFAULT_MODE].set_active(True)
                self._fire()
        else:
            high.set_sensitive(True)
            high.set_tooltip_text(DESCRIPTIONS["live"])

    def _fire(self) -> None:
        if self.on_selected is not None:
            self.on_selected(self._mode)
