"""A clearly labelled "Facial Quality" switch: Low / Med / High.

Low = a still portrait, Medium = pre-rendered lip movement, High = the full
neural face. The three levels are toggle buttons kept mutually exclusive by
hand (this GTK build has no linked-toggle enum); clicking one fires
``on_selected(mode)`` once, and clicking the already-active one is a no-op.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
import cairo  # noqa: E402 - after gi, so a machine without GTK reports gi as the missing module
from gi.repository import Gtk  # noqa: E402

QUALITY_LEVELS = (
    ("still", "Low", "A still portrait"),
    ("prerendered", "Med", "Medium: pre-rendered lip movement, works on any computer"),
    ("live", "High", "Full neural face, needs a graphics card"),
)

MODE_TO_LABEL = {mode: label for mode, label, _desc in QUALITY_LEVELS}
LABEL_TO_MODE = {label: mode for mode, label, _desc in QUALITY_LEVELS}
DESCRIPTIONS = {mode: desc for mode, _label, desc in QUALITY_LEVELS}

DEFAULT_MODE = "prerendered"
LIVE_UNAVAILABLE_REASON = "The full neural face is not available yet"


class CircularProgress(Gtk.Overlay):
    """Small circular progress gauge with its percentage centred in the ring."""

    def __init__(self) -> None:
        super().__init__()
        self.percentage = 0

        self._ring = Gtk.DrawingArea()
        self._ring.set_content_width(44)
        self._ring.set_content_height(44)
        self._ring.set_draw_func(self._draw)
        self.set_child(self._ring)

        self._label = Gtk.Label(label="0%")
        self._label.add_css_class("caption")
        self._label.set_halign(Gtk.Align.CENTER)
        self._label.set_valign(Gtk.Align.CENTER)
        self.add_overlay(self._label)

    def set_percentage(self, percentage: int | float) -> None:
        self.percentage = max(0, min(100, round(percentage)))
        self._label.set_text(f"{self.percentage}%")
        self._ring.queue_draw()

    def _draw(self, area: Gtk.DrawingArea, cr: cairo.Context, width: int, height: int) -> None:
        radius = max(1.0, min(width, height) / 2.0 - 4.0)
        centre_x = width / 2.0
        centre_y = height / 2.0
        colour = area.get_style_context().get_color()
        cr.set_line_width(4.0)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)

        cr.set_source_rgba(colour.red, colour.green, colour.blue, 0.18)
        cr.arc(centre_x, centre_y, radius, 0, math.tau)
        cr.stroke()

        if self.percentage:
            start = -math.pi / 2.0
            cr.set_source_rgba(colour.red, colour.green, colour.blue, colour.alpha)
            cr.arc(centre_x, centre_y, radius, start, start + math.tau * self.percentage / 100.0)
            cr.stroke()


def quality_label(mode: str) -> str:
    """Map a face mode to its quality label; unknown modes read as Med."""
    return MODE_TO_LABEL.get(mode, "Med")


def mode_for_label(label: str) -> str:
    """Map a quality label back to a face mode; unknown labels read as Medium."""
    return LABEL_TO_MODE.get(label, DEFAULT_MODE)


class FaceQualitySwitch(Gtk.Box):
    """A caption plus three linked toggle buttons (Low / Med / High)."""

    def __init__(self, on_selected: Callable[[str], None] | None = None, stacked: bool = False) -> None:
        # ``stacked`` puts the caption on its own line above the buttons: a narrow block for the left column.
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL if stacked else Gtk.Orientation.HORIZONTAL,
            spacing=6 if stacked else 8,
        )
        self.on_selected = on_selected
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8) if stacked else self

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
            row.append(button)

        # What High is doing right now; empty and hidden otherwise.
        self._loading = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._loading.set_visible(False)
        self._progress = CircularProgress()
        self._loading.append(self._progress)
        self._status = Gtk.Label(label="")
        self._status.set_xalign(0)
        self._status.add_css_class("voxa-face-status")
        self._loading.append(self._status)
        row.append(self._loading)
        if stacked:
            self.append(row)

    def set_status(self, text: str) -> None:
        """Show a short status after the buttons; an empty string hides it."""
        self._status.set_text(text)
        self._progress.set_visible(False)
        self._loading.set_visible(bool(text))

    def set_loading_progress(self, percentage: int | float) -> None:
        """Show High's circular startup gauge and update its centred percentage."""
        self._status.set_text("Loading high-resolution face…")
        self._progress.set_percentage(percentage)
        self._progress.set_visible(True)
        self._loading.set_visible(True)

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
