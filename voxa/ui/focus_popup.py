"""Focus pop-up: the talking head plus a status caption, shown when the window is not focused.

When the window manager reports the Voxa window is not focused (the user is working
elsewhere), a small circular cut-out of the talking head appears in the top-right with
a one-line caption of what Voxa is doing. While the window is focused it stays hidden.
"""

from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

from .assistant_view import BADGE_PATH, STATE_CAPTIONS  # noqa: E402
from .state import AssistantState  # noqa: E402

POPUP_SIZE = 96
CAPTION_WIDTH_CHARS = 18


class FocusPopup(Gtk.Box):
    """Circular talking-head cut-out with a status caption; hidden while the window is focused."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("voxa-focus-popup")
        self.set_visible(False)

        circle = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        circle.add_css_class("voxa-focus-circle")
        circle.set_size_request(POPUP_SIZE, POPUP_SIZE)
        try:
            texture = Gdk.Texture.new_from_filename(str(BADGE_PATH))
            avatar = Gtk.Image.new_from_paintable(texture)
        except Exception:
            avatar = Gtk.Image()
        avatar.set_pixel_size(POPUP_SIZE)
        avatar.set_halign(Gtk.Align.CENTER)
        avatar.set_valign(Gtk.Align.CENTER)
        avatar.add_css_class("voxa-focus-avatar")
        circle.append(avatar)
        self._circle = circle

        self._caption = Gtk.Label(label="")
        self._caption.set_xalign(0.5)
        self._caption.set_wrap(True)
        self._caption.set_max_width_chars(CAPTION_WIDTH_CHARS)
        self._caption.add_css_class("voxa-focus-caption")

        self.append(circle)
        self.append(self._caption)

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        """Set the caption; a non-empty detail replaces the default state text."""
        self._caption.set_text(detail or STATE_CAPTIONS.get(state, ""))

    def set_window_focus(self, focused: bool) -> None:
        """Show the pop-up only while the window is NOT focused (the user is elsewhere)."""
        self.set_visible(not focused)

    @property
    def caption(self) -> Gtk.Label:
        return self._caption
