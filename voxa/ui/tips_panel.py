"""Dim contextual tips shown in the top-left corner of the window."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

MAX_TIPS = 3
#: GTK CSS cannot cap a widget's width, so labels wrap at this many characters
#: to keep the panel within the ~260 px the design calls for.
LABEL_WIDTH_CHARS = 30


class TipsPanel(Gtk.Box):
    """Small vertical list of dim tip labels; hidden when there are none."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("voxa-tips")
        self._tips: list[str] = []
        self.set_visible(False)

    def set_tips(self, tips: list[str]) -> None:
        """Replace the tip labels; an identical list changes nothing (no flicker)."""
        tips = list(tips)[:MAX_TIPS]
        if tips == self._tips:
            return
        self._tips = tips

        child = self.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self.remove(child)
            child = nxt

        for text in tips:
            label = Gtk.Label(label=text)
            label.set_xalign(0)
            label.set_wrap(True)
            label.set_max_width_chars(LABEL_WIDTH_CHARS)
            self.append(label)

        self.set_visible(bool(tips))
