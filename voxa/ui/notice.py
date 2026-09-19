"""Short-lived notification bar shown at the top of the window."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

VISIBLE_SECONDS = 6
WRAP_CHARS = 60


class NoticeBar(Gtk.Revealer):
    def __init__(self) -> None:
        super().__init__()
        self.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.START)
        self._label = Gtk.Label()
        self._label.set_wrap(True)
        self._label.set_max_width_chars(WRAP_CHARS)
        self._label.set_justify(Gtk.Justification.CENTER)
        self._label.add_css_class("voxa-notice")
        self.set_child(self._label)
        self._source = 0

    def show_text(self, text: str) -> None:
        self._label.set_text(text)
        self.set_reveal_child(True)
        if self._source:
            GLib.source_remove(self._source)
        self._source = GLib.timeout_add_seconds(VISIBLE_SECONDS, self._hide)

    def _hide(self) -> bool:
        self._source = 0
        self.set_reveal_child(False)
        return False
