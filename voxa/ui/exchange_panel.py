"""Right-hand card that keeps the latest question and answer on screen."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

PANEL_WIDTH_CHARS = 34
MAX_ANSWER_HEIGHT = 300


class ExchangePanel(Gtk.Box):
    """Shows what Voxa heard and what it answered; stays until the next question."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("voxa-task-panel")

        self.heard_heading = self._heading("YOU SAID")
        self.heard = self._body("voxa-task-title")
        self.answer_heading = self._heading("VOXA")
        self.answer = self._body("voxa-exchange-answer")

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_propagate_natural_height(True)
        scroller.set_max_content_height(MAX_ANSWER_HEIGHT)
        scroller.set_child(self.answer)

        for widget in (self.heard_heading, self.heard, self.answer_heading, scroller):
            self.append(widget)
        self.set_visible(False)

    def _heading(self, text: str) -> Gtk.Label:
        label = Gtk.Label(label=text)
        label.set_xalign(0)
        label.add_css_class("voxa-section")
        return label

    def _body(self, css_class: str) -> Gtk.Label:
        label = Gtk.Label()
        label.set_xalign(0)
        label.set_wrap(True)
        label.set_selectable(True)
        label.set_max_width_chars(PANEL_WIDTH_CHARS)
        label.add_css_class(css_class)
        return label

    def show_question(self, text: str) -> None:
        """A new question replaces the previous exchange."""
        self.heard.set_text(text)
        self.answer.set_text("")
        self.answer_heading.set_visible(False)
        self.set_visible(bool(text))

    def show_answer(self, text: str) -> None:
        self.answer.set_text(text)
        self.answer_heading.set_visible(bool(text))
        self.set_visible(bool(self.heard.get_text() or text))

    def clear(self) -> None:
        self.heard.set_text("")
        self.answer.set_text("")
        self.set_visible(False)
