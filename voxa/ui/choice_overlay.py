"""Compact contextual choice prompt (issue #5).

Shown inline where the relevant task lives; disappears on answer.
"""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Gtk


class ChoiceOverlay(Gtk.Box):
    """A small question with one button per choice."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("voxa-choice")
        self.set_visible(False)
        self._prompt = Gtk.Label(xalign=0)
        self._prompt.add_css_class("dim-label")
        self._buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.append(self._prompt)
        self.append(self._buttons)

    def show_for(
        self, title: str, choices: tuple[str, ...], on_pick: Callable[[str], None]
    ) -> None:
        """Replace content with ``choices``; ``on_pick`` fires once, then hides."""
        self._prompt.set_text(title)
        child = self._buttons.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self._buttons.remove(child)
            child = next_child
        for choice in choices:
            button = Gtk.Button(label=choice)
            button.add_css_class("pill")
            button.connect(
                "clicked", lambda _b, c=choice: (on_pick(c), self.hide_overlay())
            )
            self._buttons.append(button)
        self.set_visible(True)

    def hide_overlay(self) -> None:
        self.set_visible(False)
