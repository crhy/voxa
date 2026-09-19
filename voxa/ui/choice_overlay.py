"""Contextual choice card shown when a task is blocked on a user decision."""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

from .state import VoxaTask  # noqa: E402


class ChoiceOverlay(Gtk.Box):
    """Small card with one button per choice; hidden by default so it takes no space."""

    def __init__(self, on_choice: Callable[[str, str], None] | None = None) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("voxa-choice-card")
        self.on_choice = on_choice

        self._title = Gtk.Label(label="")
        self._title.set_xalign(0)
        self._title.add_css_class("voxa-task-title")
        self._choices = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.append(self._title)
        self.append(self._choices)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        self.set_visible(False)

    def show_choices(self, task: VoxaTask) -> None:
        """Show the task title and one button per offered choice."""
        self._title.set_text(task.title)
        child = self._choices.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._choices.remove(child)
            child = nxt

        for index, choice in enumerate(task.choices):
            button = Gtk.Button(label=choice)
            button.add_css_class("voxa-choice")
            if index == 0:
                button.add_css_class("suggested-action")
            button.connect("clicked", lambda _, choice=choice: self._answer(task.id, choice))
            self._choices.append(button)

        if "Cancel" not in task.choices:
            cancel = Gtk.Button(label="Cancel")
            cancel.add_css_class("flat")
            cancel.connect("clicked", lambda *_: self.hide_choices())
            self._choices.append(cancel)

        self.set_visible(True)

    def hide_choices(self) -> None:
        self.set_visible(False)

    def _answer(self, task_id: str, choice: str) -> None:
        self.hide_choices()
        if self.on_choice is not None:
            self.on_choice(task_id, choice)

    def _on_key(self, _controller, keyval, _keycode, _mods) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.hide_choices()
            return True
        return False
