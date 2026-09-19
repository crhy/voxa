"""Compact floating task rail: ACTIVE TASKS and TO DO sections."""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from .state import TaskState, VoxaTask  # noqa: E402

MAX_CONTENT_HEIGHT = 360
#: GTK CSS cannot cap a widget's width, so labels wrap at this many characters
#: to keep the panel within the 250-285 px the design calls for.
LABEL_WIDTH_CHARS = 26
PANEL_MIN_WIDTH = 250


def _detail_for(task: VoxaTask) -> str:
    if task.detail:
        return task.detail
    if task.state is TaskState.FAILED:
        return task.error or "Failed"
    if task.state is TaskState.WAITING:
        return "Waiting for approval"
    if task.progress is not None:
        return f"{int(task.progress * 100)}%"
    return ""


class TaskPanel(Gtk.Box):
    """Task list rendered from the assistant model; hidden when empty."""

    def __init__(self, on_cancel: Callable[[str], None] | None = None) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add_css_class("voxa-task-panel")
        self.on_cancel = on_cancel
        self.set_size_request(PANEL_MIN_WIDTH, -1)

        self._content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        scroller = Gtk.ScrolledWindow()
        scroller.set_propagate_natural_height(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_max_content_height(MAX_CONTENT_HEIGHT)
        scroller.set_child(self._content)
        self.append(scroller)
        self.set_visible(False)

    def set_tasks(self, tasks: list[VoxaTask]) -> None:
        """Rebuild the panel from a task snapshot."""
        active = [t for t in tasks if t.state in (TaskState.RUNNING, TaskState.WAITING, TaskState.FAILED)]
        todo = [t for t in tasks if t.state is TaskState.QUEUED]

        child = self._content.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._content.remove(child)
            child = nxt

        if not active and not todo:
            self.set_visible(False)
            return

        self.set_visible(True)
        if active:
            self._content.append(self._section_heading("ACTIVE TASKS"))
            for task in active:
                self._content.append(self._task_row(task, cancellable=task.state is not TaskState.FAILED))
        if todo:
            self._content.append(self._section_heading("TO DO"))
            for task in todo:
                self._content.append(self._task_row(task, cancellable=False))

    def _section_heading(self, text: str) -> Gtk.Label:
        label = Gtk.Label(label=text)
        label.set_xalign(0)
        label.add_css_class("voxa-section")
        return label

    def _task_row(self, task: VoxaTask, *, cancellable: bool) -> Gtk.Box:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        bullet = Gtk.Label(label="●" if task.state is not TaskState.QUEUED else "□")
        bullet.add_css_class("voxa-task-bullet")

        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text.set_hexpand(True)
        title = Gtk.Label(label=task.title)
        title.set_xalign(0)
        title.set_wrap(True)
        title.set_max_width_chars(LABEL_WIDTH_CHARS)
        title.add_css_class("voxa-task-title")
        detail = Gtk.Label(label=_detail_for(task))
        detail.set_xalign(0)
        detail.set_wrap(True)
        detail.set_max_width_chars(LABEL_WIDTH_CHARS)
        detail.add_css_class("voxa-task-detail")
        if task.state is TaskState.FAILED:
            detail.add_css_class("error")
        text.append(title)
        text.append(detail)

        row.append(bullet)
        row.append(text)

        if cancellable and self.on_cancel is not None:
            cancel = Gtk.Button(icon_name="window-close-symbolic")
            cancel.add_css_class("flat")
            cancel.add_css_class("voxa-cancel")
            cancel.update_property([Gtk.AccessibleProperty.LABEL], ["Cancel task"])
            cancel.connect("clicked", lambda *_: self.on_cancel(task.id))
            row.append(cancel)

        return row
