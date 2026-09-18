"""Compact task panel (issue #5).

Overlay-friendly: fixed narrow width, no expand, so the avatar keeps its
absolute center. Polls the store once a second (GTK thread reads under the
store lock) instead of cross-thread signals.
"""

from __future__ import annotations

from gi.repository import GLib, Gtk

from voxa.tasks import Task, TaskState, TaskStore
from voxa.ui.choice_overlay import ChoiceOverlay


def _row_text(task: Task) -> str:
    if task.progress is not None and task.state == TaskState.RUNNING:
        return f"{task.detail} · {task.progress:.0%}" if task.detail else f"{task.progress:.0%}"
    if task.state == TaskState.RUNNING:
        return task.detail or "Working…"
    return task.detail


class TaskPanel(Gtk.Box):
    """ACTIVE TASKS + TO DO lists bound to a :class:`TaskStore`."""

    WIDTH = 272

    def __init__(self, store: TaskStore) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add_css_class("voxa-task-panel")
        self.set_size_request(self.WIDTH, -1)
        self._store = store

        self._active_list = self._section("ACTIVE TASKS")
        self._todo_list = self._section("TO DO")
        self.refresh()
        GLib.timeout_add_seconds(1, self.refresh)

    def _section(self, heading: str) -> Gtk.Box:
        label = Gtk.Label(label=heading, xalign=0)
        label.add_css_class("voxa-panel-heading")
        label.add_css_class("dim-label")
        items = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.append(label)
        self.append(items)
        return items

    def refresh(self) -> bool:
        """Rebuild rows from the store. Returns True to keep the poll timer."""
        tasks = self._store.tasks()
        self._fill(
            self._active_list,
            [task for task in tasks if task.state in (TaskState.RUNNING, TaskState.WAITING)],
        )
        self._fill(
            self._todo_list, [task for task in tasks if task.state == TaskState.QUEUED]
        )
        return True

    def _fill(self, container: Gtk.Box, tasks: list[Task]) -> None:
        child = container.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            container.remove(child)
            child = next_child
        if not tasks:
            empty = Gtk.Label(label="Nothing yet.", xalign=0)
            empty.add_css_class("dim-label")
            container.append(empty)
            return
        for task in tasks:
            container.append(self._row(task))

    def _row(self, task: Task) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        row.add_css_class("voxa-task-row")
        title = Gtk.Label(label=f"• {task.title}", xalign=0)
        title.set_wrap(True)
        row.append(title)
        detail = _row_text(task)
        if detail:
            sub = Gtk.Label(label=detail, xalign=0)
            sub.add_css_class("dim-label")
            sub.set_wrap(True)
            row.append(sub)
        if task.state == TaskState.WAITING and task.requires_user_input:
            overlay = ChoiceOverlay()
            overlay.show_for(
                "Needs a choice:",
                task.choices,
                lambda choice, task_id=task.id: (
                    self._store.resolve_choice(task_id, choice),
                    self.refresh(),
                ),
            )
            row.append(overlay)
        return row
