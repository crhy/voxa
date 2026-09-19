from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.state import AssistantModel, TaskState  # noqa: E402
from voxa.ui.task_panel import TaskPanel  # noqa: E402


def _pump() -> None:
    ctx = GLib.MainContext.default()
    for _ in range(50):
        while ctx.iteration(False):
            pass


def _descendants(widget):
    child = widget.get_first_child()
    while child is not None:
        yield child
        yield from _descendants(child)
        child = child.get_next_sibling()


def _headings(panel) -> list[str]:
    return [
        w.get_text()
        for w in _descendants(panel)
        if isinstance(w, Gtk.Label) and w.has_css_class("voxa-section")
    ]


def _cancel_buttons(panel) -> list[Gtk.Button]:
    return [w for w in _descendants(panel) if isinstance(w, Gtk.Button) and w.has_css_class("voxa-cancel")]


def test_empty_panel_is_hidden() -> None:
    panel = TaskPanel()
    panel.set_tasks([])
    _pump()
    assert not panel.get_visible()
    model = AssistantModel()
    model.on_tasks_changed = panel.set_tasks
    task = model.add_task("Later")
    model.cancel_task(task.id)
    assert not panel.get_visible()
    assert not _headings(panel)


def test_sections_and_details_from_model() -> None:
    cancelled: list[str] = []
    panel = TaskPanel(on_cancel=cancelled.append)
    model = AssistantModel()
    model.on_tasks_changed = panel.set_tasks

    running = model.add_task("Update website", detail="Editing index.html")
    model.start_task(running.id)
    waiting = model.add_task("Send confirmation")
    model.start_task(waiting.id)
    model.request_choice(waiting.id, ["Now", "Later"])
    progress = model.add_task("Organize Downloads")
    model.start_task(progress.id)
    model.update_task(progress.id, progress=0.37)
    model.add_task("Check backup")
    _pump()

    assert panel.get_visible()
    headings = _headings(panel)
    assert headings == ["ACTIVE TASKS", "TO DO"]

    titles = [w.get_text() for w in _descendants(panel) if w.has_css_class("voxa-task-title")]
    assert titles == ["Update website", "Send confirmation", "Organize Downloads", "Check backup"]

    details = [w.get_text() for w in _descendants(panel) if w.has_css_class("voxa-task-detail")]
    assert "Editing index.html" in details
    assert "Waiting for approval" in details
    assert "37%" in details

    assert len(_cancel_buttons(panel)) == 3  # the three active rows; queued rows have none
    _cancel_buttons(panel)[0].emit("clicked")
    assert cancelled == [running.id]
    # The panel only reports the request; whoever owns the model acts on it.
    assert model.tasks[running.id].state is TaskState.RUNNING
    model.cancel_task(cancelled[0])
    assert model.tasks[running.id].state is TaskState.CANCELLED


def test_failed_task_shown_in_active_until_gone() -> None:
    panel = TaskPanel()
    model = AssistantModel()
    model.on_tasks_changed = panel.set_tasks
    task = model.add_task("Backup")
    model.start_task(task.id)
    model.fail_task(task.id, "disk full")
    _pump()
    assert panel.get_visible()
    assert _headings(panel) == ["ACTIVE TASKS"]
    errors = [w for w in _descendants(panel) if w.has_css_class("voxa-task-detail") and w.has_css_class("error")]
    assert errors and errors[0].get_text() == "disk full"


def test_long_titles_wrap_instead_of_widening_the_panel() -> None:
    panel = TaskPanel(on_cancel=lambda _task_id: None)
    model = AssistantModel()
    model.on_tasks_changed = panel.set_tasks
    task = model.add_task("Update the Spaced Linux website and republish every downloadable ISO image")
    model.start_task(task.id, "Editing website/index.html and checking every internal link on the page")
    _pump()

    _minimum, natural, _min_baseline, _nat_baseline = panel.measure(Gtk.Orientation.HORIZONTAL, -1)
    assert 250 <= natural <= 290
