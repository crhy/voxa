from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.choice_overlay import ChoiceOverlay  # noqa: E402
from voxa.ui.state import AssistantModel  # noqa: E402


def _buttons(overlay) -> list[Gtk.Button]:
    buttons = []
    child = overlay._choices.get_first_child()
    while child is not None:
        if isinstance(child, Gtk.Button):
            buttons.append(child)
        child = child.get_next_sibling()
    return buttons


def test_hidden_by_default() -> None:
    overlay = ChoiceOverlay()
    assert not overlay.get_visible()


def test_choices_render_and_callback_hides() -> None:
    model = AssistantModel()
    task = model.add_task("Send confirmation")
    model.start_task(task.id)
    model.request_choice(task.id, ["Now", "Later"])

    answered: list[tuple[str, str]] = []
    overlay = ChoiceOverlay(on_choice=lambda task_id, choice: answered.append((task_id, choice)))
    overlay.show_choices(task)
    assert overlay.get_visible()
    assert overlay._title.get_text() == "Send confirmation"

    buttons = _buttons(overlay)
    assert [b.get_child().get_text() for b in buttons] == ["Now", "Later", "Cancel"]
    assert buttons[0].has_css_class("suggested-action")
    buttons[0].emit("clicked")
    assert answered == [(task.id, "Now")]
    assert not overlay.get_visible()


def test_cancel_included_when_offered() -> None:
    overlay = ChoiceOverlay()
    from voxa.ui.state import VoxaTask
    task = VoxaTask(title="Pick", choices=["Cancel", "Keep"])
    overlay.show_choices(task)
    labels = [b.get_child().get_text() for b in _buttons(overlay)]
    assert labels == ["Cancel", "Keep"]


def test_escape_hides_without_choice() -> None:
    answered: list[tuple[str, str]] = []
    overlay = ChoiceOverlay(on_choice=lambda task_id, choice: answered.append((task_id, choice)))
    from voxa.ui.state import VoxaTask
    overlay.show_choices(VoxaTask(title="Pick", choices=["Yes"]))
    overlay._on_key(None, Gdk.KEY_Escape, 0, 0)
    assert not overlay.get_visible()
    assert answered == []
