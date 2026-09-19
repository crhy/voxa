from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from gi.repository import GLib  # noqa: E402

from voxa.ui.shell import AssistantShell, build_header  # noqa: E402
from voxa.ui.state import AssistantModel, AssistantState, TaskState  # noqa: E402


def _pump(iterations: int = 200) -> None:
    """Push the main context until widgets realize, with a hard bound."""
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        if not ctx.iteration(False):
            continue
    for _ in range(20):
        ctx.iteration(False)


def _present(shell: AssistantShell, width: int, height: int) -> Gtk.Window:
    window = Gtk.Window()
    window.set_default_size(width, height)
    window.set_child(shell)
    window.present()
    _pump()
    return window


def _center_offset(shell: AssistantShell, window: Gtk.Window) -> float:
    ok_shell, shell_rect = shell.compute_bounds(window)
    ok_view, view_rect = shell.assistant_view.compute_bounds(shell)
    assert ok_shell and ok_view
    view_center = view_rect.get_x() + view_rect.get_width() / 2
    return view_center - shell_rect.get_width() / 2


def test_assistant_is_true_center_with_and_without_tasks() -> None:
    for width, height in ((1200, 760), (850, 600)):
        model = AssistantModel()
        shell = AssistantShell(model)
        window = _present(shell, width, height)

        assert abs(_center_offset(shell, window)) <= 2.0

        task = model.add_task("Sending weekly report")
        model.start_task(task.id, "Working…")
        _pump()
        assert shell.task_panel.get_visible()
        assert abs(_center_offset(shell, window)) <= 2.0

        model.cancel_task(task.id)
        _pump()
        assert not shell.task_panel.get_visible()
        assert abs(_center_offset(shell, window)) <= 2.0
        window.close()


def test_task_panel_floats_beside_not_over_center() -> None:
    model = AssistantModel()
    task = model.add_task("Updating Spaced Linux")
    model.start_task(task.id, "Step 3 of 6")
    shell = AssistantShell(model)
    window = _present(shell, 1200, 760)

    ok_panel, panel_rect = shell.task_panel.compute_bounds(shell)
    ok_view, view_rect = shell.assistant_view.compute_bounds(shell)
    assert ok_panel and ok_view
    assert panel_rect.get_x() < view_rect.get_x()
    window.close()


def test_state_wiring_caption_and_buttons() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    _present(shell, 1200, 760)

    model.set_state(AssistantState.LISTENING, "x")
    _pump()
    assert shell.assistant_view.caption.get_text() == "x"
    assert not shell.status_controls.offline_button.has_css_class("selected")
    assert shell.assistant_view.has_css_class("listening")
    assert not shell.assistant_view.has_css_class("thinking")

    model.set_state(AssistantState.OFFLINE)
    _pump()
    assert shell.status_controls.offline_button.has_css_class("selected")
    assert shell.status_controls.active_button.has_css_class("dimmed")
    assert not shell.assistant_view.has_css_class("listening")


def test_tasks_and_choice_wiring() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    _present(shell, 1200, 760)

    task = model.add_task("Publish release notes")
    model.start_task(task.id)
    _pump()
    assert shell.task_panel.get_visible()

    model.request_choice(task.id, ["newest ISO", "release ISO", "Cancel"])
    _pump()
    assert shell.choice_overlay.get_visible()

    buttons = []
    child = shell.choice_overlay._choices.get_first_child()
    while child is not None:
        if isinstance(child, Gtk.Button):
            buttons.append(child)
        child = child.get_next_sibling()
    buttons[0].emit("clicked")

    assert model.tasks[task.id].state is TaskState.RUNNING
    assert not shell.choice_overlay.get_visible()


def test_callbacks_fire() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    fired: list[str] = []
    shell.on_active = lambda: fired.append("active")
    shell.on_offline = lambda: fired.append("offline")
    shell.on_attach = lambda: fired.append("attach")
    shell.on_model_selected = lambda name: fired.append(name)
    _present(shell, 1200, 760)

    shell.status_controls.active_button.emit("clicked")
    shell.status_controls.offline_button.emit("clicked")
    shell.attachment_button.emit("clicked")
    shell.set_models(["qwen3.8-flash-next", "llama3.1"], selected="qwen3.8-flash-next")
    shell.model_selector._dropdown.set_selected(1)

    assert fired == ["active", "offline", "attach", "llama3.1"]
    assert shell.model_selector.get_selected() == "llama3.1"


def test_header_has_no_window_title() -> None:
    header = build_header()
    title = header.get_title_widget()
    assert isinstance(title, Gtk.Label)
    assert title.get_text() == ""
    assert header.get_first_child() is not None


def test_header_stays_compact() -> None:
    """A Gtk.Picture badge once rendered at texture size and made the header ~400 px tall."""
    header = build_header()
    window = Gtk.Window()
    window.set_default_size(1200, 760)
    window.set_titlebar(None)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(header)
    window.set_child(box)
    window.present()
    _pump()
    _minimum, natural, _mb, _nb = header.measure(Gtk.Orientation.VERTICAL, -1)
    assert natural < 64


def test_assistant_stays_centered_under_a_real_header() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    view = Adw.ToolbarView()
    view.add_top_bar(build_header())
    view.set_content(shell)
    window = Adw.Window()
    window.set_default_size(1200, 760)
    window.set_content(view)
    window.present()
    _pump()

    ok_shell, shell_rect = shell.compute_bounds(window)
    ok_view, view_rect = shell.assistant_view.compute_bounds(shell)
    assert ok_shell and ok_view
    assert abs((view_rect.get_x() + view_rect.get_width() / 2) - shell_rect.get_width() / 2) <= 2
    assert abs((view_rect.get_y() + view_rect.get_height() / 2) - shell_rect.get_height() / 2) <= 40
    assert shell_rect.get_y() < 64  # the header did not push the stage down


def test_choice_card_never_overlaps_the_task_list() -> None:
    model = AssistantModel()
    shell = AssistantShell(model)
    for title in ("Update site", "Sync backup", "Check installer", "Publish notes"):
        model.start_task(model.add_task(title, "working").id)
    waiting = model.add_task("Choose an image")
    model.start_task(waiting.id)
    model.request_choice(waiting.id, ["newest ISO", "release ISO", "Cancel"])
    window = _present(shell, 850, 600)
    _pump()

    ok_panel, panel = shell.task_panel.compute_bounds(shell)
    ok_card, card = shell.choice_overlay.compute_bounds(shell)
    assert ok_panel and ok_card and shell.choice_overlay.get_visible()
    separated_vertically = card.get_y() + card.get_height() <= panel.get_y() + 1
    separated_horizontally = card.get_x() + card.get_width() <= panel.get_x() or panel.get_x() + panel.get_width() <= card.get_x()
    assert separated_vertically or separated_horizontally
    window.destroy()
