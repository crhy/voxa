"""Centered assistant shell: a true-center stage with floating overlays.

Every secondary element (task rail, choice card, attachment button, model
picker, status controls) is an ``Gtk.Overlay`` child, so none of them steals
space from the stage and the assistant stays at the physical center of the
window. Model mutations arrive from worker threads and are marshalled into the
GTK thread with ``GLib.idle_add`` before any widget is touched.
"""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from voxa.config import Settings  # noqa: E402

from .assistant_view import BADGE_PATH, AssistantView  # noqa: E402
from .choice_overlay import ChoiceOverlay  # noqa: E402
from .exchange_panel import ExchangePanel  # noqa: E402
from .model_selector import ModelSelector  # noqa: E402
from .notice import NoticeBar  # noqa: E402
from .state import AssistantModel, AssistantState  # noqa: E402
from .status_controls import StatusControls  # noqa: E402
from .task_panel import TaskPanel  # noqa: E402

BADGE_SIZE = 24
EDGE_MARGIN = 24


class AssistantShell(Gtk.Overlay):
    """The whole application body: centered assistant plus floating overlays."""

    def __init__(self, model: AssistantModel, settings: Settings | None = None) -> None:
        super().__init__()
        self.model = model
        self.settings = settings

        self.on_active: Callable[[], None] | None = None
        self.on_offline: Callable[[], None] | None = None
        self.on_attach: Callable[[], None] | None = None
        self.on_model_selected: Callable[[str], None] | None = None
        self.on_backend_selected: Callable[[str], None] | None = None

        self.set_hexpand(True)
        self.set_vexpand(True)

        # The stage owns the full dimensions of the shell. Spacers take the
        # slack so the assistant is centered both horizontally and vertically.
        stage = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        stage.set_halign(Gtk.Align.CENTER)
        stage.set_valign(Gtk.Align.CENTER)
        stage.set_hexpand(True)
        stage.set_vexpand(True)

        top_spacer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        top_spacer.set_vexpand(True)
        bottom_spacer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        bottom_spacer.set_vexpand(True)

        character_id = "" if settings is None else settings.character_id
        self.assistant_view = AssistantView(character_id)
        self.assistant_view.set_halign(Gtk.Align.CENTER)
        self.assistant_view.set_valign(Gtk.Align.CENTER)

        stage.append(top_spacer)
        stage.append(self.assistant_view)
        stage.append(bottom_spacer)
        self.set_child(stage)

        # Floating task rail.
        self.task_panel = TaskPanel(on_cancel=self.model.cancel_task)
        self.task_panel.set_halign(Gtk.Align.START)
        self.task_panel.set_valign(Gtk.Align.CENTER)
        self.task_panel.set_margin_start(EDGE_MARGIN)
        self.add_overlay(self.task_panel)

        # Latest question and answer, kept on screen until the next question.
        self.exchange_panel = ExchangePanel()
        self.exchange_panel.set_halign(Gtk.Align.END)
        self.exchange_panel.set_valign(Gtk.Align.CENTER)
        self.exchange_panel.set_margin_end(EDGE_MARGIN)
        self.add_overlay(self.exchange_panel)

        # Notifications slide in at the top instead of crowding the controls.
        self.notice_bar = NoticeBar()
        self.add_overlay(self.notice_bar)

        # Contextual choice card, above the bottom controls.
        self.choice_overlay = ChoiceOverlay(on_choice=self._answer_choice)
        # Top-left, level with the task rail: it can never overlap the centered
        # tasks list, the assistant, or the bottom controls at any window size.
        self.choice_overlay.set_halign(Gtk.Align.START)
        self.choice_overlay.set_valign(Gtk.Align.START)
        self.choice_overlay.set_margin_start(EDGE_MARGIN)
        self.choice_overlay.set_margin_top(EDGE_MARGIN)
        self.add_overlay(self.choice_overlay)

        # Bottom row: attachment, model picker, status controls.
        self.attachment_button = Gtk.Button(icon_name="mail-attachment-symbolic")
        self.attachment_button.add_css_class("flat")
        self.attachment_button.set_tooltip_text("Attach files")
        self.attachment_button.update_property(
            [Gtk.AccessibleProperty.LABEL], ["Attach files"]
        )
        self.attachment_button.set_halign(Gtk.Align.START)
        self.attachment_button.set_valign(Gtk.Align.END)
        self.attachment_button.set_margin_start(EDGE_MARGIN)
        self.attachment_button.set_margin_bottom(EDGE_MARGIN)
        self.attachment_button.connect("clicked", lambda *_: self._fire(self.on_attach))
        self.add_overlay(self.attachment_button)

        self.model_selector = ModelSelector(
            on_model_selected=self._model_selected,
            on_backend_selected=self._backend_selected,
        )
        self.model_selector.set_halign(Gtk.Align.CENTER)
        self.model_selector.set_valign(Gtk.Align.END)
        self.model_selector.set_margin_bottom(EDGE_MARGIN)
        self.add_overlay(self.model_selector)

        self.status_controls = StatusControls(
            on_active=self._activate,
            on_offline=self._go_offline,
        )
        self.status_controls.set_halign(Gtk.Align.END)
        self.status_controls.set_valign(Gtk.Align.END)
        self.status_controls.set_margin_end(EDGE_MARGIN)
        self.status_controls.set_margin_bottom(EDGE_MARGIN)
        self.add_overlay(self.status_controls)

        self.model.on_state_changed = self._on_state_changed
        self.model.on_tasks_changed = self._on_tasks_changed

        # Render whatever the model already holds.
        self._apply_state(self.model.state, self.model.detail)
        self._apply_tasks(list(self.model.tasks.values()))

    # ------------------------------------------------------------ public API

    def set_models(
        self, models: list[str], selected: str = "", backend: str = "llamacpp"
    ) -> None:
        """Fill the model dropdown and select the active backend."""
        self.model_selector.set_models(models, selected)
        self.model_selector.set_backend(backend)

    def set_backend(self, backend: str) -> None:
        """Select the backend in the main-shell picker without firing the callback."""
        self.model_selector.set_backend(backend)

    def show_notice(self, text: str) -> None:
        self.notice_bar.show_text(text)

    def set_audio_level(self, level: float) -> None:
        """Forward the microphone level to the assistant view."""
        self.assistant_view.set_audio_level(level)

    # -------------------------------------------------------------- internals

    def _on_state_changed(self, state: AssistantState, detail: str) -> None:
        GLib.idle_add(self._apply_state, state, detail)

    def _apply_state(self, state: AssistantState, detail: str) -> bool:
        self.assistant_view.set_state(state, detail)
        self.status_controls.set_state(state)
        self.assistant_view.set_listening(state is AssistantState.LISTENING)
        self.assistant_view.set_thinking(state is AssistantState.THINKING)
        self.assistant_view.set_speaking(state is AssistantState.SPEAKING)
        return False

    def _on_tasks_changed(self, tasks) -> None:
        GLib.idle_add(self._apply_tasks, tasks)

    def _apply_tasks(self, tasks) -> bool:
        self.task_panel.set_tasks(tasks)
        waiting = next((task for task in tasks if task.requires_user_input), None)
        if waiting is not None:
            self.choice_overlay.show_choices(waiting)
        else:
            self.choice_overlay.hide_choices()
        return False

    def _answer_choice(self, task_id: str, choice: str) -> None:
        if choice == "Cancel":
            self.model.cancel_task(task_id)
        else:
            self.model.resolve_choice(task_id, choice)

    def _model_selected(self, name: str) -> None:
        self._fire(self.on_model_selected, name)

    def _backend_selected(self, backend: str) -> None:
        self._fire(self.on_backend_selected, backend)

    def _activate(self) -> None:
        self._fire(self.on_active)

    def _go_offline(self) -> None:
        self._fire(self.on_offline)

    @staticmethod
    def _fire(callback: Callable | None, *args) -> None:
        if callback is not None:
            callback(*args)


def build_header(menu_model: Gio.MenuModel | None = None) -> Adw.HeaderBar:
    """A clean header: badge plus the name on the left, menu button on the right."""
    header = Adw.HeaderBar()

    brand = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    try:
        texture = Gdk.Texture.new_from_filename(str(BADGE_PATH))
        # Gtk.Image with a pixel size is a hard cap; a Gtk.Picture would render the
        # 1024 px texture at full size and blow the header up.
        badge = Gtk.Image.new_from_paintable(texture)
        badge.set_pixel_size(BADGE_SIZE)
        badge.set_valign(Gtk.Align.CENTER)
        brand.append(badge)
    except Exception:
        pass

    name = Gtk.Label(label="Voxa")
    name.add_css_class("voxa-brand")
    brand.append(name)
    header.pack_start(brand)
    header.set_title_widget(Gtk.Label())

    if menu_model is not None:
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_menu_model(menu_model)
        header.pack_end(menu_button)

    return header
