"""New-interface shell (issue #5): header, center stage, bottom bar.

Layout contract:

- The header holds only the Voxa name (left) and the menu (right).
- The avatar is centered in the *entire* window: it is the overlay's main
  child, and the task panel is an overlay child, so the panel never shifts
  the avatar off absolute center.
- The bottom bar holds the attachment escape hatch, the model selector,
  a tiny GPU readout, and the ACTIVE / OFFLINE buttons.

MainWindow owns all backends and wires the ``on_*`` callbacks.
"""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Adw, Gio, Gtk

from voxa.tasks import TaskStore
from voxa.ui.assistant_view import AssistantView
from voxa.ui.model_selector import ModelSelector
from voxa.ui.status_controls import ActiveOfflineControls
from voxa.ui.styles import install_css
from voxa.ui.task_panel import TaskPanel


class AssistantShell(Gtk.Box):
    """The whole main-view content below the toolbar."""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        on_active: Callable[[], None],
        on_offline: Callable[[], None],
        on_model_selected: Callable[[str], None],
        on_attach: Callable[[], None],
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        install_css()

        header = Adw.HeaderBar()
        title = Gtk.Label(label="Voxa")
        title.add_css_class("title-1")
        header.pack_start(title)
        header.pack_end(self._menu_button())
        self.append(header)

        center = Gtk.Overlay()
        center.set_hexpand(True)
        center.set_vexpand(True)
        self.assistant = AssistantView()
        center.set_child(self.assistant)
        self.tasks = TaskPanel(task_store)
        self.tasks.set_halign(Gtk.Align.START)
        self.tasks.set_valign(Gtk.Align.FILL)
        self.tasks.set_margin_top(12)
        self.tasks.set_margin_bottom(12)
        self.tasks.set_margin_start(12)
        center.add_overlay(self.tasks)
        self.append(center)

        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        bottom.set_margin_top(8)
        bottom.set_margin_bottom(12)
        bottom.set_margin_start(16)
        bottom.set_margin_end(16)
        attach = Gtk.Button(icon_name="mail-attachment-symbolic")
        attach.set_tooltip_text("Attach a file (escape hatch)")
        attach.connect("clicked", lambda _b: on_attach())
        self.models = ModelSelector()
        self.models.connect_changed(on_model_selected)
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.gpu_label = Gtk.Label(label="")
        self.gpu_label.add_css_class("dim-label")
        self.gpu_label.set_width_chars(8)
        self.agent = ActiveOfflineControls()
        self.agent.connect_active(on_active)
        self.agent.connect_offline(on_offline)
        bottom.append(attach)
        bottom.append(self.models)
        bottom.append(spacer)
        bottom.append(self.gpu_label)
        bottom.append(self.agent)
        self.append(bottom)

    @staticmethod
    def _menu_button() -> Gtk.Widget:
        menu = Gio.Menu()
        menu.append("Preferences", "win.preferences")
        menu.append("Ask AI", "win.ask")
        menu.append("Speak reply", "win.speak")
        menu.append("Manage models…", "win.models")
        menu.append("Keyboard Shortcuts", "win.shortcuts")
        menu.append("About Voxa", "app.about")
        return Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)

    def set_agent_state(self, *, running: bool, offline: bool) -> None:
        self.agent.set_agent_state(running=running, offline=offline)

    def set_gpu_text(self, text: str, tooltip: str = "") -> None:
        self.gpu_label.set_text(text)
        self.gpu_label.set_tooltip_text(tooltip)
