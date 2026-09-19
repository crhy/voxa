"""Developer preview: renders the assistant shell with demo data.

Run it to see the layout, or render it to a PNG::

    python3 -m voxa.ui.preview --screenshot /tmp/voxa-preview.png
"""

from __future__ import annotations

import argparse

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from .shell import AssistantShell, build_header  # noqa: E402
from .state import AssistantModel, AssistantState  # noqa: E402
from .styles import install_styles  # noqa: E402

MODELS = ["qwen3.8-flash-next", "llama3.1"]
BACKEND = "llama.cpp"
CYCLE_STATES = [
    AssistantState.READY,
    AssistantState.LISTENING,
    AssistantState.THINKING,
    AssistantState.SPEAKING,
    AssistantState.WORKING,
    AssistantState.WAITING,
]
CAPTURE_DELAY_MS = 2000


def _demo_model() -> AssistantModel:
    model = AssistantModel()
    model.set_state(AssistantState.READY)

    editing = model.add_task("Update spaced-linux site")
    model.start_task(editing.id, "Editing index.html")

    backup = model.add_task("Sync backup")
    model.start_task(backup.id)
    model.update_task(backup.id, progress=0.37)

    model.add_task("Publish release notes")

    installer = model.add_task("Choose installer image")
    model.start_task(installer.id)
    model.request_choice(installer.id, ["newest ISO", "release ISO", "Cancel"])

    return model


class PreviewApp(Adw.Application):
    def __init__(self, width: int, height: int, screenshot: str | None) -> None:
        super().__init__(application_id="io.github.crhy.voxa.preview")
        self.width = width
        self.height = height
        self.screenshot = screenshot
        self.model = AssistantModel()
        self.state_index = 0

    def do_activate(self) -> None:
        install_styles()

        self.model = _demo_model()
        shell = AssistantShell(self.model)
        shell.set_models(MODELS, selected=MODELS[0], backend_label=BACKEND)

        menu = Gio.Menu()
        menu.append_item(Gio.MenuItem.new("Preferences", "app.preferences"))
        menu.append_item(Gio.MenuItem.new("About Voxa", "app.about"))

        view = Adw.ToolbarView()
        view.add_top_bar(build_header(menu))
        view.set_content(shell)

        window = Adw.ApplicationWindow(application=self)
        window.set_default_size(self.width, self.height)
        window.set_content(view)
        window.present()

        if self.screenshot is not None:
            self.model.set_state(AssistantState.READY)
            GLib.timeout_add(CAPTURE_DELAY_MS, self._capture, window)
        else:
            GLib.timeout_add(3000, self._cycle_state)

    def _cycle_state(self) -> bool:
        self.state_index = (self.state_index + 1) % len(CYCLE_STATES)
        self.model.set_state(CYCLE_STATES[self.state_index])
        return True

    def _capture(self, window: Adw.ApplicationWindow) -> bool:
        width, height = window.get_width(), window.get_height()
        paintable = Gtk.WidgetPaintable.new(window)
        snapshot = Gtk.Snapshot()
        paintable.snapshot(snapshot, width, height)
        node = snapshot.to_node()
        renderer = window.get_native().get_renderer()
        texture = renderer.render_texture(node, None)
        texture.save_to_png(self.screenshot)
        print(self.screenshot)
        self.quit()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Voxa interface preview")
    parser.add_argument("--screenshot", metavar="PATH", help="render the window to a PNG and exit")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    args = parser.parse_args()

    app = PreviewApp(width=args.width, height=args.height, screenshot=args.screenshot)
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
