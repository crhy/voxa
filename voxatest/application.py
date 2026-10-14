from __future__ import annotations

import sys

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gio, Gtk  # noqa: E402

from . import APP_ID, APP_NAME, APP_VERSION  # noqa: E402
from .window import VoxaTestWindow  # noqa: E402


class VoxaTestApplication(Adw.Application):
    def __init__(self, cases: str = "", autostart: bool = False) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.window: VoxaTestWindow | None = None
        self._cases = cases
        self._autostart = autostart
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<Control>q"])

        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self._show_about)
        self.add_action(about_action)

    def do_activate(self) -> None:
        if self.window is None:
            self.window = VoxaTestWindow(self, cases=self._cases, autostart=self._autostart)
        self.window.present()

    def _show_about(self, *_args) -> None:
        about = Adw.AboutDialog(
            application_name=APP_NAME,
            application_icon=APP_ID,
            developer_name="crhy",
            version=APP_VERSION,
            website="https://github.com/crhy/voxa",
            issue_url="https://github.com/crhy/voxa/issues",
            license_type=Gtk.License.MIT_X11,
            comments="End-to-end test harness for Voxa: speak prompts, capture replies, grade results.",
        )
        about.add_credit_section("Built with", ["GTK 4", "libadwaita", "Faster Whisper", "GStreamer", "Ollama", "Edge TTS", "eSpeak NG"])
        about.present(self.window)


def main() -> int:
    """Launch the standalone VoxaTest desktop application.

    ``--cases FILE`` chooses the case file and ``--autostart`` begins the run at once, so a
    recorded session needs no clicks.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="voxatest-gui", add_help=False)
    parser.add_argument("--cases", default="")
    parser.add_argument("--autostart", action="store_true")
    options, rest = parser.parse_known_args(sys.argv[1:])
    app = VoxaTestApplication(cases=options.cases, autostart=options.autostart)
    return app.run([sys.argv[0], *rest])


if __name__ == "__main__":
    raise SystemExit(main())
