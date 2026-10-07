"""The desktop clipboard, readable and writable from any thread (GTK itself is only touched on the main loop)."""

from __future__ import annotations

import threading
import time

import gi

gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib  # noqa: E402


def _on_main(action, timeout: float) -> None:
    """Run action(done) on the GTK main loop and wait until it calls done() or the timeout passes."""
    finished = threading.Event()

    def start() -> bool:
        try:
            action(finished.set)
        except Exception:
            finished.set()
        return GLib.SOURCE_REMOVE

    GLib.idle_add(start)
    if threading.current_thread() is threading.main_thread():
        context = GLib.MainContext.default()
        deadline = time.monotonic() + timeout
        while not finished.is_set() and time.monotonic() < deadline:
            context.iteration(False)
            time.sleep(0.002)
    else:
        finished.wait(timeout)


def read_text(timeout: float = 1.5) -> str:
    """The clipboard's text, or "" when it holds no text or cannot be read in time."""
    box: list[str] = []

    def action(done) -> None:
        clipboard = Gdk.Display.get_default().get_clipboard()

        def ready(source, result) -> None:
            try:
                box.append(source.read_text_finish(result) or "")
            except Exception:
                pass
            done()

        clipboard.read_text_async(None, ready)

    _on_main(action, timeout)
    return box[0] if box else ""


def write_text(text: str, timeout: float = 1.0) -> None:
    """Put text on the clipboard."""

    def action(done) -> None:
        Gdk.Display.get_default().get_clipboard().set_content(Gdk.ContentProvider.new_for_value(text))
        done()

    _on_main(action, timeout)
