"""Render ONE frame of a :class:`PhotoFaceRenderer` offscreen to a PNG.

``python3 -m voxa.ui.photo_face_snapshot <character-id> <out.png> [--size 480]
[--mode prerendered|still] [--viseme viseme_aa] [--blink 1.0]``
"""

from __future__ import annotations

import logging
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from .photo_face import MODES, PhotoFaceRenderer  # noqa: E402

CSS = """
.voxa-photo-face { background-color: transparent; }
.voxa-photo-clip { border-radius: 18px; }
"""


def _install_css() -> None:
    provider = Gtk.CssProvider()
    provider.load_from_string(CSS)
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


def _parse_args(argv: list[str]) -> tuple[str, str, int, str, str | None, float]:
    positional: list[str] = []
    size = 480
    mode = "prerendered"
    viseme: str | None = None
    blink = 0.0
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--size":
            size = int(argv[i + 1])
            i += 2
        elif arg == "--mode":
            mode = argv[i + 1]
            i += 2
        elif arg == "--viseme":
            viseme = argv[i + 1]
            i += 2
        elif arg == "--blink":
            blink = float(argv[i + 1])
            i += 2
        else:
            positional.append(arg)
            i += 1
    if len(positional) < 2:
        raise SystemExit("usage: <character-id> <out.png> [--size N] [--mode M] [--viseme V] [--blink B]")
    if mode not in MODES:
        mode = "prerendered"
    return positional[0], positional[1], size, mode, viseme, blink


def main(argv: list[str]) -> int:
    character_id, out_path, size, mode, viseme, blink = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    renderer = PhotoFaceRenderer()
    renderer.widget.set_size_request(size, size)
    renderer.set_character(character_id)
    renderer.set_mode(mode)

    # Force the pose through the renderer's own tick so the snapshot shows exactly what the app would draw.
    renderer._forced_viseme = {viseme: 1.0} if viseme else None
    renderer._forced_blink = blink
    if mode != "still" and renderer._pack is not None:
        for entry in renderer._pack.mouth:
            renderer._cache.get(str(renderer._pack.directory / entry.file))
        for name in renderer._pack.blink:
            renderer._cache.get(str(renderer._pack.directory / name))
    renderer._cache.get(renderer._portrait)

    _install_css()
    window = Gtk.Window(title="voxa-photo-face-snapshot")
    window.set_default_size(size, size)
    window.set_child(renderer.widget)
    window.present()

    context = GLib.MainContext.default()
    # The renderer's tick never goes idle, so run a fixed number of loop turns instead of draining the queue.
    deadline = GLib.get_monotonic_time() + 1_500_000
    while GLib.get_monotonic_time() < deadline:
        context.iteration(False)

    paintable = Gtk.WidgetPaintable.new(window)
    snap = Gtk.Snapshot()
    paintable.snapshot(snap, size, size)
    node = snap.to_node()
    texture = window.get_native().get_renderer().render_texture(node, None)
    if not texture.save_to_png(out_path):
        logging.error("could not write %s", out_path)
        return 1
    logging.info("wrote %s (%dx%d)", out_path, size, size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
