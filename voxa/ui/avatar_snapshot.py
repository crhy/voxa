"""Render a GL avatar to a PNG without a desktop (Task H2 snapshot CLI).

Usage::

    python3 -m voxa.ui.avatar_snapshot <character-id-or-path.glb> <out.png> [--size 480] [--speaking]

The tool builds a :class:`Gtk.Window` holding the GLArea avatar widget, pumps the
main loop until at least five frames have rendered (bounded by a ten-second
timeout), reads the framebuffer with ``glReadPixels`` inside the render callback,
and writes the result as a PNG. It exits 0 on success and 1 with a message when
GL cannot start, so it is safe to run under ``xvfb-run -a`` with software GL.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402

from .avatar_3d import Gl3DFaceRenderer  # noqa: E402
from .avatars import get_avatar, model_is_downloaded  # noqa: E402


class SnapshotView:
    """Minimal stand-in for :class:`AssistantView` used by the renderer."""

    def _on_3d_renderer_ready(self, renderer) -> bool:
        return False

    def _fallback_3d_renderer(self) -> None:
        pass

    def add_css_class(self, name: str) -> None:
        pass

    def remove_css_class(self, name: str) -> None:
        pass


class SnapshotRenderer(Gl3DFaceRenderer):
    """A :class:`Gl3DFaceRenderer` that captures its framebuffer each frame."""

    def __init__(self, view) -> None:
        super().__init__(view)
        self._frames = 0
        self._pixels: bytes | None = None
        self._width = 0
        self._height = 0

    def _on_render(self, area, context, *args):
        ok = super()._on_render(area, context, *args)
        if ok and self._gl_ok:
            self._capture(area)
        return ok

    def _capture(self, area) -> None:
        """Read the framebuffer after the render callback and flip it upright."""
        import OpenGL.GL as gl

        width = max(1, area.get_width())
        height = max(1, area.get_height())
        raw = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        row = width * 4
        flipped = bytearray()
        for y in range(height - 1, -1, -1):
            flipped += raw[y * row:(y + 1) * row]
        self._pixels = bytes(flipped)
        self._width = width
        self._height = height
        self._frames += 1


def _resolve_model_path(argument: str) -> str | None:
    """Resolve a character id or a ``.glb`` path to an existing model path."""
    if argument.endswith(".glb"):
        path = os.path.expanduser(argument)
        return path if os.path.exists(path) else None

    avatar = get_avatar(argument)
    if avatar is not None and model_is_downloaded(avatar):
        return str(avatar.model_path)
    return None


def _parse_args(argv: list[str]) -> tuple[str, str, int, bool, bool, float, bool]:
    """Return ``(source, out_path, size, speaking, verbose, pose_time, blink)`` from the CLI arguments."""
    positional: list[str] = []
    size = 480
    speaking = False
    verbose = False
    pose_time = 0.0
    blink = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--size":
            index += 1
            if index >= len(argv):
                raise SystemExit("--size needs a value")
            size = int(argv[index])
        elif token == "--time":
            index += 1
            if index >= len(argv):
                raise SystemExit("--time needs a value")
            pose_time = float(argv[index])
        elif token == "--speaking":
            speaking = True
        elif token == "--blink":
            blink = True
        elif token == "--verbose":
            verbose = True
        elif token.startswith("--"):
            raise SystemExit(f"unknown option: {token}")
        else:
            positional.append(token)
        index += 1

    if len(positional) < 2:
        raise SystemExit("usage: <character-id-or-path.glb> <out.png> [--size N] [--time T] [--speaking] [--blink] [--verbose]")
    return positional[0], positional[1], size, speaking, verbose, pose_time, blink


def main(argv: list[str]) -> int:
    """Entry point for ``python3 -m voxa.ui.avatar_snapshot``."""
    source, out_path, size, speaking, verbose, pose_time, blink = _parse_args(argv)
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    path = _resolve_model_path(source)
    if path is None:
        logging.error("model for %r is not available", source)
        return 1

    os.environ["VOXA_AVATAR_MODEL"] = path

    view = SnapshotView()
    renderer = SnapshotRenderer(view)
    renderer._area.set_size_request(size, size)
    renderer._pose_frozen = True
    renderer._pose_time = pose_time
    renderer._shader_time = pose_time
    renderer._blink_force = blink
    if speaking:
        renderer._viseme = {"viseme_aa": 1.0}

    window = Gtk.Window(title="voxa-snapshot")
    window.set_default_size(size, size)
    window.set_child(renderer.widget)
    window.present()

    # Pump the main loop until the GL area has drawn a few frames (at most ten seconds).
    context = GLib.MainContext.default()
    deadline = time.monotonic() + 10.0
    while renderer._frames < 5 and time.monotonic() < deadline:
        renderer.queue_render()
        while context.iteration(False):
            pass
        time.sleep(0.02)

    if renderer._frames < 5 or renderer._pixels is None:
        logging.error("GL did not start: only %d frame(s) rendered", renderer._frames)
        return 1

    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(bytes(renderer._pixels)),
        GdkPixbuf.Colorspace.RGB,
        True,
        8,
        renderer._width,
        renderer._height,
        renderer._width * 4,
    )
    if not pixbuf.savev(out_path, "png", [], []):
        logging.error("could not write %s", out_path)
        return 1

    logging.info("wrote %s (%dx%d)", out_path, renderer._width, renderer._height)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
