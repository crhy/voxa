"""The talking head as a small window that floats over other programs while Voxa is not focused."""
from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")

from gi.repository import Gdk, Gtk  # noqa: E402

from . import x11hints  # noqa: E402
from .assistant_view import STATE_CAPTIONS  # noqa: E402
from .photo_face import PhotoFaceRenderer  # noqa: E402
from .state import AssistantState  # noqa: E402

FACE_SIZE = 168
PADDING = 8


def overlay_supported() -> bool:
    """True only on an X11 display (the floating window needs Xlib); False on Wayland or on any error."""
    try:
        gi.require_version("GdkX11", "4.0")
        from gi.repository import GdkX11
        return isinstance(Gdk.Display.get_default(), GdkX11.X11Display)
    except Exception:
        return False


class FocusWindow(Gtk.Window):
    def __init__(self, on_activate=None) -> None:
        super().__init__()
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_focusable(False)
        self.set_can_focus(False)
        self.set_title("Voxa")
        self.remove_css_class("background")
        self.add_css_class("voxa-focus-window")

        self.renderer = PhotoFaceRenderer(None)
        self.renderer.set_mode("prerendered")

        canvas = self.renderer.widget
        canvas.natural_size = FACE_SIZE
        canvas.clip_radius = FACE_SIZE / 2
        canvas.set_size_request(FACE_SIZE, FACE_SIZE)
        canvas.set_hexpand(False)
        canvas.set_vexpand(False)
        canvas.set_halign(Gtk.Align.CENTER)
        canvas.add_css_class("voxa-focus-face")

        caption = Gtk.Label()
        caption.set_xalign(0.5)
        caption.set_wrap(True)
        caption.set_justify(Gtk.Justification.CENTER)
        caption.set_max_width_chars(20)
        caption.set_halign(Gtk.Align.CENTER)
        caption.add_css_class("voxa-focus-window-caption")
        self._caption = caption

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(PADDING)
        box.set_margin_bottom(PADDING)
        box.set_margin_start(PADDING)
        box.set_margin_end(PADDING)
        box.append(canvas)
        box.append(caption)
        self.set_child(box)

        click = Gtk.GestureClick()
        click.connect("released", lambda *_: self.activate_main())
        box.add_controller(click)

        self._on_activate = on_activate
        self._character = object()
        self._speaking = False
        self._xid = 0
        self._position = (0, 0)

    def set_character(self, character_id: str | None) -> None:
        if character_id == self._character:
            return
        self._character = character_id
        try:
            self.renderer.set_character(character_id or None)
        except Exception:
            pass

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        self._caption.set_text(detail or STATE_CAPTIONS.get(state, ""))
        self.renderer.set_state(state, detail)
        self.renderer.set_listening(state is AssistantState.LISTENING)
        self.renderer.set_thinking(state is AssistantState.THINKING)
        speaking = state is AssistantState.SPEAKING
        if speaking != self._speaking:
            self.renderer.set_speaking(speaking)
        self._speaking = speaking

    def set_word_timeline(self, words) -> None:
        self.renderer.set_word_timeline(words)

    def reset_word_timeline(self) -> None:
        self.renderer.reset_word_timeline()

    def set_speech_clock(self, clock) -> None:
        self.renderer.set_speech_clock(clock)

    def set_audio_level(self, level: float) -> None:
        self.renderer.set_audio_level(level)

    def show_at_corner(self, monitor: tuple[int, int, int, int] | None = None) -> None:
        if monitor is None:
            try:
                g = Gdk.Display.get_default().get_monitors().get_item(0).get_geometry()
                monitor = (g.x, g.y, g.width, g.height)
            except Exception:
                monitor = (0, 0, 1920, 1080)
        width = FACE_SIZE + 2 * PADDING
        self._position = x11hints.corner_position(monitor, (width, FACE_SIZE))
        if self._xid == 0 and overlay_supported():
            try:
                self.realize()
                from gi.repository import GdkX11
                self._xid = GdkX11.X11Surface.get_xid(self.get_surface())
                x11hints.detach_from_window_manager(self._xid, *self._position)
            except Exception:
                pass
        self.set_visible(True)
        self.keep_on_top()

    def keep_on_top(self) -> None:
        if self._xid and self.get_visible():
            x11hints.raise_at(self._xid, *self._position)

    def hide_window(self) -> None:
        self.set_visible(False)

    def activate_main(self) -> None:
        if self._on_activate is not None:
            self._on_activate()

    @property
    def caption(self) -> Gtk.Label:
        return self._caption

    @property
    def canvas(self):
        return self.renderer.widget
