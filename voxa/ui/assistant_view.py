"""Central assistant view: artwork plus a state caption.

This is the widget boundary where a live 3D avatar renderer will later be
swapped in: the rest of the application only calls ``set_state``,
``set_listening``, ``set_thinking``, ``set_speaking`` and
``set_audio_level`` and never inspects how the avatar is drawn.
"""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

from .state import AssistantState  # noqa: E402

BADGE_PATH = Path(__file__).resolve().parent / "assets" / "voxa-badge.png"
AVATAR_SIZE = 320

STATE_CAPTIONS = {
    AssistantState.OFFLINE: "Offline",
    AssistantState.READY: "Ready",
    AssistantState.LISTENING: "Listening…",
    AssistantState.THINKING: "Thinking…",
    AssistantState.SPEAKING: "Speaking…",
    AssistantState.WORKING: "Working…",
    AssistantState.WAITING: "Waiting for you…",
    AssistantState.ERROR: "Something went wrong",
}


class AssistantView(Gtk.Box):
    """The Voxa presence at the center of the window."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("voxa-assistant-view")
        self._audio_level = 0.0

        # A Gtk.Image with a pixel size is a hard cap. A Gtk.Picture would grow to the
        # texture's own size (1024 px) or fill all the space it is offered, which
        # leaves no negative space around the assistant.
        try:
            texture = Gdk.Texture.new_from_filename(str(BADGE_PATH))
            picture = Gtk.Image.new_from_paintable(texture)
        except Exception:
            picture = Gtk.Image()
        picture.set_pixel_size(AVATAR_SIZE)
        picture.set_halign(Gtk.Align.CENTER)
        self.append(picture)

        self.caption = Gtk.Label(label="Offline")
        self.caption.set_xalign(0.5)
        self.caption.add_css_class("voxa-state")
        self.append(self.caption)

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        """Update the caption; a non-empty detail replaces the default text."""
        self.caption.set_text(detail if detail else STATE_CAPTIONS.get(state, "Ready"))

    def set_listening(self, active: bool) -> None:
        self._set_activity("listening", active)

    def set_thinking(self, active: bool) -> None:
        self._set_activity("thinking", active)

    def set_speaking(self, active: bool) -> None:
        self._set_activity("speaking", active)

    def set_audio_level(self, level: float) -> None:
        """Store the clamped 0..1 audio level; cheap enough for 20 Hz calls."""
        self._audio_level = max(0.0, min(1.0, float(level)))

    @property
    def audio_level(self) -> float:
        return self._audio_level

    def _set_activity(self, css_class: str, active: bool) -> None:
        if active:
            self.add_css_class(css_class)
        else:
            self.remove_css_class(css_class)
