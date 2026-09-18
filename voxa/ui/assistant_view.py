"""Center-stage assistant view (issue #5).

The avatar is the absolute visual center of the window with generous
negative space and a tiny state caption beneath it. The widget boundary is
deliberate: the ``Adw.Avatar`` placeholder here is later replaced by a live
3D renderer without touching callers, which only use the hooks below plus
``set_audio_level()`` (future lip movement).

All hooks must be called on the GTK thread (via ``GLib.idle_add`` from
backend threads, matching the existing convention in ``window.py``).
"""

from __future__ import annotations

from gi.repository import Adw, Gtk

from voxa.ui.state import STATUS_TEXT, AssistantState


class AssistantView(Gtk.Box):
    """Centered avatar, state caption, and listening level."""

    def __init__(self) -> None:
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            hexpand=True,
            vexpand=True,
        )
        self._state = AssistantState.READY
        self._audio_level = 0.0
        self._speaking = False

        self.avatar = Adw.Avatar(text="Voxa", size=224, show_initials=True)
        self.avatar.set_halign(Gtk.Align.CENTER)
        self.caption = Gtk.Label(label=STATUS_TEXT[AssistantState.READY])
        self.caption.add_css_class("dim-label")
        self.caption.set_halign(Gtk.Align.CENTER)
        self.level = Gtk.LevelBar(min_value=0.0, max_value=1.0, value=0.0)
        self.level.set_size_request(220, -1)
        self.level.set_halign(Gtk.Align.CENTER)
        self.level.set_visible(False)

        self.append(self.avatar)
        self.append(self.caption)
        self.append(self.level)

    @property
    def state(self) -> AssistantState:
        return self._state

    def set_state(self, state: AssistantState, detail: str | None = None) -> None:
        """Single entry point: caption, level visibility, avatar activity."""
        self._state = state
        self.caption.set_text(detail or STATUS_TEXT[state])
        listening = state == AssistantState.LISTENING
        self.level.set_visible(listening or self._speaking)
        if not listening:
            self.level.set_value(0.0)

    # Convenience hooks; each maps onto set_state.
    def set_listening(self, active: bool) -> None:
        self.set_state(AssistantState.LISTENING if active else AssistantState.READY)

    def set_thinking(self, active: bool) -> None:
        if active:
            self.set_state(AssistantState.THINKING)
        elif self._state == AssistantState.THINKING:
            self.set_state(AssistantState.READY)

    def set_speaking(self, active: bool) -> None:
        self._speaking = active
        if active:
            self.set_state(AssistantState.SPEAKING)
        elif self._state == AssistantState.SPEAKING:
            self.set_state(AssistantState.READY)
        self.level.set_visible(active or self._state == AssistantState.LISTENING)

    def set_audio_level(self, level: float) -> None:
        """Microphone/TTS level 0.0..1.0; later drives avatar lips."""
        self._audio_level = min(1.0, max(0.0, level))
        if self.level.get_visible():
            self.level.set_value(self._audio_level)
