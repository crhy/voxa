"""Central assistant view: artwork plus a state caption.

This is the widget boundary where the photo face renderer is swapped in:
the rest of the application only calls ``set_state``, ``set_listening``,
``set_thinking``, ``set_speaking`` and ``set_audio_level`` and never
inspects how the avatar is drawn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

from .avatars import get_avatar, portrait_is_available  # noqa: E402
from .state import AssistantState  # noqa: E402

BADGE_PATH = Path(__file__).resolve().parent / "assets" / "voxa-badge.png"
AVATAR_SIZE = 320

STATE_CAPTIONS = {
    AssistantState.OFFLINE: "Offline",
    AssistantState.READY: "Ready",
    AssistantState.PAUSED: "Paused",
    AssistantState.LISTENING: "Listening",
    AssistantState.THINKING: "Thinking…",
    AssistantState.SPEAKING: "Speaking…",
    AssistantState.WORKING: "Working…",
    AssistantState.WAITING: "Waiting for you…",
    AssistantState.ERROR: "Something went wrong",
}

STATE_HINTS = {
    AssistantState.OFFLINE: "Press ACTIVE to start listening",
    AssistantState.READY: "Say “Voxa”, then your request",
    AssistantState.PAUSED: "Say “Voxa” to continue",
    AssistantState.LISTENING: "Go ahead",
    AssistantState.THINKING: "",
    AssistantState.SPEAKING: "Say “Voxa” to interrupt",
    AssistantState.WORKING: "",
    AssistantState.WAITING: "Answer to continue",
    AssistantState.ERROR: "Try again",
}


class AvatarRenderer(Protocol):
    """Renderer contract for the assistant presence.

    The public view only needs these calls; renderer implementations are free
    to expose additional attributes such as ``widget``, ``caption`` and
    ``audio_level`` for internal wiring.
    """

    widget: Gtk.Box
    caption: Gtk.Label
    hint: Gtk.Label

    def set_character(self, character_id: str | None) -> None: ...

    def set_state(self, state: AssistantState, detail: str = "") -> None: ...

    def set_listening(self, active: bool) -> None: ...

    def set_thinking(self, active: bool) -> None: ...

    def set_speaking(self, active: bool) -> None: ...

    def set_audio_level(self, level: float) -> None: ...

    def set_emotion(self, name: str) -> None: ...

    def set_viseme(self, name: str) -> None: ...

    def set_gaze_target(self, x: float, y: float) -> None: ...

    def set_activity_intensity(self, value: float) -> None: ...


class AvatarRendererBase:
    """Common defaults for future avatar API calls."""

    def __init__(self) -> None:
        self._audio_level = 0.0

    def set_character(self, character_id: str | None) -> None:
        pass

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        raise NotImplementedError

    def set_listening(self, active: bool) -> None:
        raise NotImplementedError

    def set_thinking(self, active: bool) -> None:
        raise NotImplementedError

    def set_speaking(self, active: bool) -> None:
        raise NotImplementedError

    def set_audio_level(self, level: float) -> None:
        self._audio_level = max(0.0, min(1.0, float(level)))

    @property
    def audio_level(self) -> float:
        return self._audio_level

    def set_emotion(self, name: str) -> None:
        pass

    def set_viseme(self, name: str) -> None:
        pass

    def set_gaze_target(self, x: float, y: float) -> None:
        pass

    def set_activity_intensity(self, value: float) -> None:
        pass


class StaticAssistantRenderer(AvatarRendererBase):
    """Current badge-plus-caption renderer."""

    def __init__(self, view: AssistantView) -> None:
        super().__init__()
        self._view = view

        try:
            texture = Gdk.Texture.new_from_filename(str(BADGE_PATH))
            avatar = Gtk.Image.new_from_paintable(texture)
        except Exception:
            avatar = Gtk.Image()
        avatar.set_pixel_size(AVATAR_SIZE)
        avatar.set_halign(Gtk.Align.CENTER)
        avatar.add_css_class("voxa-avatar")

        self._avatar = avatar
        self.caption = Gtk.Label(label="Offline")
        self.caption.set_xalign(0.5)
        self.caption.add_css_class("voxa-state")

        self.hint = Gtk.Label(label=STATE_HINTS[AssistantState.OFFLINE])
        self.hint.set_xalign(0.5)
        self.hint.add_css_class("voxa-state-hint")

        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.widget.add_css_class("voxa-avatar-renderer")
        self.widget.append(avatar)
        self.widget.append(self.caption)
        self.widget.append(self.hint)

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        if detail:
            self.caption.set_text(detail)
            self.hint.set_text("")
            self.hint.set_visible(False)
        else:
            self.caption.set_text(STATE_CAPTIONS.get(state, ""))
            hint = STATE_HINTS.get(state, "")
            self.hint.set_text(hint)
            self.hint.set_visible(bool(hint))
        for other in AssistantState:
            css_class = other.name.lower()
            if other is state:
                self._view.add_css_class(css_class)
            else:
                self._view.remove_css_class(css_class)

    def set_listening(self, active: bool) -> None:
        self._set_activity("listening", active)

    def set_thinking(self, active: bool) -> None:
        self._set_activity("thinking", active)

    def set_speaking(self, active: bool) -> None:
        self._set_activity("speaking", active)

    def set_audio_level(self, level: float) -> None:
        super().set_audio_level(level)
        self._update_audio_activity()

    def _set_activity(self, css_class: str, active: bool) -> None:
        if active:
            self._view.add_css_class(css_class)
        else:
            self._view.remove_css_class(css_class)

    def _update_audio_activity(self) -> None:
        for css_class in ("audio-low", "audio-medium", "audio-high"):
            self._avatar.remove_css_class(css_class)
        if self._audio_level <= 0.0:
            return
        if self._audio_level < 0.35:
            self._avatar.add_css_class("audio-low")
        elif self._audio_level < 0.70:
            self._avatar.add_css_class("audio-medium")
        else:
            self._avatar.add_css_class("audio-high")


def avatar_3d_enabled() -> bool:
    """The 3D avatar is on by default; only an explicit off value disables it."""
    value = os.environ.get("VOXA_3D_AVATAR", "").strip().lower()
    return value not in ("0", "false", "no", "off")


def _portrait_available(character_id: str) -> bool:
    """True only when the character resolves to an avatar with a portrait on disk."""
    avatar = get_avatar(character_id)
    return avatar is not None and portrait_is_available(avatar)


class AssistantView(Gtk.Box):
    """The Voxa presence at the center of the window."""

    def __init__(self, character_id: str | None = None, face_mode: str = "prerendered") -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("voxa-assistant-view")

        self._character_id = "grace" if character_id is None else character_id
        self._face_mode = face_mode
        self._static_renderer = StaticAssistantRenderer(self)
        self._photo_renderer = None
        self._renderer: AvatarRenderer = self._static_renderer
        self._3d_renderer = None
        self._append_static()
        if self._character_id:
            self.set_character(self._character_id)

    def _append_static(self) -> None:
        """Show the static badge and make it the active renderer."""
        if self._photo_renderer is not None:
            widget = self._photo_renderer.widget
            if widget.get_parent() is self:
                self.remove(widget)
            self._photo_renderer = None
        if self._static_renderer.widget.get_parent() is not self:
            self.append(self._static_renderer.widget)
        self._static_renderer.widget.set_visible(True)
        self._static_renderer._avatar.set_visible(True)
        # With no photo above it, the badge and its caption sit in the middle of the space.
        self._static_renderer.widget.set_vexpand(True)
        self._static_renderer.widget.set_valign(Gtk.Align.CENTER)
        self._renderer = self._static_renderer

    def _show_photo(self, renderer) -> None:
        """Put the photo face above the caption; only the badge image is hidden, the status text stays."""
        if self._photo_renderer is not None and self._photo_renderer is not renderer:
            old = self._photo_renderer.widget
            if old.get_parent() is self:
                self.remove(old)
        widget = renderer.widget
        widget.set_size_request(AVATAR_SIZE, AVATAR_SIZE)
        widget.set_hexpand(True)
        widget.set_vexpand(True)
        if widget.get_parent() is not self:
            self.prepend(widget)
        if self._static_renderer.widget.get_parent() is not self:
            self.append(self._static_renderer.widget)
        self._static_renderer.widget.set_visible(True)
        self._static_renderer.widget.set_vexpand(False)
        self._static_renderer.widget.set_valign(Gtk.Align.END)
        self._static_renderer._avatar.set_visible(False)
        self._photo_renderer = renderer
        self._renderer = renderer

    def _enable_3d_renderer(self, *_args) -> bool:
        """Kept for callers outside this view; the 3D renderer is no longer used."""
        return False

    def _on_3d_renderer_ready(self, renderer) -> bool:
        """Kept for callers outside this view; the 3D renderer is no longer used."""
        return False

    def _fallback_3d_renderer(self) -> None:
        """Kept for callers outside this view; nothing to fall back from."""
        self._3d_renderer = None

    @property
    def renderer(self) -> AvatarRenderer:
        return self._renderer

    @property
    def caption(self) -> Gtk.Label:
        return getattr(self._renderer, "caption", self._static_renderer.caption)

    @property
    def hint(self) -> Gtk.Label:
        return getattr(self._renderer, "hint", self._static_renderer.hint)

    @property
    def audio_level(self) -> float:
        return getattr(self._renderer, "audio_level", self._static_renderer.audio_level)

    def set_face_mode(self, mode: str) -> None:
        """Remember the face mode and push it into the live photo renderer."""
        self._face_mode = mode
        if self._photo_renderer is not None:
            self._photo_renderer.set_mode(mode)

    def get_face_mode(self) -> str:
        return self._face_mode

    def set_character(self, character_id: str | None) -> None:
        """Switch the assistant avatar without rebuilding the whole view."""
        self._character_id = character_id if character_id is not None else ""

        if not self._character_id:
            self._append_static()
            return

        if _portrait_available(self._character_id):
            try:
                from .photo_face import PhotoFaceRenderer

                renderer = PhotoFaceRenderer(self)
                renderer.set_character(self._character_id)
                renderer.set_mode(self._face_mode)
                self._show_photo(renderer)
            except Exception as exc:
                logging.warning("photo face renderer unavailable; falling back: %s", exc)
                self._append_static()
        else:
            self._append_static()

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        """Update the caption; a non-empty detail replaces the default text."""
        self._renderer.set_state(state, detail)
        if self._renderer is not self._static_renderer:
            self._static_renderer.set_state(state, detail)

    def set_listening(self, active: bool) -> None:
        self._renderer.set_listening(active)
        if self._renderer is not self._static_renderer:
            self._static_renderer.set_listening(active)

    def set_thinking(self, active: bool) -> None:
        self._renderer.set_thinking(active)
        if self._renderer is not self._static_renderer:
            self._static_renderer.set_thinking(active)

    def set_speaking(self, active: bool) -> None:
        self._renderer.set_speaking(active)
        if self._renderer is not self._static_renderer:
            self._static_renderer.set_speaking(active)

    def set_audio_level(self, level: float) -> None:
        """Store the clamped 0..1 audio level; cheap enough for 20 Hz calls."""
        self._renderer.set_audio_level(level)
        if self._renderer is not self._static_renderer:
            self._static_renderer.set_audio_level(level)

    def set_emotion(self, name: str) -> None:
        self._renderer.set_emotion(name)

    def set_viseme(self, name: str) -> None:
        self._renderer.set_viseme(name)

    def set_gaze_target(self, x: float, y: float) -> None:
        self._renderer.set_gaze_target(x, y)

    def set_activity_intensity(self, value: float) -> None:
        self._renderer.set_activity_intensity(value)

    def set_word_timeline(self, words: list[tuple[str, float, float]]) -> None:
        """Forward Edge TTS word boundaries to the active renderer, if it has one."""
        if hasattr(self._renderer, "set_word_timeline"):
            self._renderer.set_word_timeline(words)

    def set_speech_clock(self, clock) -> None:
        """Forward the playback clock to the active renderer, if it has one."""
        if hasattr(self._renderer, "set_speech_clock"):
            self._renderer.set_speech_clock(clock)
