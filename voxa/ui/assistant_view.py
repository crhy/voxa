"""Central assistant view: artwork plus a state caption.

This is the widget boundary where a live 3D avatar renderer will later be
swapped in: the rest of the application only calls ``set_state``,
``set_listening``, ``set_thinking``, ``set_speaking`` and
``set_audio_level`` and never inspects how the avatar is drawn.
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


class AvatarRenderer(Protocol):
    """Renderer contract for the assistant presence.

    The public view only needs these calls; renderer implementations are free
    to expose additional attributes such as ``widget``, ``caption`` and
    ``audio_level`` for internal wiring.
    """

    widget: Gtk.Box
    caption: Gtk.Label

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

        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.widget.add_css_class("voxa-avatar-renderer")
        self.widget.append(avatar)
        self.widget.append(self.caption)

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        self.caption.set_text(detail if detail else STATE_CAPTIONS.get(state, "Ready"))
        if state == AssistantState.READY:
            self._view.add_css_class("ready")
        else:
            self._view.remove_css_class("ready")
        if state == AssistantState.ERROR:
            self._view.add_css_class("error")
        else:
            self._view.remove_css_class("error")

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


class AssistantView(Gtk.Box):
    """The Voxa presence at the center of the window."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("voxa-assistant-view")

        self._static_renderer = StaticAssistantRenderer(self)
        self._renderer: AvatarRenderer = self._static_renderer
        self._3d_renderer = None
        self._3d_enabled = False
        self._3d_pending = False
        self.append(self._static_renderer.widget)

        if os.environ.get("VOXA_3D_AVATAR") == "1":
            try:
                from .avatar_3d import Gl3DFaceRenderer

                renderer = Gl3DFaceRenderer(self)
                if renderer.widget is None:
                    raise RuntimeError("3D avatar renderer produced no widget")
                self._3d_renderer = renderer
                self.append(renderer.widget)
                self._static_renderer.widget.set_visible(False)
            except Exception as exc:
                logging.warning("3D avatar renderer unavailable; falling back: %s", exc)

        self.connect("realize", self._enable_3d_renderer)

    def _enable_3d_renderer(self, *_args) -> bool:
        """Prepare the GLArea renderer once the assistant view is realized.

        GLArea initializes its GL context asynchronously.  This method only
        attaches and prepares the candidate renderer; the GLArea render callback
        activates it when the GL context is actually valid.
        """
        if self._3d_enabled or self._3d_renderer is None:
            return False

        renderer = self._3d_renderer
        widget = renderer.widget
        if widget.get_parent() is not self:
            self.append(widget)

        try:
            renderer._initialize_gl()
        except Exception as exc:
            logging.warning("3D avatar renderer unavailable; falling back: %s", exc)
            self._fallback_3d_renderer()
            return False

        if renderer._gl_ok and renderer._render_error is None:
            return self._activate_3d_renderer(renderer)

        self._3d_pending = True
        return False

    def _activate_3d_renderer(self, renderer=None) -> bool:
        if renderer is None:
            renderer = self._3d_renderer
        if renderer is None:
            return False

        widget = renderer.widget
        if widget.get_parent() is not self:
            self.append(widget)

        if self._static_renderer.widget.get_parent() is self:
            self.remove(self._static_renderer.widget)

        self._renderer = renderer
        self._3d_enabled = True
        self._3d_pending = False
        return True

    def _on_3d_renderer_ready(self, renderer) -> bool:
        """Idle-safe callback used by the GLArea render handler."""
        self._activate_3d_renderer(renderer)
        return False

    def _fallback_3d_renderer(self) -> None:
        renderer = self._3d_renderer
        if renderer is not None:
            widget = renderer.widget
            if widget.get_parent() is self:
                self.remove(widget)

        self._3d_renderer = None
        self._renderer = self._static_renderer
        self._3d_enabled = False
        self._3d_pending = False

        if self._static_renderer.widget.get_parent() is not self:
            self.append(self._static_renderer.widget)
        self._static_renderer.widget.set_visible(True)

    @property
    def renderer(self) -> AvatarRenderer:
        return self._renderer

    @property
    def caption(self) -> Gtk.Label:
        return self._renderer.caption

    @property
    def audio_level(self) -> float:
        return self._renderer.audio_level

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        """Update the caption; a non-empty detail replaces the default text."""
        self._renderer.set_state(state, detail)

    def set_listening(self, active: bool) -> None:
        self._renderer.set_listening(active)

    def set_thinking(self, active: bool) -> None:
        self._renderer.set_thinking(active)

    def set_speaking(self, active: bool) -> None:
        self._renderer.set_speaking(active)

    def set_audio_level(self, level: float) -> None:
        """Store the clamped 0..1 audio level; cheap enough for 20 Hz calls."""
        self._renderer.set_audio_level(level)

    def set_emotion(self, name: str) -> None:
        self._renderer.set_emotion(name)

    def set_viseme(self, name: str) -> None:
        self._renderer.set_viseme(name)

    def set_gaze_target(self, x: float, y: float) -> None:
        self._renderer.set_gaze_target(x, y)

    def set_activity_intensity(self, value: float) -> None:
        self._renderer.set_activity_intensity(value)
