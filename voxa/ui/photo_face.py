"""Play a photoreal face pack (mouth frames + blink patches) in a GTK widget.

A face pack is a folder of JPEG mouth frames plus three eye-blink crops
(:mod:`voxa.ui.face_pack`).  Every frame of one pack shows the same portrait
with the head perfectly still, so a mouth frame can be swapped for another and
an eye patch pasted at ``eye_box`` with no alignment work.  This renderer draws
those frames as textures in one custom widget's snapshot (GPU-composited), with a
short cross-fade between mouth frames and a gentle head sway.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gsk", "4.0")
gi.require_version("Graphene", "1.0")
from gi.repository import Gdk, GLib, Graphene, Gsk, Gtk  # noqa: E402

from .avatars import default_avatar, get_avatar  # noqa: E402
from .face_motion import FaceMotion  # noqa: E402
from .face_pack import blink_patch, load_pack, mouth_target, pick_frame  # noqa: E402
from .phonemes import timeline, weights_at  # noqa: E402
from .state import AssistantState  # noqa: E402
from .visemes import blend, viseme_weights  # noqa: E402

MODES = ("live", "prerendered", "still")

MAX_OFFSET = 0.012
MAX_ROTATION = 1.2
CROSSFADE_MS = 60.0
CLIP_RADIUS = 18
MOTION_PERIOD = 300.0  # seconds after which the idle motion repeats; keeps FaceMotion cheap
OVERSCAN = 1.05  # the frame is drawn slightly large so the sway never shows its edges


def _now() -> float:
    return GLib.get_monotonic_time() / 1000000.0


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


@dataclass(frozen=True, slots=True)
class FaceFrame:
    base: str
    eye_patch: str | None
    eye_box: tuple[int, int, int, int]
    size: int
    offset: tuple[float, float]
    scale: float
    rotation: float


class FrameCache:
    """Decoded frames keyed by path, least-recently-used, pure Python apart from the loader."""

    def __init__(self, loader, capacity: int = 160):
        self._loader = loader
        self._capacity = max(1, int(capacity))
        self._items: OrderedDict[str, object] = OrderedDict()

    def get(self, path):
        key = str(path)
        items = self._items
        if key in items:
            items.move_to_end(key)
            return items[key]
        value = self._loader(path)
        items[key] = value
        while len(items) > self._capacity:
            items.popitem(last=False)
        return value

    def has(self, path) -> bool:
        return str(path) in self._items


def compose(
    pack,
    mode: str,
    visemes: dict[str, float],
    blink: float,
    head: tuple[float, float, float],
    previous_index: int,
    portrait: str,
) -> tuple[FaceFrame, int]:
    """Decide what to draw this tick: a :class:`FaceFrame` plus the chosen mouth index."""
    if mode == "still" or pack is None:
        return (
            FaceFrame(
                base=str(portrait),
                eye_patch=None,
                eye_box=(0, 0, 0, 0),
                size=0,
                offset=(0.0, 0.0),
                scale=1.0,
                rotation=0.0,
            ),
            previous_index,
        )
    target = mouth_target(visemes, pack.rest)
    index = pick_frame(pack, target, previous_index)
    patch = blink_patch(pack, blink)
    yaw, pitch, roll = head
    offset = (
        _clamp(yaw * 0.002, -MAX_OFFSET, MAX_OFFSET),
        _clamp(pitch * 0.002, -MAX_OFFSET, MAX_OFFSET),
    )
    rotation = _clamp(roll * 0.5, -MAX_ROTATION, MAX_ROTATION)
    return (
        FaceFrame(
            base=str(pack.directory / pack.mouth[index].file),
            eye_patch=(str(pack.directory / patch) if patch else None),
            eye_box=pack.eye_box,
            size=int(pack.size),
            offset=offset,
            scale=1.0,
            rotation=rotation,
        ),
        index,
    )


class FaceCanvas(Gtk.Widget):
    """One square face: a base frame, the frame it is fading from, and an eye patch."""

    __gtype_name__ = "VoxaFaceCanvas"

    def __init__(self) -> None:
        super().__init__()
        self.base = None
        self.previous = None
        self.fade = 1.0
        self.patch = None
        self.eye_box = (0, 0, 0, 0)
        self.frame_size = 0
        self.offset = (0.0, 0.0)
        self.zoom = 1.0
        self.rotation = 0.0
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_overflow(Gtk.Overflow.HIDDEN)

    def do_measure(self, orientation, for_size):
        return (120, 320, -1, -1)

    def do_snapshot(self, snapshot) -> None:
        width, height = self.get_width(), self.get_height()
        if self.base is None or width <= 0 or height <= 0:
            return
        side = float(min(width, height))
        left, top = (width - side) / 2.0, (height - side) / 2.0
        clip = Gsk.RoundedRect()
        clip.init_from_rect(Graphene.Rect().init(left, top, side, side), CLIP_RADIUS)
        snapshot.push_rounded_clip(clip)
        snapshot.save()
        centre = Graphene.Point().init(
            left + side / 2.0 + self.offset[0] * side, top + side / 2.0 + self.offset[1] * side
        )
        snapshot.translate(centre)
        snapshot.rotate(self.rotation)
        snapshot.scale(self.zoom * OVERSCAN, self.zoom * OVERSCAN)
        snapshot.translate(Graphene.Point().init(-side / 2.0, -side / 2.0))
        full = Graphene.Rect().init(0, 0, side, side)
        if self.previous is not None and self.fade < 1.0:
            snapshot.append_texture(self.previous, full)
            snapshot.push_opacity(self.fade)
            snapshot.append_texture(self.base, full)
            snapshot.pop()
        else:
            snapshot.append_texture(self.base, full)
        if self.patch is not None and self.frame_size:
            factor = side / self.frame_size
            x, y, w, h = self.eye_box
            snapshot.append_texture(
                self.patch, Graphene.Rect().init(x * factor, y * factor, w * factor, h * factor)
            )
        snapshot.restore()
        snapshot.pop()


class PhotoFaceRenderer:
    """Draws a face pack (or a still portrait) into a GTK widget."""

    def __init__(self, view=None):
        self._view = view
        self._mode = "prerendered"
        self._pack = None
        self._portrait = ""
        self._index = 0
        self._viseme: dict[str, float] = {}
        self._state = AssistantState.READY
        self._listening = False
        self._thinking = False
        self._speaking = False
        self._speaking_since: float | None = None
        self._raw_words: list[tuple[str, float, float]] = []
        self._word_timeline: list[tuple[float, float, str]] = []
        self._speech_clock = None
        self._audio_level = 0.0
        self._intensity = 0.0
        self._gaze = (0.0, 0.0)
        self._emotion = ""
        self._forced_viseme: dict[str, float] | None = None
        self._forced_blink: float | None = None
        self._face_motion = FaceMotion()
        self._cache = FrameCache(self._load_texture)
        self._tick_added = False
        self._last_redraw = 0.0
        self._crossfade_from: str | None = None
        self._crossfade_t0 = 0.0
        self._logged = False

        self.widget = FaceCanvas()
        self.widget.add_css_class("voxa-photo-face")
        self._base_path: str | None = None
        self._pending: deque = deque()
        self._motion_t0 = _now()
        self._ensure_render_tick()

    def _load_texture(self, path):
        return Gdk.Texture.new_from_filename(str(path))

    def _ensure_cached(self, path) -> bool:
        """True when the frame is decoded; otherwise put it at the front of the load queue."""
        if self._cache.has(path):
            return True
        if not self._pending or self._pending[0] != path:
            self._pending.appendleft(path)
        return False

    def _load_some(self, budget: float = 0.006) -> None:
        """Decode queued frames for a few milliseconds.

        Done from the tick itself, not from an idle callback: while a tick callback is installed the frame
        clock can keep the main loop busy enough that idle callbacks never get a turn.
        """
        deadline = _now() + budget
        while self._pending:
            path = self._pending.popleft()
            if not self._cache.has(path):
                try:
                    self._cache.get(path)
                except Exception as exc:  # a broken frame must not stop the others
                    if not self._logged:
                        self._logged = True
                        logging.warning("photo face frame %s failed: %s", path, exc)
            if _now() >= deadline:
                break

    def set_character(self, character_id: str | None) -> None:
        """Load the pack (may be None -> still portrait) and the portrait; reset the index."""
        self._pack = load_pack(character_id) if character_id else None
        avatar = get_avatar(character_id) if character_id else None
        if avatar is None:
            avatar = default_avatar()
        self._portrait = str(avatar.portrait_path)
        self._index = 0
        self._pending.clear()
        self._base_path = None
        if self._pack is not None:
            self._pending.extend(str(self._pack.directory / frame.file) for frame in self._pack.mouth)
            self._pending.extend(str(self._pack.directory / name) for name in self._pack.blink)

    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self._mode = mode

    def get_mode(self) -> str:
        return self._mode

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        self._state = state

    def set_listening(self, active: bool) -> None:
        self._listening = bool(active)

    def set_thinking(self, active: bool) -> None:
        self._thinking = bool(active)

    def set_speaking(self, active: bool) -> None:
        self._speaking = bool(active)
        if active:
            self._speaking_since = _now()
            self._ensure_render_tick()
        else:
            self._speaking_since = None
            self._raw_words = []
            self._word_timeline = []

    def set_word_timeline(self, words: list[tuple[str, float, float]]) -> None:
        self._raw_words.extend(words)
        self._word_timeline = timeline(self._raw_words)

    def set_speech_clock(self, clock) -> None:
        self._speech_clock = clock

    def set_audio_level(self, level: float) -> None:
        self._audio_level = max(0.0, min(1.0, float(level)))

    def set_emotion(self, name: str) -> None:
        self._emotion = name

    def set_viseme(self, name: str) -> None:
        self._forced_viseme = {name: 1.0}

    def set_gaze_target(self, x: float, y: float) -> None:
        self._gaze = (float(x), float(y))

    def set_activity_intensity(self, value: float) -> None:
        self._intensity = max(0.0, min(1.0, float(value)))

    def queue_render(self) -> None:
        self.widget.queue_draw()

    def _ensure_render_tick(self) -> None:
        if self._tick_added:
            return
        self._tick_added = True
        # A plain timer, not a frame-clock tick callback: a tick callback keeps the frame clock running flat
        # out, which starves the rest of the window's idle work.
        self._was_parented = False
        self.widget.add_tick_callback(self._on_tick)

    def _speech_visemes(self) -> dict[str, float]:
        if self._word_timeline and self._speech_clock is not None:
            return weights_at(self._word_timeline, self._speech_clock())
        return viseme_weights(_now() - self._speaking_since)

    def _on_tick(self, widget, *args):
        if self.widget.get_parent() is not None:
            self._was_parented = True
        elif self._was_parented:
            self._tick_added = False
            return GLib.SOURCE_REMOVE  # removed from the window: this renderer is finished
        try:
            self._load_some()
            now = _now()
            speaking = self._speaking_since is not None
            if speaking:
                self._viseme = blend(self._viseme, self._speech_visemes(), 0.45)
            elif self._viseme:
                self._viseme = blend(self._viseme, {}, 0.45)
            if self._forced_viseme:
                self._viseme = dict(self._forced_viseme)
            # FaceMotion rebuilds its blink schedule from time zero on every call, so its cost grows with the
            # time it is given. Feeding it the monotonic clock (seconds since boot) made every tick take longer
            # than a frame and froze the rest of the window. Use the renderer's own age, wrapped.
            pose = self._face_motion.pose(
                (now - self._motion_t0) % MOTION_PERIOD,
                speaking=self._speaking,
                listening=self._listening,
                energy=self._intensity,
            )
            blink = pose.morphs.get("eyeBlinkLeft", 0.0)
            if self._forced_blink is not None:
                blink = self._forced_blink
            fading = self.widget.previous is not None and self.widget.fade < 1.0
            interval = 1.0 / 30.0 if (speaking or self._viseme or blink >= 0.15 or fading) else 1.0 / 10.0
            if now - self._last_redraw < interval:
                return GLib.SOURCE_CONTINUE
            self._last_redraw = now
            frame, self._index = compose(
                self._pack, self._mode, self._viseme, blink, pose.head, self._index, self._portrait
            )
            self._draw(frame)
        except Exception as exc:  # nothing may raise out of a tick callback
            if not self._logged:
                self._logged = True
                logging.warning("photo face tick failed (%s); falling back to still portrait", exc)
            self._draw_still()
        return GLib.SOURCE_CONTINUE

    def _draw(self, frame: FaceFrame) -> None:
        """Hand the frame to the canvas; a frame that is not decoded yet keeps the current one on screen."""
        canvas = self.widget
        now = _now()
        base = str(frame.base)
        if base != self._base_path and self._ensure_cached(base):
            if canvas.base is not None:
                canvas.previous = canvas.base
                self._crossfade_t0 = now
            canvas.base = self._cache.get(base)
            self._base_path = base
        if canvas.previous is not None:
            canvas.fade = _clamp((now - self._crossfade_t0) * 1000.0 / CROSSFADE_MS, 0.0, 1.0)
            if canvas.fade >= 1.0:
                canvas.previous = None
        canvas.patch = None
        if frame.eye_patch is not None and self._ensure_cached(frame.eye_patch):
            canvas.patch = self._cache.get(frame.eye_patch)
        canvas.eye_box = frame.eye_box
        canvas.frame_size = frame.size
        canvas.offset = frame.offset
        canvas.zoom = frame.scale
        canvas.rotation = frame.rotation
        canvas.queue_draw()

    def _draw_still(self) -> None:
        if not self._portrait:
            return
        try:
            canvas = self.widget
            canvas.base = self._cache.get(self._portrait)
            self._base_path = self._portrait
            canvas.previous = None
            canvas.patch = None
            canvas.offset = (0.0, 0.0)
            canvas.zoom = 1.0
            canvas.rotation = 0.0
            canvas.queue_draw()
        except Exception:  # the portrait itself is unreadable: leave the canvas as it is
            pass
