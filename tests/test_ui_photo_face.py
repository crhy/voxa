"""Tests for the photo face renderer (append-style, fakes only)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("gi")  # the renderer needs GTK; machines without it skip this file

from voxa.ui.face_pack import FacePack, MouthFrame  # noqa: E402
from voxa.ui.photo_face import (  # noqa: E402
    LIVE_TEXTURE_CACHE,
    FrameCache,
    PhotoFaceRenderer,
    compose,
    fallback_mode,
)


def _fake_pack() -> FacePack:
    mouth = (
        MouthFrame(file="m000.jpg", open=0.0, width=0.5),
        MouthFrame(file="m060.jpg", open=0.5, width=0.6),
        MouthFrame(file="m119.jpg", open=1.0, width=0.6),
    )
    return FacePack(
        id="fake",
        directory=__import__("pathlib").Path("/tmp/fake"),
        size=512,
        eye_box=(120, 90, 280, 70),
        mouth=mouth,
        blink=("b0.jpg", "b1.jpg", "b2.jpg"),
        rest=(0.0, 0.5),
    )


def test_frame_cache_loads_once():
    calls = []

    def loader(path):
        calls.append(str(path))
        return object()

    cache = FrameCache(loader, capacity=3)
    first = cache.get("/a")
    again = cache.get("/a")
    assert again is first
    assert calls == ["/a"]


def test_frame_cache_evicts_lru():
    cache = FrameCache(lambda path: path, capacity=2)
    cache.get("/a")
    cache.get("/b")
    cache.get("/a")
    cache.get("/c")
    assert cache.has("/b") is False
    assert cache.has("/a") is True
    assert cache.has("/c") is True


def test_frame_cache_clear_releases_everything():
    cache = FrameCache(lambda path: path, capacity=2)
    cache.get("/a")
    cache.get("/b")
    cache.clear()
    assert cache.has("/a") is False
    assert cache.has("/b") is False


def test_compose_still_mode():
    frame, index = compose(_fake_pack(), "still", {}, 0.0, (0.0, 0.0, 0.0), 7, "/portrait.jpg")
    assert frame.base == "/portrait.jpg"
    assert frame.eye_patch is None
    assert frame.offset == (0.0, 0.0)
    assert frame.scale == 1.0
    assert frame.rotation == 0.0
    assert index == 7


def test_compose_pack_none():
    frame, index = compose(None, "prerendered", {"viseme_aa": 1.0}, 1.0, (0.0, 0.0, 0.0), 3, "/portrait.jpg")
    assert frame.base == "/portrait.jpg"
    assert frame.eye_patch is None
    assert index == 3


def test_live_mode_uses_animated_pack_only_while_speaking_without_frames():
    assert fallback_mode("live", speaking=True) == "prerendered"
    assert fallback_mode("live", speaking=False) == "live"
    assert fallback_mode("still", speaking=True) == "still"


def test_live_fallback_decodes_selected_mouth_frame_on_demand(monkeypatch):
    """High must animate before neural frames arrive without preloading the whole pack."""
    loaded = []
    renderer = PhotoFaceRenderer.__new__(PhotoFaceRenderer)
    renderer._cache = FrameCache(lambda path: loaded.append(str(path)) or str(path))
    renderer._pending = deque()
    renderer._logged = False
    renderer._base_path = "/portrait.jpg"
    renderer._crossfade_t0 = 0.0
    renderer.widget = SimpleNamespace(
        base="portrait-texture",
        previous=None,
        fade=1.0,
        patch=None,
        eye_box=(0, 0, 0, 0),
        frame_size=0,
        offset=(0.0, 0.0),
        zoom=1.0,
        rotation=0.0,
        queue_draw=lambda: None,
    )
    monkeypatch.setattr("voxa.ui.photo_face._now", lambda: 1.0)
    frame, _index = compose(
        _fake_pack(), "prerendered", {"viseme_aa": 1.0}, 0.0, (0.0, 0.0, 0.0), 0, "/portrait.jpg"
    )

    renderer._draw(frame)

    assert loaded == [frame.base]
    assert renderer._base_path == frame.base
    assert renderer.widget.base == frame.base


def test_compose_prerendered_open_frame():
    frame, index = compose(_fake_pack(), "prerendered", {"viseme_aa": 1.0}, 0.0, (0.0, 0.0, 0.0), 0, "/p.jpg")
    assert index == 2
    assert frame.base.endswith("m119.jpg")


def test_compose_blink_patch_above_threshold():
    frame, _index = compose(_fake_pack(), "prerendered", {}, 0.5, (0.0, 0.0, 0.0), 0, "/p.jpg")
    assert frame.eye_patch is not None
    assert frame.eye_patch.endswith("b1.jpg")


def test_compose_never_moves_the_whole_photo():
    # Sub-pixel shifts make a photo shimmer, so the head pose must not move or rotate the frame.
    frame, _index = compose(_fake_pack(), "prerendered", {}, 0.0, (1000.0, -1000.0, 1000.0), 0, "/p.jpg")
    assert frame.offset == (0.0, 0.0)
    assert frame.rotation == 0.0


def test_compose_live_falls_back_to_matching_portrait():
    frame, index = compose(
        _fake_pack(), "live", {"viseme_aa": 1.0}, 0.5, (5.0, 3.0, 2.0), 4, "/portrait.jpg"
    )
    assert index == 4
    assert frame.base == "/portrait.jpg"
    assert frame.eye_patch is None
    assert frame.offset == (0.0, 0.0)
    assert frame.rotation == 0.0


def test_renderer_mode():
    renderer = PhotoFaceRenderer.__new__(PhotoFaceRenderer)
    renderer.widget = None
    renderer._mode = "prerendered"
    renderer.set_mode("still")
    assert renderer.get_mode() == "still"
    renderer.set_mode("bogus")
    assert renderer.get_mode() == "still"


def test_renderer_public_methods_match():
    from voxa.ui.avatar_3d import Gl3DFaceRenderer

    wanted = {name for name in dir(Gl3DFaceRenderer) if name.startswith("set_")}
    have = {name for name in dir(PhotoFaceRenderer) if name.startswith("set_")}
    assert wanted <= have


def test_renderer_set_character_none_pack(monkeypatch):
    import voxa.ui.photo_face as pf

    monkeypatch.setattr(pf, "load_pack", lambda cid, directory=None, portrait=None: None)

    class FakeAvatar:
        portrait_path = "/portrait.jpg"

    monkeypatch.setattr(pf, "get_avatar", lambda cid: None)
    monkeypatch.setattr(pf, "default_avatar", lambda: FakeAvatar())

    renderer = PhotoFaceRenderer()
    renderer.set_character("aoife")
    assert renderer._pack is None
    assert renderer._portrait == "/portrait.jpg"
    assert renderer._index == 0


def test_renderer_widget_under_gtk():
    pytest.importorskip("gi")
    from gi.repository import Gtk

    if not Gtk.init_check():
        pytest.skip("no GTK")
    renderer = PhotoFaceRenderer()
    assert renderer.widget is not None
    renderer.queue_render()


def test_live_frames_are_decoded_from_plain_bytes():
    """The face server's JPEGs arrive as Python bytes; Gdk needs them wrapped, or every tick fails."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 100, 50)).save(out, format="JPEG")
    renderer = PhotoFaceRenderer()
    texture = renderer._live_texture("live:test:0", out.getvalue())
    assert texture.get_width() == 16 and texture.get_height() == 16
    assert renderer._live_texture("live:test:0", out.getvalue()) is texture  # cached by key
    for index in range(LIVE_TEXTURE_CACHE + 3):
        renderer._live_texture(f"live:test:{index + 1}", out.getvalue())
    assert len(renderer._live_textures) == LIVE_TEXTURE_CACHE
