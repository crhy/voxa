"""Tests for the photo face renderer (append-style, fakes only)."""

from __future__ import annotations

import pytest

pytest.importorskip("gi")  # the renderer needs GTK; machines without it skip this file

from voxa.ui.face_pack import FacePack, MouthFrame  # noqa: E402
from voxa.ui.photo_face import FrameCache, PhotoFaceRenderer, compose  # noqa: E402


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


def test_compose_live_equals_prerendered():
    pack = _fake_pack()
    live, li = compose(pack, "live", {"viseme_aa": 1.0}, 0.5, (5.0, 3.0, 2.0), 0, "/p.jpg")
    pre, pi = compose(pack, "prerendered", {"viseme_aa": 1.0}, 0.5, (5.0, 3.0, 2.0), 0, "/p.jpg")
    assert li == pi
    assert live.base == pre.base
    assert live.eye_patch == pre.eye_patch
    assert live.offset == pre.offset
    assert live.rotation == pre.rotation


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

    monkeypatch.setattr(pf, "load_pack", lambda cid, directory=None: None)

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
