from __future__ import annotations

import sys
import types

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

import voxa.ui.avatar_3d as avatar_3d  # noqa: E402
from voxa.ui.assistant_view import AssistantView, StaticAssistantRenderer  # noqa: E402
from voxa.ui.avatar_3d import _build_mesh, _mesh_data, _uv_sphere  # noqa: E402
from voxa.ui.avatars import RENDERER_GL3D, AvatarDescriptor, default_avatar  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402


def test_uv_sphere_has_triangles() -> None:
    points, triangles = _uv_sphere(1.0, (0.0, 0.0, 0.0), rings=4, segments=6)
    assert points
    assert len(triangles) == 2 * 4 * 6
    assert all(len(triangle) == 3 for triangle in triangles)


def test_mesh_mouth_open_increases_with_audio_level() -> None:
    closed = _build_mesh(0.0)
    open_mesh = _build_mesh(1.0, speaking=True)

    closed_y = [y for _, y, _, _, _, _, kind in closed if kind == 1.0]
    open_y = [y for _, y, _, _, _, _, kind in open_mesh if kind == 1.0]
    assert max(open_y) - min(open_y) > max(closed_y) - min(closed_y)


def test_mesh_data_is_float_array() -> None:
    data = _mesh_data(_build_mesh(0.2))
    assert len(data) == len(_build_mesh(0.2)) * 7
    assert all(float(value) == value for value in data[:2])


def test_3d_opt_in_falls_back_when_renderer_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    class _UnavailableRenderer:
        def __init__(self, view):
            raise RuntimeError("no GL support")

    monkeypatch.setattr(avatar_3d, "Gl3DFaceRenderer", _UnavailableRenderer)
    view = AssistantView()
    assert isinstance(view.renderer, StaticAssistantRenderer)


def test_3d_opt_in_falls_back_when_gl_initialization_fails(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    def _fail_initialize(self):
        raise RuntimeError("forced GLArea failure")

    monkeypatch.setattr(avatar_3d.Gl3DFaceRenderer, "_initialize_gl", _fail_initialize)
    view = AssistantView()
    view._enable_3d_renderer()
    assert isinstance(view.renderer, StaticAssistantRenderer)
    assert view._3d_renderer is None


def test_3d_opt_in_enables_glarea_when_view_realizes(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    view = AssistantView()
    window = Gtk.Window()
    window.set_child(view)

    try:
        window.show()
        main_context = GLib.MainContext.default()
        for _ in range(100):
            while main_context.iteration(False):
                pass

        if not isinstance(view.renderer, avatar_3d.Gl3DFaceRenderer):
            pytest.skip("GLArea rendering did not initialize in this environment")

        assert view.renderer._gl_ok
        assert view.renderer._render_error is None
        assert view.renderer.widget.get_parent() is view
        assert not view._static_renderer.widget.get_parent()
    finally:
        window.destroy()


def test_3d_renderer_public_state_setters_work_when_gl_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    class _UnavailableRenderer:
        def __init__(self, view):
            raise RuntimeError("no GL support")

    monkeypatch.setattr(avatar_3d, "Gl3DFaceRenderer", _UnavailableRenderer)
    view = AssistantView()
    view.set_state(AssistantState.READY)
    view.set_audio_level(0.7)
    assert view.caption.get_text() == "Ready"
    assert view.audio_level == 0.7


def test_avatar_model_path_expands_tilde(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOXA_AVATAR_MODEL", "avatars/triangle.glb")
    assert avatar_3d.avatar_model_path() == str(tmp_path / "avatars/triangle.glb")


def test_3d_renderer_falls_back_when_avatar_model_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    monkeypatch.setenv("VOXA_AVATAR_MODEL", str(tmp_path / "missing.glb"))

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    assert renderer._model_vertices is None


def test_3d_renderer_falls_back_when_avatar_model_is_invalid(monkeypatch, tmp_path) -> None:
    path = tmp_path / "bad.glb"
    path.write_bytes(b"not a glb")

    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    monkeypatch.setenv("VOXA_AVATAR_MODEL", str(path))

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    assert renderer._model_vertices is None


def test_3d_renderer_uses_model_when_load_mesh_succeeds(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    monkeypatch.setenv("VOXA_AVATAR_MODEL", str(tmp_path / "triangle.glb"))

    vertices = [
        (-0.85, -0.85, 0.0, 0.98, 0.98, 0.98, 0.0),
        (0.85, -0.85, 0.0, 0.98, 0.98, 0.98, 0.0),
        (-0.85, 0.85, 0.0, 0.98, 0.98, 0.98, 0.0),
    ]
    monkeypatch.setattr(avatar_3d, "load_mesh", lambda path, color=(1.0, 1.0, 1.0): vertices)

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    assert renderer._model_vertices == vertices


def test_mesh_uses_character_cosmetic_colors() -> None:
    avatar = AvatarDescriptor(
        id="custom",
        display_name="Custom",
        model_path="custom.glb",
        thumbnail_path="custom.png",
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
        skin_tone=(1.0, 0.0, 0.0),
        hair_color=(0.0, 1.0, 0.0),
    )

    mesh = _build_mesh(0.0, avatar=avatar)

    face_colors = {tuple(round(channel, 6) for channel in color) for _x, _y, _z, *color, kind in mesh if kind == 0.0}
    mouth_colors = {tuple(round(channel, 6) for channel in color) for _x, _y, _z, *color, kind in mesh if kind == 1.0}
    hair_colors = {tuple(round(channel, 6) for channel in color) for _x, _y, _z, *color, kind in mesh if kind == 3.0}

    assert (1.0, 0.0, 0.0) in face_colors
    assert mouth_colors == {(0.45, 0.0, 0.0)}
    assert hair_colors == {(0.0, 1.0, 0.0)}


def test_mesh_omits_hair_when_character_has_no_hair_color() -> None:
    avatar = AvatarDescriptor(
        id="no-hair",
        display_name="No Hair",
        model_path="no-hair.glb",
        thumbnail_path="no-hair.png",
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
        skin_tone=(0.2, 0.4, 0.6),
        hair_color=None,
    )

    mesh = _build_mesh(0.0, avatar=avatar)
    assert not any(kind == 3.0 for *_vertex, kind in mesh)


def test_3d_renderer_set_character_uses_downloaded_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    monkeypatch.delenv("VOXA_AVATAR_MODEL", raising=False)

    model_path = tmp_path / "grace.glb"
    model_path.write_bytes(b"glb")

    avatar = AvatarDescriptor(
        id="grace",
        display_name="Grace",
        model_path=str(model_path),
        thumbnail_path="grace.png",
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
    )

    vertices = [(-0.1, -0.1, 0.0, 0.5, 0.7, 0.8, 0.0)]
    monkeypatch.setattr(avatar_3d, "get_avatar", lambda _character_id: avatar)
    monkeypatch.setattr(avatar_3d, "model_is_downloaded", lambda _avatar: True)
    monkeypatch.setattr(avatar_3d, "load_mesh", lambda path, color=(1.0, 1.0, 1.0): vertices)

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer.set_character("grace")
    assert renderer._avatar == avatar
    assert renderer._model_vertices == vertices


def test_3d_renderer_explicit_model_overrides_character_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    explicit_path = tmp_path / "override.glb"
    monkeypatch.setenv("VOXA_AVATAR_MODEL", str(explicit_path))

    avatar = AvatarDescriptor(
        id="grace",
        display_name="Grace",
        model_path=str(tmp_path / "grace.glb"),
        thumbnail_path="grace.png",
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
    )

    monkeypatch.setattr(avatar_3d, "get_avatar", lambda _character_id: avatar)
    monkeypatch.setattr(avatar_3d, "model_is_downloaded", lambda _avatar: True)
    monkeypatch.setattr(avatar_3d, "load_mesh", lambda path, color=(1.0, 1.0, 1.0): [path])

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer.set_character("grace")
    assert renderer._model_vertices == [str(explicit_path)]


def test_3d_renderer_set_character_empty_keeps_default_avatar() -> None:
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer.set_character("")
    assert renderer._avatar == default_avatar()


def test_3d_renderer_initial_mesh_uses_current_avatar_colors(monkeypatch) -> None:
    avatar = AvatarDescriptor(
        id="custom",
        display_name="Custom",
        model_path="custom.glb",
        thumbnail_path="custom.png",
        voice="en-US",
        renderer=RENDERER_GL3D,
        skin_tone=(0.2, 0.3, 0.4),
        hair_color=(0.6, 0.7, 0.8),
    )
    fake_gl = types.ModuleType("OpenGL.GL")
    fake_gl.GL_COLOR_BUFFER_BIT = 1
    fake_gl.GL_DEPTH_BUFFER_BIT = 2
    fake_gl.GL_DEPTH_TEST = 4
    fake_gl.GL_FLOAT = 1
    fake_gl.glClear = lambda _flags: None
    fake_gl.glClearColor = lambda *args: None
    fake_gl.glEnable = lambda _mode: None
    fake_gl.glGenBuffers = lambda _count: [1]
    fake_gl.glGenVertexArrays = lambda _count: [1]
    fake_gl.glEnableVertexAttribArray = lambda _index: None
    fake_gl.glVertexAttribPointer = lambda *args: None

    uploaded_meshes = []
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._avatar = avatar
    renderer._model_vertices = None
    monkeypatch.setattr(avatar_3d, "_ensure_pyopengl_context", lambda: None)
    monkeypatch.setitem(sys.modules, "OpenGL", types.ModuleType("OpenGL"))
    monkeypatch.setitem(sys.modules, "OpenGL.GL", fake_gl)
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _vertex, _fragment: object())
    monkeypatch.setattr(renderer, "_upload_mesh", lambda _gl, mesh: uploaded_meshes.append(mesh))

    renderer._setup_gl(None)

    assert uploaded_meshes[0][0][3:6] == avatar.skin_tone
