from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

import voxa.ui.avatar_3d as avatar_3d  # noqa: E402
from voxa.ui.assistant_view import AssistantView, StaticAssistantRenderer  # noqa: E402
from voxa.ui.avatar_3d import _build_mesh, _mesh_data, _uv_sphere  # noqa: E402
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
