from __future__ import annotations

import os
import subprocess
import sys
import types
from array import array

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

import voxa.ui.avatar_3d as avatar_3d  # noqa: E402
from voxa.ui.assistant_view import AssistantView, StaticAssistantRenderer, avatar_3d_enabled  # noqa: E402
from voxa.ui.avatar_3d import _build_mesh, _draw_order, _mesh_data, _pixbuf_rgba, _uv_sphere  # noqa: E402
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
    view = AssistantView("")
    assert isinstance(view.renderer, StaticAssistantRenderer)


def test_3d_opt_in_falls_back_when_gl_initialization_fails(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    def _fail_initialize(self):
        raise RuntimeError("forced GLArea failure")

    monkeypatch.setattr(avatar_3d.Gl3DFaceRenderer, "_initialize_gl", _fail_initialize)
    view = AssistantView("")
    view._enable_3d_renderer()
    assert isinstance(view.renderer, StaticAssistantRenderer)
    assert view._3d_renderer is None


def test_3d_opt_in_enables_glarea_when_view_realizes(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_3D_AVATAR", "1")

    view = AssistantView("")
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


def test_3d_renderer_caption_and_hint_follow_state_without_gl() -> None:
    view = AssistantView()
    renderer = avatar_3d.Gl3DFaceRenderer(view)
    renderer.set_state(AssistantState.READY)
    assert renderer.caption.get_text() == "Ready"
    assert renderer.hint.get_text() == "Say “Voxa”, then your request"
    assert renderer.hint.get_visible()
    renderer.set_state(AssistantState.THINKING)
    assert renderer.hint.get_visible() is False
    renderer.set_state(AssistantState.READY, detail="Go ahead")
    assert renderer.caption.get_text() == "Go ahead"
    assert renderer.hint.get_text() == ""
    assert renderer.hint.get_visible() is False


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


def test_avatar_3d_enabled_defaults_to_on(monkeypatch) -> None:
    monkeypatch.delenv("VOXA_3D_AVATAR", raising=False)
    assert avatar_3d_enabled() is True
    monkeypatch.setenv("VOXA_3D_AVATAR", "0")
    assert avatar_3d_enabled() is False
    monkeypatch.setenv("VOXA_3D_AVATAR", "off")
    assert avatar_3d_enabled() is False


def test_mouth_level_follows_speech_motion(monkeypatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(avatar_3d, "_now", lambda: clock["t"])

    class _FakeView:
        def add_css_class(self, _name: str) -> None:
            pass

        def remove_css_class(self, _name: str) -> None:
            pass

    renderer = avatar_3d.Gl3DFaceRenderer(_FakeView())
    renderer._area = types.SimpleNamespace(
        add_tick_callback=lambda _callback: None,
        queue_draw=lambda: None,
        add_css_class=lambda _name: None,
        remove_css_class=lambda _name: None,
    )

    renderer.set_audio_level(1.0)
    assert renderer.mouth_level() == 0.0

    renderer.set_audio_level(0.0)
    renderer.set_speaking(True)
    assert renderer._speaking_since is not None

    moved = False
    for _ in range(100):
        clock["t"] += 0.01
        if renderer.mouth_level() > 0.0:
            moved = True
            break
    assert moved

    renderer.set_speaking(False)
    assert renderer._speaking_since is None
    assert renderer.mouth_level() == 0.0


def test_vertex_shader_contains_jaw_uniforms_and_constants() -> None:
    assert "u_jaw" in avatar_3d.VERTEX_SHADER
    assert "u_is_model" in avatar_3d.VERTEX_SHADER
    for value in (
        avatar_3d.JAW_TOP,
        avatar_3d.JAW_TOP - 0.10,
        avatar_3d.JAW_BOTTOM - 0.12,
        avatar_3d.JAW_BOTTOM,
        avatar_3d.JAW_FRONT - 0.10,
        avatar_3d.JAW_FRONT + 0.05,
        avatar_3d.JAW_HALF_WIDTH,
        avatar_3d.JAW_HALF_WIDTH + 0.10,
        avatar_3d.JAW_DROP,
    ):
        assert str(value) in avatar_3d.VERTEX_SHADER


def test_jaw_weight_covers_chin_and_not_forehead_back_or_side() -> None:
    assert avatar_3d.jaw_weight(0.0, 0.5, 0.3) == 0.0
    assert avatar_3d.jaw_weight(0.0, -0.2, -0.4) == 0.0
    assert avatar_3d.jaw_weight(0.6, -0.2, 0.3) == 0.0
    assert avatar_3d.jaw_weight(0.0, -0.2, 0.3) > 0.5


def test_draw_order_puts_opaque_before_mask_and_blend() -> None:
    parts = [
        types.SimpleNamespace(alpha_mode="BLEND"),
        types.SimpleNamespace(alpha_mode="OPAQUE"),
        types.SimpleNamespace(alpha_mode="MASK"),
    ]
    order = _draw_order(parts)
    assert order == [1, 2, 0]


def test_pixbuf_rgba_returns_tightly_packed_w_h_4_bytes() -> None:
    width, height = 4, 3
    rowstride = width * 3 + 4
    buf = bytearray(height * rowstride)
    for y in range(height):
        for x in range(width):
            src = y * rowstride + x * 3
            buf[src] = x
            buf[src + 1] = y
            buf[src + 2] = 255
    pixbuf = GdkPixbuf.Pixbuf.new_from_data(buf, GdkPixbuf.Colorspace.RGB, False, 8, width, height, rowstride)
    packed = _pixbuf_rgba(pixbuf)
    assert len(packed) == width * height * 4
    assert packed[3] == 255


def test_load_scene_failure_falls_back_to_flat_mesh_path(monkeypatch, tmp_path) -> None:
    from voxa.ui.gltf import GltfLoadError

    monkeypatch.setenv("VOXA_3D_AVATAR", "1")
    monkeypatch.setenv("VOXA_AVATAR_MODEL", str(tmp_path / "scan.glb"))

    vertices = [(-0.1, -0.1, 0.0, 0.5, 0.7, 0.8, 0.0)]
    monkeypatch.setattr(avatar_3d, "load_scene", lambda _path: (_ for _ in ()).throw(GltfLoadError("too large")))
    monkeypatch.setattr(avatar_3d, "_load_avatar_model", lambda path, color=(1.0, 1.0, 1.0): vertices)

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    assert renderer._scene is None
    assert renderer._model_vertices == vertices


def test_snapshot_cli_writes_nonflat_png(tmp_path) -> None:
    model = "/home/rhy/voxa-characters/out/grace-long01.glb"
    if not os.path.exists(model):
        pytest.skip("test model not present")

    out = tmp_path / "snap.png"
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1")
    result = subprocess.run(
        [sys.executable, "-m", "voxa.ui.avatar_snapshot", model, str(out), "--size", "480"],
        env=env,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip("GL could not start in this environment")

    assert out.exists()
    pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(out))
    assert pixbuf.get_width() == 480
    assert pixbuf.get_height() == 480

    data = pixbuf.read_pixel_bytes().get_data()
    stride = pixbuf.get_rowstride()
    channels = pixbuf.get_n_channels()
    distinct = set()
    for y in range(140, 340):
        for x in range(140, 340):
            src = y * stride + x * channels
            distinct.add(tuple(data[src:src + channels]))
    assert len(distinct) > 500


def test_3d_renderer_morph_upload_and_decay(monkeypatch) -> None:
    import numpy as np

    from voxa.ui.gltf_scene import Part, Scene

    class _FakeGL:
        GL_ARRAY_BUFFER = 1
        GL_ELEMENT_ARRAY_BUFFER = 2
        GL_FLOAT = 3
        GL_UNSIGNED_INT = 4
        GL_TRIANGLES = 5
        GL_DEPTH_TEST = 6
        GL_CULL_FACE = 7
        GL_COLOR_BUFFER_BIT = 8
        GL_DEPTH_BUFFER_BIT = 9
        GL_TEXTURE_2D = 10
        GL_TEXTURE0 = 11
        GL_STATIC_DRAW = 12
        GL_DYNAMIC_DRAW = 13
        GL_FALSE = 0
        GL_REPEAT = 14
        GL_LINEAR = 15
        GL_LINEAR_MIPMAP_LINEAR = 16
        GL_RGBA = 17
        GL_UNSIGNED_BYTE = 18
        GL_TRUE = 1

        def __init__(self) -> None:
            self.subdata: list[tuple] = []

        def __getattr__(self, name: str):
            def _record(*args):
                if name in ("glGenBuffers", "glGenVertexArrays", "glGenTextures"):
                    return [1]
                if name == "glGetUniformLocation":
                    return 0
                if name == "glBufferSubData":
                    self.subdata.append(args)
                return None

            return _record

    morph_part = Part(
        name="head",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=None,
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        morph_names=("viseme_aa", "viseme_oh"),
        morph_deltas=np.array(
            [
                [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    plain_part = Part(
        name="body",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=None,
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        morph_names=(),
        morph_deltas=None,
    )

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(parts=[morph_part, plain_part])
    fake_gl = _FakeGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _vertex, _fragment: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)

    area = types.SimpleNamespace(get_width=lambda: 100, get_height=lambda: 100)

    renderer._viseme = {"viseme_aa": 1.0}
    renderer._render_textured(fake_gl, area)
    assert fake_gl.subdata, "morph part must re-upload positions"
    assert renderer._part_gl[0]["weights"] == {"viseme_aa": 1.0}
    assert renderer._part_gl[1]["weights"] == {}
    assert len(fake_gl.subdata) == 1, "part without morphs is never re-uploaded"

    renderer._speaking_since = None
    renderer._area = types.SimpleNamespace(queue_draw=lambda: None, add_tick_callback=lambda _callback: None)
    ticks = 0
    while renderer._viseme and ticks < 30:
        renderer._on_tick(None)
        ticks += 1
    assert renderer._viseme == {}
    assert ticks <= 30

    renderer._render_textured(fake_gl, area)
    base_bytes = array("f", morph_part.positions.reshape(-1)).tobytes()
    assert fake_gl.subdata[-1] == (fake_gl.GL_ARRAY_BUFFER, 0, len(base_bytes), base_bytes)
    assert renderer._part_gl[0]["weights"] == {}


def test_head_framing_zooms_a_tall_bust_to_its_head() -> None:
    ys = np.linspace(-0.75, 0.75, 40)
    xs = np.linspace(-0.2, 0.2, 5)
    grid = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    scale, offset_x, offset_y = avatar_3d.head_framing([grid])
    assert scale > 1.5
    assert abs(offset_x) < 0.05
    assert 0.4 < offset_y < 0.5


def test_head_framing_keeps_a_head_only_cube_at_scale_one() -> None:
    cube = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    scale, _offset_x, _offset_y = avatar_3d.head_framing([cube])
    assert scale == 1.0


class _UploadGL:
    GL_ARRAY_BUFFER = 1
    GL_ELEMENT_ARRAY_BUFFER = 2
    GL_FLOAT = 3
    GL_UNSIGNED_INT = 4
    GL_TRIANGLES = 5
    GL_DEPTH_TEST = 6
    GL_CULL_FACE = 7
    GL_COLOR_BUFFER_BIT = 8
    GL_DEPTH_BUFFER_BIT = 9
    GL_TEXTURE_2D = 10
    GL_TEXTURE0 = 11
    GL_STATIC_DRAW = 12
    GL_DYNAMIC_DRAW = 13
    GL_FALSE = 0
    GL_REPEAT = 14
    GL_LINEAR = 15
    GL_LINEAR_MIPMAP_LINEAR = 16
    GL_RGBA = 17
    GL_UNSIGNED_BYTE = 18
    GL_TRUE = 1

    def __init__(self) -> None:
        self.subdata: list[tuple] = []

    def __getattr__(self, name: str):
        def _record(*args):
            if name in ("glGenBuffers", "glGenVertexArrays", "glGenTextures"):
                return [1]
            if name == "glGetUniformLocation":
                return 0
            if name == "glBufferSubData":
                self.subdata.append(args)
            return None

        return _record


def test_idle_scene_tick_advances_pose(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    part = Part(
        name="head",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=None,
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        morph_names=(),
        morph_deltas=None,
    )
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(parts=[part])
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _v, _f: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)
    renderer._area = types.SimpleNamespace(queue_draw=lambda: None, add_tick_callback=lambda _c: None)

    assert renderer._face_motion is not None
    before = renderer._pose_seq
    result = renderer._on_tick(None)
    assert result == GLib.SOURCE_CONTINUE
    assert renderer._pose_seq == before + 1
    assert "eyeBlinkLeft" in renderer._face_morphs
    assert renderer._viseme == {}


def test_blink_morph_reaches_upload(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    part = Part(
        name="head",
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=None,
        indices=np.array([[0, 1, 2]], dtype=np.uint32),
        morph_names=("eyeBlinkLeft",),
        morph_deltas=np.array([[[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(parts=[part])
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _v, _f: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)

    renderer._face_morphs = {"eyeBlinkLeft": 1.0}
    renderer._pose_seq = 1
    area = types.SimpleNamespace(get_width=lambda: 100, get_height=lambda: 100)
    renderer._render_textured(fake_gl, area)
    assert fake_gl.subdata, "blink morph must re-upload positions"
    base_bytes = array("f", part.positions.reshape(-1)).tobytes()
    assert fake_gl.subdata[-1][3] != base_bytes, "uploaded positions differ from base"


def test_eye_gaze_rotates_vertices_about_centre() -> None:
    from voxa.ui.gltf_scene import Part

    eye = np.array(
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    part = Part(name="low-poly eye", positions=eye, normals=np.zeros_like(eye), uvs=None, indices=np.array([[0, 1, 2]], dtype=np.uint32))
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    groups = renderer._eye_groups_of(part)
    assert len(groups) == 2

    rotated = renderer._apply_gaze(eye, groups, (0.3, 0.2))
    assert not np.array_equal(rotated, eye)

    unchanged = renderer._apply_gaze(eye, groups, (0.0, 0.0))
    assert np.array_equal(unchanged, eye)


def test_rotation_matrix_is_identity_at_zero_and_tilts_at_ninety() -> None:
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    identity = renderer._rotation_matrix((0.0, 0.0, 0.0))
    assert np.allclose(identity, np.eye(3))
    tilted = renderer._rotation_matrix((0.0, 90.0, 0.0))
    assert not np.allclose(tilted, np.eye(3))


def test_visemes_override_mouth_units_while_speaking() -> None:
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._speaking_since = 0.0
    renderer._face_morphs = {"mouthSmileLeft": 1.0, "browInnerUp": 1.0}
    renderer._viseme = {"viseme_aa": 1.0}
    merged = renderer._merged_weights()
    assert merged["mouthSmileLeft"] == 0.5
    assert merged["browInnerUp"] == 1.0
    assert merged["viseme_aa"] == 1.0


def test_speech_visemes_prefers_word_timeline_over_rhythm(monkeypatch) -> None:
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    words = [("hello", 0.0, 0.5), ("world", 0.5, 0.5)]
    spread = avatar_3d.timeline(words)
    renderer._word_timeline = spread
    renderer._speech_clock = lambda: 0.25
    assert renderer._speech_visemes() == avatar_3d.weights_at(spread, 0.25)

    renderer._word_timeline = []
    renderer._speech_clock = None
    renderer._speaking_since = 0.0
    monkeypatch.setattr(avatar_3d, "_now", lambda: 0.3)
    assert renderer._speech_visemes() == avatar_3d.viseme_weights(0.3)


def test_part_material_assigns_eye_hair_teeth_and_skin_uniforms() -> None:
    assert avatar_3d.part_material("Low-Poly Eye", 0, False) == (0.0, 0.9, 96.0)
    assert avatar_3d.part_material("Hair", 0, False) == (0.0, 0.10, 14.0)
    assert avatar_3d.part_material("Teeth", 0, False) == (0.0, 0.4, 60.0)
    assert avatar_3d.part_material("Head", 3, True) == (1.0, 0.12, 24.0)
    assert avatar_3d.part_material("Head", 3, False) == (0.0, 0.05, 16.0)


def test_skin_part_index_picks_most_morphed_part() -> None:
    from voxa.ui.gltf_scene import Part

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    parts = [
        Part(name="low-poly eye", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
        Part(
            name="head",
            positions=positions,
            normals=normals,
            uvs=None,
            indices=indices,
            morph_names=("a", "b"),
            morph_deltas=np.zeros((2, 3, 3), dtype=np.float32),
        ),
        Part(name="hair", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=("c",), morph_deltas=np.zeros((1, 3, 3), dtype=np.float32)),
    ]
    assert avatar_3d.skin_part_index(parts) == 1


def test_build_scene_gl_compiles_background_and_stores_materials(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(
        parts=[
            Part(name="low-poly eye", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
            Part(name="head", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=("a", "b"), morph_deltas=np.zeros((2, 3, 3), dtype=np.float32)),
            Part(name="hair", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
        ]
    )
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _vertex, _fragment: object())
    renderer._build_scene_gl(fake_gl)

    assert renderer._background_program is not None
    assert renderer._background_vao is not None
    assert renderer._part_gl[0]["material"] == (0.0, 0.9, 96.0)
    assert renderer._part_gl[1]["material"] == (1.0, 0.12, 24.0)
    assert renderer._part_gl[2]["material"] == (0.0, 0.10, 14.0)


def test_render_textured_sets_material_uniforms_per_part(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(
        parts=[
            Part(name="low-poly eye", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
            Part(name="head", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=("a", "b"), morph_deltas=np.zeros((2, 3, 3), dtype=np.float32)),
            Part(name="hair", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
        ]
    )
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _vertex, _fragment: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)

    uniform_values: list[float] = []
    fake_gl.glUniform1f = lambda _loc, value: uniform_values.append(value)
    area = types.SimpleNamespace(get_width=lambda: 100, get_height=lambda: 100)
    renderer._render_textured(fake_gl, area)

    assert 0.9 in uniform_values
    assert 96.0 in uniform_values
    assert 1.0 in uniform_values
    assert 0.12 in uniform_values
    assert 24.0 in uniform_values
    assert 0.10 in uniform_values
    assert 14.0 in uniform_values


def test_delete_scene_gl_releases_background_resources() -> None:
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    renderer._background_program = 7
    renderer._background_vao = 8
    renderer._delete_scene_gl(fake_gl)
    assert renderer._background_program is None
    assert renderer._background_vao is None


def test_perspective_w_table() -> None:
    assert avatar_3d.perspective_w(0.0, 6.0) == 1.0
    assert avatar_3d.perspective_w(1.0, 6.0) == pytest.approx(5.0 / 6.0)
    assert avatar_3d.perspective_w(1.0, 6.0) < 1.0
    assert avatar_3d.perspective_w(2.0, 6.0) < avatar_3d.perspective_w(1.0, 6.0)
    assert avatar_3d.perspective_w(5.0, 6.0) == 0.2
    assert avatar_3d.perspective_w(100.0, 6.0) == 0.2
    assert avatar_3d.perspective_w(3.0, 0.0) == 1.0


def test_supersample_size_table() -> None:
    assert avatar_3d.supersample_size(100, 80) == (200, 160)
    assert avatar_3d.supersample_size(100, 80, factor=3) == (300, 240)
    assert avatar_3d.supersample_size(3000, 80) == (4096, 160)
    assert avatar_3d.supersample_size(0, 0) == (1, 1)
    assert avatar_3d.supersample_size(-5, 10) == (1, 20)


def test_textured_shaders_mention_cam_dist_and_spec_color() -> None:
    assert "u_cam_dist" in avatar_3d.TEXTURED_VERTEX_SHADER
    assert "specColor" in avatar_3d.TEXTURED_FRAGMENT_SHADER


def test_render_textured_sets_cam_dist_uniform(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(
        parts=[
            Part(name="hair", positions=positions, normals=normals, uvs=None, indices=indices, morph_names=(), morph_deltas=None),
        ]
    )
    fake_gl = _UploadGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _vertex, _fragment: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)

    uniform_values: list[float] = []
    fake_gl.glUniform1f = lambda _loc, value: uniform_values.append(value)
    area = types.SimpleNamespace(get_width=lambda: 100, get_height=lambda: 100)
    renderer._render_textured(fake_gl, area)

    assert 6.0 in uniform_values
    assert renderer._part_gl[0]["material"] == (0.0, 0.10, 14.0)

def test_alpha_passes_table() -> None:
    assert avatar_3d.alpha_passes("OPAQUE") == (0.0,)
    assert avatar_3d.alpha_passes("MASK") == (1.0, 2.0)
    assert avatar_3d.alpha_passes("BLEND") == (1.0, 2.0)


def test_textured_shader_has_alpha_pass_uniform_and_glint() -> None:
    shader = avatar_3d.TEXTURED_FRAGMENT_SHADER
    assert "u_alpha_pass" in shader
    assert "glint" in shader


def test_blend_part_draws_twice_with_soft_alpha(monkeypatch) -> None:
    from voxa.ui.gltf_scene import Part, Scene

    class _RecordGL:
        GL_ARRAY_BUFFER = 1
        GL_ELEMENT_ARRAY_BUFFER = 2
        GL_FLOAT = 3
        GL_UNSIGNED_INT = 4
        GL_TRIANGLES = 5
        GL_DEPTH_TEST = 6
        GL_CULL_FACE = 7
        GL_COLOR_BUFFER_BIT = 8
        GL_DEPTH_BUFFER_BIT = 9
        GL_TEXTURE_2D = 10
        GL_TEXTURE0 = 11
        GL_STATIC_DRAW = 12
        GL_DYNAMIC_DRAW = 13
        GL_FALSE = 0
        GL_REPEAT = 14
        GL_LINEAR = 15
        GL_LINEAR_MIPMAP_LINEAR = 16
        GL_RGBA = 17
        GL_UNSIGNED_BYTE = 18
        GL_TRUE = 1
        GL_BLEND = 100
        GL_SRC_ALPHA = 101
        GL_ONE_MINUS_SRC_ALPHA = 102
        GL_DRAW_FRAMEBUFFER_BINDING = 103
        GL_FRAMEBUFFER = 104

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def __getattr__(self, name: str):
            def _record(*args):
                self.calls.append((name, args))
                if name in ("glGenBuffers", "glGenVertexArrays", "glGenTextures"):
                    return [1]
                if name == "glGetUniformLocation":
                    return 0
                return None

            return _record

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    opaque = Part(
        name="head", positions=positions, normals=normals, uvs=None, indices=indices,
        morph_names=(), morph_deltas=None, alpha_mode="OPAQUE",
    )
    blend = Part(
        name="brow", positions=positions, normals=normals, uvs=None, indices=indices,
        morph_names=(), morph_deltas=None, alpha_mode="BLEND",
    )

    renderer = avatar_3d.Gl3DFaceRenderer(object())
    renderer._scene = Scene(parts=[opaque, blend])
    fake_gl = _RecordGL()
    renderer._gl = fake_gl
    monkeypatch.setattr(renderer, "_compile_program", lambda _gl, _v, _f: object())
    renderer._build_scene_gl(fake_gl)
    renderer._draw_indices = _draw_order(renderer._scene.parts)
    renderer._viseme = {}
    area = types.SimpleNamespace(get_width=lambda: 100, get_height=lambda: 100)

    renderer._render_textured(fake_gl, area)

    draws = [args for name, args in fake_gl.calls if name == "glDrawElements"]
    assert len(draws) == 3, "opaque once, blend twice"
    assert ("glEnable", (fake_gl.GL_BLEND,)) in fake_gl.calls
    assert ("glDepthMask", (False,)) in fake_gl.calls
    assert ("glBlendFunc", (fake_gl.GL_SRC_ALPHA, fake_gl.GL_ONE_MINUS_SRC_ALPHA)) in fake_gl.calls
    assert fake_gl.calls[-2] == ("glDisable", (fake_gl.GL_BLEND,))
    assert fake_gl.calls[-1] == ("glDepthMask", (True,))
