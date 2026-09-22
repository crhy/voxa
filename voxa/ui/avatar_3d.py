"""First GLArea 3D avatar renderer.

The renderer is intentionally small: it is a vertex-colored GLArea sphere with
eyes and a mouth that opens with audio level. It is opt-in only and falls back
to the static renderer when GL initialization or shader compilation fails.

A downloaded .glb avatar model can be rendered instead of the procedural face
by setting VOXA_AVATAR_MODEL to the model path (with VOXA_3D_AVATAR=1). If the
file is missing or unreadable the renderer logs a warning and keeps drawing the
procedural sphere-face, so a bad model never produces a blank avatar.

The current GTK/GLArea path uses GLSL ES 300 because the test display stack
reports GL core versions as unsupported.
"""

from __future__ import annotations

import logging
import math
import os
from array import array
from ctypes import c_void_p

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

from .assistant_view import AVATAR_SIZE  # noqa: E402
from .avatars import AvatarDescriptor, default_avatar, get_avatar, model_is_downloaded  # noqa: E402
from .gltf import GltfLoadError, load_mesh  # noqa: E402
from .state import AssistantState  # noqa: E402

FACE_COLOR = (0.52, 0.70, 0.84)
MOUTH_COLOR = (0.22, 0.40, 0.58)
EYE_COLOR = (0.88, 0.95, 1.00)

def _blend(source: tuple[float, float, float], target: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(source[channel] * (1.0 - amount) + target[channel] * amount for channel in range(3))


def _face_colors(avatar: AvatarDescriptor | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float] | None]:
    skin = avatar.skin_tone if avatar is not None and avatar.skin_tone is not None else FACE_COLOR
    mouth = _blend(skin, (0.0, 0.0, 0.0), 0.55)
    eyes = _blend(skin, (1.0, 1.0, 1.0), 0.55)
    hair = avatar.hair_color if avatar is not None else None
    return skin, mouth, eyes, hair


VERTEX_SHADER = """
#version 300 es
precision highp float;

layout(location=0) in vec3 pos;
layout(location=1) in vec3 color;
layout(location=2) in float kind;

uniform float u_time;
uniform float u_audio;

out vec3 fragColor;

void main() {
    vec3 p = pos;
    float bob = sin(u_time * 1.25 + p.y * 2.0) * 0.015;
    p.y += bob * (1.0 - kind);

    float z = max(p.z, 0.001);
    vec3 light = normalize(vec3(0.0, 0.25, 1.0));
    float shade = clamp(dot(normalize(vec3(p.x, p.y, z)), light), 0.0, 1.0);
    fragColor = color * (0.55 + 0.55 * shade + u_audio * 0.08 * kind);

    gl_Position = vec4(p, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 300 es
precision mediump float;

in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""


def _single(value):
    if isinstance(value, (list, tuple, array)):
        return int(value[0])
    return int(value)


def _is_true(value, true_value):
    try:
        return int(value[0]) == true_value
    except Exception:
        return int(value) == true_value


def _uv_sphere(radius, center, rings=6, segments=12):
    points = []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        z = radius * math.sin(theta)
        xy_radius = radius * math.cos(theta)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            points.append(
                (
                    center[0] + xy_radius * math.cos(phi),
                    center[1] + xy_radius * math.sin(phi),
                    center[2] + z,
                )
            )

    triangles = []
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = ring * segments + segment
            b = (ring + 1) * segments + segment
            c = (ring + 1) * segments + next_segment
            d = ring * segments + next_segment
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    return points, triangles


def _add_sphere(verts, radius, center, color, kind):
    points, triangles = _uv_sphere(radius, center)
    for a, b, c in triangles:
        for index in (a, b, c):
            x, y, z = points[index]
            verts.append((x, y, z, color[0], color[1], color[2], float(kind)))


def _build_mesh(
    audio_level: float,
    speaking: bool = False,
    avatar: AvatarDescriptor | None = None,
) -> list[tuple[float, float, float, float, float, float, float]]:
    level = max(0.0, min(1.0, float(audio_level)))
    mouth_open = 0.018 + (level * 0.18 if speaking else level * 0.04)
    face_color, mouth_color, eye_color, hair_color = _face_colors(avatar)
    verts = []

    _add_sphere(verts, 0.82, (0.0, 0.0, 0.0), face_color, 0.0)
    if hair_color is not None:
        _add_sphere(verts, 0.84, (0.0, 0.12, -0.18), hair_color, 3.0)
    _add_sphere(verts, 0.12, (-0.33, 0.20, 0.72), eye_color, 2.0)
    _add_sphere(verts, 0.12, (0.33, 0.20, 0.72), eye_color, 2.0)

    y_top = -0.12 + mouth_open
    y_bottom = -0.12 - mouth_open
    x_left = -0.16
    x_right = 0.16
    z = 0.96

    quad = [
        (x_left, y_top, z),
        (x_right, y_top, z),
        (x_right, y_bottom, z),
        (x_left, y_bottom, z),
    ]
    mouth_rgb = (mouth_color[0], mouth_color[1], mouth_color[2])
    for x, y, mouth_z in quad:
        verts.append((x, y, mouth_z, mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0))

    verts.extend(
        [
            quad[0] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
            quad[1] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
            quad[2] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
            quad[0] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
            quad[2] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
            quad[3] + (mouth_rgb[0], mouth_rgb[1], mouth_rgb[2], 1.0),
        ]
    )

    return verts


def _mesh_data(triangles):
    values = array("f")
    for vertex in triangles:
        values.extend(vertex)
    return values


def avatar_model_path() -> str | None:
    """Resolve VOXA_AVATAR_MODEL to an absolute-ish path, or None when unset."""
    raw = os.environ.get("VOXA_AVATAR_MODEL", "").strip()
    if not raw:
        return None

    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(os.path.expanduser("~"), path)
    return path


def _load_avatar_model(path: str, color: tuple[float, float, float] = FACE_COLOR) -> list[tuple[float, float, float, float, float, float, float]]:
    """Load a downloaded .glb mesh, raising GltfLoadError with a clear reason."""
    return load_mesh(path, color=color)


def _ensure_pyopengl_context() -> None:
    """Make PyOpenGL use the GLArea EGL context from GTK's render callback.

    GTK4 provides the GL context through the render callback, but PyOpenGL's
    default platform helper does not know about it and reports no valid context.
    """
    from OpenGL import platform

    platform.GetCurrentContext = lambda: 1
    platform.CurrentContextIsValid = lambda: True


class Gl3DFaceRenderer:
    """GLArea-based face renderer used only when VOXA_3D_AVATAR=1 succeeds."""

    def __init__(self, view) -> None:
        self._view = view
        self._gl = None
        self._gl_ok = False
        self._render_error = None
        self._program = None
        self._vbo = 0
        self._vao = 0
        self._audio_level = 0.0
        self._state = AssistantState.OFFLINE
        self._speaking = False
        self._avatar = default_avatar()

        self._model_vertices = self._load_model(self._avatar)

        self._area = Gtk.GLArea()
        self._area.set_size_request(AVATAR_SIZE, AVATAR_SIZE)
        self._area.add_css_class("voxa-avatar")
        self._area.connect("render", self._on_render)

        self.caption = Gtk.Label(label="Offline")
        self.caption.set_xalign(0.5)
        self.caption.add_css_class("voxa-state")

        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.widget.add_css_class("voxa-avatar-renderer")
        self.widget.append(self._area)
        self.widget.append(self.caption)

    def _effective_model_path(self, avatar: AvatarDescriptor | None) -> str | None:
        """Return the explicit override first, then a downloaded character model."""
        explicit_path = avatar_model_path()
        if explicit_path:
            return explicit_path

        if avatar is not None and model_is_downloaded(avatar):
            return str(avatar.model_path)

        return None

    def _load_model(self, avatar: AvatarDescriptor | None = None):
        """Load a model when available, or return None for the procedural face."""
        path = self._effective_model_path(avatar)
        if path is None:
            return None

        face_color, _mouth, _eyes, _hair = _face_colors(avatar)
        try:
            vertices = _load_avatar_model(path, color=face_color)
        except (GltfLoadError, OSError) as exc:
            logging.warning(
                "avatar model %s could not be loaded (%s); falling back to the procedural face",
                path,
                exc,
            )
            return None

        logging.info("3D avatar model %s loaded (%d triangles)", path, len(vertices) // 3)
        return vertices

    def set_character(self, character_id: str | None) -> None:
        """Switch to a character without replacing the GLArea widget."""
        avatar = default_avatar() if not character_id else get_avatar(character_id)
        if avatar is None:
            avatar = default_avatar()

        self._avatar = avatar
        self._model_vertices = self._load_model(avatar)

        if self._gl is not None and self._gl_ok:
            vertices = self._model_vertices or _build_mesh(
                self._audio_level,
                speaking=self._speaking,
                avatar=avatar,
            )
            self._upload_mesh(self._gl, vertices)

    def _initialize_gl(self) -> None:
        try:
            if hasattr(self._area, "set_gl_clear_color"):
                self._area.set_gl_clear_color(0.18, 0.22, 0.28, 1.0)

            self._area.realize()

            if self._render_error is not None:
                raise RuntimeError(f"GLArea render failed: {self._render_error}")
        except Exception as exc:
            logging.warning("3D avatar initialization failed: %s", exc)
            raise RuntimeError("3D avatar renderer is unavailable") from exc

    def _on_render(self, area, context, *args):
        try:
            if context is not None and hasattr(context, "make_current"):
                context.make_current()

            if not self._gl_ok:
                self._setup_gl(area)
                self._gl_ok = True

            self._render_gl(area)

            if self._view is not None:
                GLib.idle_add(self._view._on_3d_renderer_ready, self)
            return True
        except Exception as exc:
            self._render_error = exc
            self._gl_ok = False
            logging.warning("3D avatar render failed: %s", exc)
            if self._view is not None:
                GLib.idle_add(self._view._fallback_3d_renderer)
            return False

    def _setup_gl(self, area):
        _ensure_pyopengl_context()
        import OpenGL.GL as gl

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.18, 0.22, 0.28, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

        program = self._compile_program(gl, VERTEX_SHADER, FRAGMENT_SHADER)
        self._gl = gl
        self._program = program
        self._vbo = _single(gl.glGenBuffers(1))
        self._vao = _single(gl.glGenVertexArrays(1))

        stride = 7 * 4
        self._upload_mesh(gl, self._model_vertices or _build_mesh(0.0, avatar=self._avatar))
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, c_void_p(0))
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, c_void_p(12))
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, False, stride, c_void_p(24))

    def _upload_mesh(self, gl, vertices) -> None:
        """Upload a triangle list into the renderer's single interleaved VBO."""
        data = _mesh_data(vertices)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            len(data) * 4,
            data.tobytes(),
            gl.GL_STATIC_DRAW,
        )

    def _compile_program(self, gl, vertex_source: str, fragment_source: str):
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        gl.glShaderSource(vertex, vertex_source)
        gl.glCompileShader(vertex)
        if not _is_true(gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS), gl.GL_TRUE):
            raise RuntimeError(gl.glGetShaderInfoLog(vertex))

        gl.glShaderSource(fragment, fragment_source)
        gl.glCompileShader(fragment)
        if not _is_true(gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS), gl.GL_TRUE):
            raise RuntimeError(gl.glGetShaderInfoLog(fragment))

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        if not _is_true(gl.glGetProgramiv(program, gl.GL_LINK_STATUS), gl.GL_TRUE):
            raise RuntimeError("GL shader link failed")

        return program

    def _render_gl(self, area):
        gl = self._gl
        if gl is None:
            raise RuntimeError("GL shader program is not compiled")

        width = max(1, area.get_width())
        height = max(1, area.get_height())
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.18, 0.22, 0.28, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self._program is None:
            raise RuntimeError("missing GL program")

        if self._model_vertices is not None:
            # A downloaded model is static geometry: it was uploaded once in
            # _setup_gl, so each frame only needs the audio/time uniforms.
            triangles = self._model_vertices
        else:
            triangles = _build_mesh(self._audio_level, speaking=self._speaking, avatar=self._avatar)
            self._upload_mesh(gl, triangles)

        gl.glUseProgram(self._program)

        time = GLib.get_monotonic_time() / 1000000.0
        time_loc = gl.glGetUniformLocation(self._program, "u_time")
        audio_loc = gl.glGetUniformLocation(self._program, "u_audio")
        if time_loc >= 0:
            gl.glUniform1f(time_loc, time)
        if audio_loc >= 0:
            gl.glUniform1f(audio_loc, self._audio_level)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(triangles))

    @property
    def audio_level(self) -> float:
        return self._audio_level

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        self._state = state
        self.caption.set_text(detail if detail else "Ready")
        if state == AssistantState.READY:
            self._view.add_css_class("ready")
        else:
            self._view.remove_css_class("ready")
        if state == AssistantState.ERROR:
            self._view.add_css_class("error")
        else:
            self._view.remove_css_class("error")

    def set_listening(self, active: bool) -> None:
        if active:
            self._view.add_css_class("listening")
        else:
            self._view.remove_css_class("listening")

    def set_thinking(self, active: bool) -> None:
        if active:
            self._view.add_css_class("thinking")
        else:
            self._view.remove_css_class("thinking")

    def set_speaking(self, active: bool) -> None:
        self._speaking = bool(active)
        if active:
            self._view.add_css_class("speaking")
        else:
            self._view.remove_css_class("speaking")

    def set_audio_level(self, level: float) -> None:
        self._audio_level = max(0.0, min(1.0, float(level)))
        self._update_audio_activity()

    def set_emotion(self, name: str) -> None:
        pass

    def set_viseme(self, name: str) -> None:
        pass

    def set_gaze_target(self, x: float, y: float) -> None:
        pass

    def set_activity_intensity(self, value: float) -> None:
        pass

    def _update_audio_activity(self) -> None:
        for css_class in ("audio-low", "audio-medium", "audio-high"):
            self._area.remove_css_class(css_class)
        if self._audio_level <= 0.0:
            return
        if self._audio_level < 0.35:
            self._area.add_css_class("audio-low")
        elif self._audio_level < 0.70:
            self._area.add_css_class("audio-medium")
        else:
            self._area.add_css_class("audio-high")
