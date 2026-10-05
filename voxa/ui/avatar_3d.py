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
import zlib
from array import array
from ctypes import c_void_p

import gi
import numpy as np

gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402

from .assistant_view import AVATAR_SIZE, STATE_CAPTIONS, STATE_HINTS  # noqa: E402
from .avatars import AvatarDescriptor, default_avatar, get_avatar, model_is_downloaded  # noqa: E402
from .face_motion import FaceMotion  # noqa: E402
from .gltf import MODEL_RADIUS, GltfLoadError, load_mesh  # noqa: E402
from .gltf_scene import load_scene  # noqa: E402
from .phonemes import timeline, weights_at  # noqa: E402
from .speech_motion import speech_level  # noqa: E402
from .state import AssistantState  # noqa: E402
from .visemes import blend, viseme_weights  # noqa: E402


def _now() -> float:
    return GLib.get_monotonic_time() / 1000000.0

FACE_COLOR = (0.52, 0.70, 0.84)
MOUTH_COLOR = (0.22, 0.40, 0.58)
EYE_COLOR = (0.88, 0.95, 1.00)

# Jaw region of a downloaded bust, expressed as fractions of MODEL_RADIUS (0.85
# in gltf.py), the half-extent of the box a normalized model sits in: the face
# points towards +z and up is +y.  The shader and jaw_weight() share these.
JAW_TOP = 0.02 * MODEL_RADIUS
JAW_BOTTOM = -0.38 * MODEL_RADIUS
JAW_FRONT = 0.05 * MODEL_RADIUS
JAW_HALF_WIDTH = 0.20 * MODEL_RADIUS
JAW_DROP = 0.07 * MODEL_RADIUS

def _blend(source: tuple[float, float, float], target: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(source[channel] * (1.0 - amount) + target[channel] * amount for channel in range(3))


def _face_colors(avatar: AvatarDescriptor | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float] | None]:
    skin = avatar.skin_tone if avatar is not None and avatar.skin_tone is not None else FACE_COLOR
    mouth = _blend(skin, (0.0, 0.0, 0.0), 0.55)
    eyes = _blend(skin, (1.0, 1.0, 1.0), 0.55)
    hair = avatar.hair_color if avatar is not None else None
    return skin, mouth, eyes, hair


VERTEX_SHADER = f"""
#version 300 es
precision highp float;

layout(location=0) in vec3 pos;
layout(location=1) in vec3 color;
layout(location=2) in float kind;

uniform float u_time;
uniform float u_audio;
uniform float u_jaw;
uniform float u_is_model;

out vec3 fragColor;

void main() {{
    vec3 p = pos;
    float bob = sin(u_time * 1.25 + p.y * 2.0) * 0.015;
    p.y += bob * (1.0 - kind);

    if (u_is_model > 0.5) {{
        // Lower front of the head = jaw and chin. No morph targets in these
        // models, so the jaw is a soft region that slides down and slightly
        // back, which reads as a talking mouth from the front.
        float below = smoothstep({JAW_TOP}, {JAW_TOP - 0.10}, p.y);
        float above = smoothstep({JAW_BOTTOM - 0.12}, {JAW_BOTTOM}, p.y);
        float front = smoothstep({JAW_FRONT - 0.10}, {JAW_FRONT + 0.05}, p.z);
        float side  = 1.0 - smoothstep({JAW_HALF_WIDTH}, {JAW_HALF_WIDTH + 0.10}, abs(p.x));
        float w = below * above * front * side;
        p.y -= u_jaw * {JAW_DROP} * w;
        p.z -= u_jaw * {JAW_DROP} * 0.25 * w;
    }}

    float z = max(p.z, 0.001);
    vec3 light = normalize(vec3(0.0, 0.25, 1.0));
    float shade = clamp(dot(normalize(vec3(p.x, p.y, z)), light), 0.0, 1.0);
    fragColor = color * (0.55 + 0.55 * shade + u_audio * 0.08 * kind);

    gl_Position = vec4(p, 1.0);
}}
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


TEXTURED_VERTEX_SHADER = """
#version 300 es
precision highp float;

layout(location=0) in vec3 pos;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;

uniform float u_time;
uniform mat3 u_view;
uniform float u_aspect;
uniform float u_scale;
uniform vec2 u_offset;
uniform mat3 u_rot;
uniform vec3 u_pivot;
uniform float u_cam_dist;

out vec3 fragNormal;
out vec2 fragUv;

void main() {
    vec3 p = pos;
    float bob = sin(u_time * 1.25 + p.y * 2.0) * 0.015;
    p.y += bob;
    p = u_rot * (p - u_pivot) + u_pivot;
    p = u_view * p;
    p.xy = (p.xy - u_offset) * u_scale;
    float z = p.z * u_scale;
    float w = (u_cam_dist > 0.0) ? max((u_cam_dist - z) / u_cam_dist, 0.2) : 1.0;
    fragNormal = u_rot * normal;
    fragUv = uv;
    gl_Position = vec4(p.x / u_aspect, p.y, -z * 0.5 * w / max(u_scale, 1e-4), w);
}
"""

TEXTURED_FRAGMENT_SHADER = """
#version 300 es
precision highp float;

uniform sampler2D u_tex;
uniform vec4 u_base_color;
uniform float u_alpha_mode;
uniform float u_alpha_cutoff;
uniform float u_alpha_pass;
uniform float u_skin;
uniform float u_spec;
uniform float u_shine;

in vec3 fragNormal;
in vec2 fragUv;
out vec4 outColor;

void main() {
    vec4 texel = texture(u_tex, fragUv) * u_base_color;
    // Opaque parts (u_alpha_pass 0.0) never discard; the solid core pass (1.0) keeps texels
    // at or above the cutoff; the soft fringe pass (2.0) keeps only the translucent band below it.
    if (u_alpha_pass > 0.5 && u_alpha_pass < 1.5 && texel.a < u_alpha_cutoff) discard;
    if (u_alpha_pass > 1.5 && (texel.a < 0.02 || texel.a >= u_alpha_cutoff)) discard;
    vec3 n = normalize(fragNormal);
    vec3 v = vec3(0.0, 0.0, 1.0);
    // Textures are sRGB: light in linear space, convert back at the end.
    vec3 base = pow(texel.rgb, vec3(2.2));

    vec3 lKey = normalize(vec3(-0.45, 0.55, 0.75));
    vec3 lFill = normalize(vec3(0.85, 0.05, 0.4));
    vec3 lRim = normalize(vec3(0.1, 0.4, -0.9));

    // Wrap diffuse: shadows stay soft, never black.
    float dKey = max((dot(n, lKey) + 0.35) / 1.35, 0.0);
    float dFill = max((dot(n, lFill) + 0.35) / 1.35, 0.0);
    float edge = 1.0 - max(dot(n, v), 0.0);
    float dRim = pow(edge, 2.0);

    vec3 light = vec3(0.06, 0.06, 0.07)
        + vec3(1.0, 0.96, 0.90) * dKey
        + vec3(0.55, 0.62, 0.72) * 0.35 * dFill
        + vec3(0.9, 0.95, 1.0) * 0.45 * dRim;

    vec3 h = normalize(lKey + v);
    float spec = pow(max(dot(n, h), 0.0), u_shine) * u_spec;
    // Cheap subsurface cue, only for skin parts (u_skin is 1.0 there).
    vec3 sss = vec3(0.25, 0.07, 0.05) * pow(edge, 3.0) * u_skin;

    // Non-eye parts tint their specular by the surface so hair never shows a white band.
    vec3 specColor = mix(base * 2.0 + 0.04, vec3(1.0), step(0.8, u_spec));
    // Fixed eye catchlight: a small glint that survives any head angle on eye parts.
    vec3 lEye = normalize(vec3(-0.25, 0.35, 0.9));
    float glint = pow(max(dot(n, normalize(lEye + v)), 0.0), 220.0) * step(0.8, u_spec);
    vec3 result = base * light + specColor * spec + sss + vec3(glint) * 0.9;
    float alpha = u_alpha_pass > 1.5 ? texel.a : 1.0;
    outColor = vec4(pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2)), alpha);
}
"""

BACKGROUND_VERTEX_SHADER = """
#version 300 es
precision highp float;
out vec2 bgUv;
void main() {
    // Full-screen triangle from gl_VertexID: (-1,-1), (3,-1), (-1,3).
    vec2 p = vec2(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID >> 1) * 4 - 1));
    bgUv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT_SHADER = """
#version 300 es
precision mediump float;
in vec2 bgUv;
out vec4 outColor;
void main() {
    outColor = vec4(mix(vec3(0.10, 0.12, 0.16), vec3(0.20, 0.24, 0.30), bgUv.y), 1.0);
}
"""

# Fixed slight downward tilt (rotation about x) as a column-major mat3, so the
# head is seen a touch from above. Identity is acceptable; this is a small tilt.
_VIEW_TILT = 0.18
_VIEW_MATRIX = (
    1.0,
    0.0,
    0.0,
    0.0,
    math.cos(_VIEW_TILT),
    math.sin(_VIEW_TILT),
    0.0,
    -math.sin(_VIEW_TILT),
    math.cos(_VIEW_TILT),
)

_ALPHA_MODE_RANK = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}


def alpha_passes(alpha_mode: str) -> tuple[float, ...]:
    """Return the u_alpha_pass values to draw a part with: opaque one pass, soft parts two."""
    if alpha_mode == "OPAQUE":
        return (0.0,)
    return (1.0, 2.0)

SUPERSAMPLE = 2


def perspective_w(z_scaled: float, cam_dist: float) -> float:
    """Perspective divide weight matching TEXTURED_VERTEX_SHADER's w.

    cam_dist 0 means orthographic (w is always 1); nearer points (larger z)
    get a smaller w, so they divide to larger screen positions.
    """
    if cam_dist > 0.0:
        return max((cam_dist - z_scaled) / cam_dist, 0.2)
    return 1.0


def supersample_size(
    width: int,
    height: int,
    factor: int = SUPERSAMPLE,
    max_side: int = 4096,
) -> tuple[int, int]:
    """Offscreen render size: each side scaled by factor, clamped to [1, max_side]."""
    return (max(1, min(width * factor, max_side)), max(1, min(height * factor, max_side)))


def is_eye_part_name(name: str) -> bool:
    lowered = name.lower()
    return ("poly" in lowered or "eye" in lowered) and "brow" not in lowered and "lash" not in lowered


def part_material(name: str, morph_count: int, is_skin: bool) -> tuple[float, float, float]:
    """Return (u_skin, u_spec, u_shine) for a part, chosen by name and morph count.

    Eyes get a strong tight specular (visible catchlight); skin a weak broad one;
    hair and teeth in between; everything else a faint sheen.
    """
    lowered = name.lower()
    if is_eye_part_name(lowered):
        return (0.0, 0.9, 96.0)
    if "teeth" in lowered:
        return (0.0, 0.4, 60.0)
    if "hair" in lowered:
        return (0.0, 0.10, 14.0)
    if is_skin:
        return (1.0, 0.12, 24.0)
    return (0.0, 0.05, 16.0)


def skin_part_index(parts) -> int:
    """Index of the part with the most morph targets (the skinned body part), or -1."""
    best, best_count = -1, 0
    for index, part in enumerate(parts):
        count = len(part.morph_deltas) if part.morph_deltas is not None else 0
        if count > best_count:
            best, best_count = index, count
    return best


def _draw_order(parts):
    """Return part indices in draw order: OPAQUE first, then MASK, then BLEND."""
    return sorted(
        range(len(parts)),
        key=lambda index: _ALPHA_MODE_RANK.get(parts[index].alpha_mode, 1),
    )


def _pixbuf_rgba(pixbuf) -> bytes:
    """Return a tightly packed ``w*h*4`` RGBA byte string from a pixbuf.

    Handles a rowstride larger than ``w * channels`` (copying row by row) and
    expands 3-channel data to RGBA.
    """
    width = pixbuf.get_width()
    height = pixbuf.get_height()
    channels = pixbuf.get_n_channels()
    stride = pixbuf.get_rowstride()
    data = pixbuf.read_pixel_bytes().get_data()

    if channels == 4 and stride == width * 4:
        return bytes(data)

    out = bytearray(width * height * 4)
    for y in range(height):
        src = y * stride
        dst = y * width * 4
        if channels == 4:
            out[dst:dst + width * 4] = data[src:src + width * 4]
        else:
            for x in range(width):
                s = src + x * channels
                d = dst + x * 4
                out[d:d + 3] = data[s:s + 3]
                out[d + 3] = 255
    return bytes(out)


def _decode_image(image: bytes) -> bytes | None:
    """Decode encoded image bytes to packed RGBA, or None when undecodable."""
    try:
        loader = GdkPixbuf.PixbufLoader()
        loader.write(image)
        loader.close()
        pixbuf = loader.get_pixbuf().add_alpha(False, 0, 0, 0)
        return _pixbuf_rgba(pixbuf), pixbuf.get_width(), pixbuf.get_height()
    except Exception as exc:
        logging.warning("avatar texture could not be decoded: %s", exc)
        return None


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = (x - edge0) / (edge1 - edge0) if edge1 != edge0 else (1.0 if x >= edge0 else 0.0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def jaw_weight(x: float, y: float, z: float) -> float:
    """Pure-Python mirror of the shader's jaw region weight, for tests and tuning."""
    below = _smoothstep(JAW_TOP, JAW_TOP - 0.10, y)
    above = _smoothstep(JAW_BOTTOM - 0.12, JAW_BOTTOM, y)
    front = _smoothstep(JAW_FRONT - 0.10, JAW_FRONT + 0.05, z)
    side = 1.0 - _smoothstep(JAW_HALF_WIDTH, JAW_HALF_WIDTH + 0.10, abs(x))
    return below * above * front * side


def eye_framing(parts) -> tuple[float, float, float] | None:
    """Frame a character from its eyes, the one landmark every face shares.

    Hair height varies a lot between characters, so framing from the top of the model cuts some faces off at
    the mouth. The distance between the eyes fixes the scale instead: a face is about three eye-spacings from
    chin to hairline, and the view shows about 7.0 eye-spacings top to bottom, centred a little below the eyes
    so the whole face and some neck are visible. Returns None when the model has no separate eye mesh.
    """
    for part in parts:
        name = part.name.lower()
        if ("poly" in name or "eye" in name) and "brow" not in name and "lash" not in name and len(part.positions):
            pos = np.asarray(part.positions, dtype=np.float64).reshape(-1, 3)
            mid_x = float(pos[:, 0].mean())
            left, right = pos[pos[:, 0] < mid_x], pos[pos[:, 0] >= mid_x]
            if len(left) == 0 or len(right) == 0:
                return None
            spacing = float(right[:, 0].mean() - left[:, 0].mean())
            if spacing <= 1e-6:
                return None
            eye_y = float(pos[:, 1].mean())
            scale = max(0.5, min(3.5, 2.0 / (7.0 * spacing)))
            return (scale, mid_x, eye_y - 0.55 * spacing)
    return None


def head_framing(parts_positions: list[np.ndarray]) -> tuple[float, float, float]:
    """Return (scale, offset_x, offset_y) framing a bust so its head fills the view.

    The head is the top of the model: every vertex with y above
    ``y_max - 0.42 * height`` belongs to the head region. The view is centred on
    that region's x-centre and on ``y_max - 0.20 * height`` in y, and scaled so
    the head region's smaller dimension fills about 85% of the clip-space view
    (which spans -1..1 in y). Models whose height/width ratio is between 0.9 and
    1.15 keep scale 1.0. The scale is clamped to 1.0-3.2.
    """
    arrays = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in parts_positions if len(p)]
    if not arrays:
        return (1.0, 0.0, 0.0)
    all_pos = np.concatenate(arrays)
    if all_pos.size == 0:
        return (1.0, 0.0, 0.0)
    x = all_pos[:, 0]
    y = all_pos[:, 1]
    y_max = float(y.max())
    y_min = float(y.min())
    x_min = float(x.min())
    x_max = float(x.max())
    height = y_max - y_min
    width = x_max - x_min
    if height <= 0.0:
        return (1.0, 0.0, 0.0)
    threshold = y_max - 0.42 * height
    head = all_pos[y > threshold]
    if head.size == 0:
        return (1.0, 0.0, 0.0)
    head_y = head[:, 1]
    head_x = head[:, 0]
    head_height = float(head_y.max() - head_y.min())
    if head_height <= 0.0:
        return (1.0, 0.0, 0.0)
    offset_x = float(head_x.mean())
    offset_y = y_max - 0.20 * height
    head_width = float(head_x.max() - head_x.min())
    if head_width <= 0.0:
        return (1.0, 0.0, 0.0)
    if width > 0.0 and 0.9 <= height / width <= 1.15:
        scale = 1.0
    else:
        scale = min(3.2, max(1.0, (0.85 * 2.0) / min(head_height, head_width)))
    return (float(scale), offset_x, offset_y)


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
    mouth_level: float,
    speaking: bool = False,
    avatar: AvatarDescriptor | None = None,
) -> list[tuple[float, float, float, float, float, float, float]]:
    level = max(0.0, min(1.0, float(mouth_level)))
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
        self._speaking_since: float | None = None
        self._viseme: dict[str, float] = {}
        self._raw_words: list[tuple[str, float, float]] = []
        self._word_timeline: list[tuple[float, float, str]] = []
        self._speech_clock = None
        self._tick_added = False
        self._avatar = default_avatar()

        self._scene = None
        self._textured_program = None
        self._background_program = None
        self._background_vao = None
        self._part_gl: list[dict] = []
        self._draw_indices: list[int] = []
        self._framing: tuple[float, float, float] = (1.0, 0.0, 0.0)

        self._supersample = True
        self._supersample_logged = False
        self._ss_fbo = None
        self._ss_rbo = None
        self._ss_tex = None
        self._ss_size = None

        self._face_motion = None
        self._face_morphs: dict[str, float] = {}
        self._face_head: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._face_gaze: tuple[float, float] = (0.0, 0.0)
        self._pose_seq = 0
        self._listening = False
        self._last_redraw = 0.0
        self._neck_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._pose_time = 0.0
        self._pose_last: float | None = None
        self._pose_frozen = False
        self._blink_force = False
        self._shader_time: float | None = None

        self._model_vertices = self._load_model(self._avatar)

        self._area = Gtk.GLArea()
        # Without a depth buffer the far side of the head is drawn over the face.
        self._area.set_has_depth_buffer(True)
        self._area.set_size_request(AVATAR_SIZE, AVATAR_SIZE)
        self._area.add_css_class("voxa-avatar")
        self._area.connect("render", self._on_render)

        self.caption = Gtk.Label(label="Offline")
        self.caption.set_xalign(0.5)
        self.caption.add_css_class("voxa-state")

        self.hint = Gtk.Label(label=STATE_HINTS[AssistantState.OFFLINE])
        self.hint.set_xalign(0.5)
        self.hint.add_css_class("voxa-state-hint")

        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.widget.add_css_class("voxa-avatar-renderer")
        self.widget.append(self._area)
        self.widget.append(self.caption)
        self.widget.append(self.hint)

    def _effective_model_path(self, avatar: AvatarDescriptor | None) -> str | None:
        """Return the explicit override first, then a downloaded character model."""
        explicit_path = avatar_model_path()
        if explicit_path:
            return explicit_path

        if avatar is not None and model_is_downloaded(avatar):
            return str(avatar.model_path)

        return None

    def _load_model(self, avatar: AvatarDescriptor | None = None):
        """Load a model when available, or return None for the procedural face.

        Tries the textured :func:`load_scene` path first; on success the scene is
        kept in ``self._scene`` and ``None`` is returned (the flat vertex list is
        unused). On :class:`GltfLoadError` it falls back to ``load_mesh`` for
        large scanned busts.
        """
        self._scene = None
        path = self._effective_model_path(avatar)
        if path is None:
            return None

        face_color, _mouth, _eyes, _hair = _face_colors(avatar)
        try:
            self._scene = load_scene(path)
            logging.info("3D avatar scene %s loaded (%d parts)", path, len(self._scene.parts))
            return None
        except GltfLoadError as exc:
            logging.debug(
                "avatar model %s is too large or textured-load failed (%s); trying the flat mesh path",
                path,
                exc,
            )

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

        if self._gl is None or not self._gl_ok:
            return

        if self._part_gl:
            self._delete_scene_gl(self._gl)
        if self._scene is not None:
            try:
                self._build_scene_gl(self._gl)
                self._draw_indices = _draw_order(self._scene.parts)
            except Exception as exc:
                logging.warning(
                    "textured avatar path failed (%s); falling back to the flat mesh", exc
                )
                self._scene = None
                self._part_gl = []
                self._textured_program = None
                self._background_program = None
                self._background_vao = None
        if self._scene is None:
            vertices = self._model_vertices or _build_mesh(
                self.mouth_level(),
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

            if self._scene is not None:
                self._ensure_render_tick()

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

        self._gl = gl
        if self._scene is not None:
            try:
                self._build_scene_gl(gl)
                self._draw_indices = _draw_order(self._scene.parts)
                return
            except Exception as exc:
                logging.warning(
                    "textured avatar path failed (%s); falling back to the flat mesh", exc
                )
                self._scene = None
                self._part_gl = []
                self._textured_program = None
                self._background_program = None
                self._background_vao = None

        program = self._compile_program(gl, VERTEX_SHADER, FRAGMENT_SHADER)
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

    def _build_scene_gl(self, gl) -> None:
        """Create per-part GL resources for the textured scene.

        Each part gets a VAO, a dynamic position VBO, static normal and uv VBOs,
        an element buffer, and a texture (1x1 white when the part has no image).
        """
        self._textured_program = self._compile_program(
            gl, TEXTURED_VERTEX_SHADER, TEXTURED_FRAGMENT_SHADER
        )
        self._background_program = self._compile_program(
            gl, BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER
        )
        self._background_vao = _single(gl.glGenVertexArrays(1))
        self._part_gl = []
        self._skin_index = skin_part_index(self._scene.parts)
        self._framing = eye_framing(self._scene.parts) or head_framing([part.positions for part in self._scene.parts])
        self._neck_pivot = self._neck_pivot_of(self._scene.parts)
        if self._face_motion is None:
            seed = zlib.crc32((self._avatar.id or "").encode()) & 0xFFFFFFFF
            self._face_motion = FaceMotion(seed=seed)
        for part_index, part in enumerate(self._scene.parts):
            positions = array("f", part.positions.reshape(-1))
            normals = array("f", part.normals.reshape(-1))
            if part.uvs is None:
                uvs = array("f", [0.0] * (len(part.positions) * 2))
            else:
                uvs = array("f", part.uvs.reshape(-1))
            indices = array("I", part.indices.reshape(-1))

            vao = _single(gl.glGenVertexArrays(1))
            pos_vbo = _single(gl.glGenBuffers(1))
            nrm_vbo = _single(gl.glGenBuffers(1))
            uv_vbo = _single(gl.glGenBuffers(1))
            ebo = _single(gl.glGenBuffers(1))

            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pos_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(positions) * 4, positions.tobytes(), gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, nrm_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(normals) * 4, normals.tobytes(), gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, uv_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(uvs) * 4, uvs.tobytes(), gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(indices) * 4, indices.tobytes(), gl.GL_STATIC_DRAW)

            gl.glEnableVertexAttribArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pos_vbo)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, c_void_p(0))
            gl.glEnableVertexAttribArray(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, nrm_vbo)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, 0, c_void_p(0))
            gl.glEnableVertexAttribArray(2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, uv_vbo)
            gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, False, 0, c_void_p(0))

            texture = self._upload_texture(gl, part)
            morph_count = len(part.morph_deltas) if part.morph_deltas is not None else 0
            self._part_gl.append(
                {
                    "vao": vao,
                    "pos_vbo": pos_vbo,
                    "nrm_vbo": nrm_vbo,
                    "uv_vbo": uv_vbo,
                    "ebo": ebo,
                    "texture": texture,
                    "part": part,
                    "weights": {},
                    "merged": {},
                    "pose_seq": -1,
                    "eye_groups": self._eye_groups_of(part),
                    "material": part_material(
                        part.name, morph_count, part_index == self._skin_index
                    ),
                }
            )

    def _upload_texture(self, gl, part):
        """Upload a part's image as a mipmapped RGBA texture, or 1x1 white."""
        decoded = _decode_image(part.image) if part.image else None
        if decoded is None:
            width, height = 1, 1
            data = b"\xff\xff\xff\xff"
        else:
            data, width, height = decoded

        texture = _single(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data,
        )
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        return texture

    def _delete_scene_gl(self, gl) -> None:
        """Release every GL resource owned by the textured scene."""
        if self._textured_program is not None:
            gl.glDeleteProgram(self._textured_program)
            self._textured_program = None
        if self._background_program is not None:
            gl.glDeleteProgram(self._background_program)
            self._background_program = None
        if self._background_vao is not None:
            gl.glDeleteVertexArrays(1, [self._background_vao])
            self._background_vao = None
        self._delete_supersample_gl(gl)
        for entry in self._part_gl:
            gl.glDeleteVertexArrays(1, [entry["vao"]])
            gl.glDeleteBuffers(1, [entry["pos_vbo"], entry["nrm_vbo"], entry["uv_vbo"], entry["ebo"]])
            if entry.get("texture") is not None:
                gl.glDeleteTextures(1, [entry["texture"]])
        self._part_gl = []

    def _is_eye_part(self, name: str) -> bool:
        name = name.lower()
        return ("poly" in name or "eye" in name) and "brow" not in name and "lash" not in name

    def _neck_pivot_of(self, parts) -> tuple[float, float, float]:
        """Pivot for head rotation: a neck point 25% below the top of the head region."""
        all_pos = [np.asarray(part.positions, dtype=np.float64).reshape(-1, 3) for part in parts if len(part.positions)]
        if not all_pos:
            return (0.0, 0.0, 0.0)
        stacked = np.concatenate(all_pos, axis=0)
        y_min, y_max = float(stacked[:, 1].min()), float(stacked[:, 1].max())
        height = max(1e-6, y_max - y_min)
        head = stacked[stacked[:, 1] > y_max - 0.42 * height]
        if len(head) == 0:
            head = stacked
        head_y_min = float(head[:, 1].min())
        head_height = max(1e-6, y_max - head_y_min)
        pivot_y = y_max - 0.25 * head_height
        return (float(head[:, 0].mean()), pivot_y, float(head[:, 2].mean()))

    def _eye_groups_of(self, part) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-eye vertex-index groups and their centres, for CPU gaze rotation."""
        if not self._is_eye_part(part.name) or not len(part.positions):
            return []
        pos = np.asarray(part.positions, dtype=np.float64).reshape(-1, 3)
        mid_x = float(pos[:, 0].mean())
        groups: list[tuple[np.ndarray, np.ndarray]] = []
        for mask in (pos[:, 0] < mid_x, pos[:, 0] >= mid_x):
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                continue
            center = pos[idx].mean(axis=0)
            groups.append((idx, center))
        return groups

    def _rotation_matrix(self, head: tuple[float, float, float]) -> np.ndarray:
        """Row-major 3x3 rotation Rz(roll) @ Rx(pitch) @ Ry(yaw), angles in degrees."""
        yaw, pitch, roll = (math.radians(angle) for angle in head)

        def axis(angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        ry = axis(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
        rz = axis(roll)
        return rz @ rx @ ry

    def _apply_gaze(self, positions: np.ndarray, eye_groups, gaze: tuple[float, float]) -> np.ndarray:
        """Rotate each eye group's vertices about its centre by the gaze angles (radians)."""
        gx, gy = gaze
        if not eye_groups or (abs(gx) < 1e-9 and abs(gy) < 1e-9):
            return positions
        out = positions.astype(np.float64, copy=True).reshape(-1, 3)
        cx, sx = math.cos(gx), math.sin(gx)
        cy, sy = math.cos(gy), math.sin(gy)
        ry = np.array([[cx, 0.0, -sx], [0.0, 1.0, 0.0], [sx, 0.0, cx]], dtype=np.float64)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cy, -sy], [0.0, sy, cy]], dtype=np.float64)
        rot = rx @ ry
        for idx, center in eye_groups:
            local = out[idx] - center
            out[idx] = local @ rot.T + center
        return out

    def _merged_weights(self) -> dict[str, float]:
        """Pose morphs merged with visemes; visemes win, mouth smiles halve while speaking."""
        merged = dict(self._face_morphs)
        for name, value in self._viseme.items():
            merged[name] = value
        if self._speaking_since is not None:
            for name in list(merged):
                if "mouthSmile" in name:
                    merged[name] *= 0.5
        return merged

    def _compute_face_pose(self) -> None:
        """Sample the face model at the current pose time into renderer state."""
        if self._face_motion is None:
            return
        speaking = self._speaking_since is not None
        pose = self._face_motion.pose(
            self._pose_time,
            speaking=speaking,
            listening=self._listening,
            energy=self.mouth_level(),
        )
        morphs = dict(pose.morphs)
        if self._blink_force:
            for name in list(morphs):
                if "eyeBlink" in name:
                    morphs[name] = 1.0
        self._face_morphs = morphs
        self._face_head = pose.head
        # FacePose gives the gaze in degrees; the eye rotation works in radians.
        self._face_gaze = (math.radians(pose.gaze[0]), math.radians(pose.gaze[1]))

    def _update_face_pose(self) -> None:
        """Advance the pose clock (unless frozen) and recompute the face pose."""
        if self._face_motion is None:
            return
        now = _now()
        if self._pose_last is None:
            self._pose_last = now
        elif not self._pose_frozen:
            self._pose_time += now - self._pose_last
            self._pose_last = now
        self._compute_face_pose()
        self._pose_seq += 1

    def _render_textured(self, gl, area) -> None:
        """Draw the textured scene: opaque parts first, then masked, then blended.

        When supersampling is on, the whole scene is drawn into an offscreen
        framebuffer at SUPERSAMPLE times the widget size and then blitted down
        with linear filtering; on any failure it falls back to direct drawing.
        """
        width = max(1, area.get_width())
        height = max(1, area.get_height())
        default_fbo = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        draw_w, draw_h = width, height
        ss_w, ss_h = width, height
        if self._supersample:
            try:
                ss_w, ss_h = supersample_size(width, height)
                if self._ss_size != (ss_w, ss_h):
                    self._delete_supersample_gl(gl)
                    self._create_supersample_gl(gl, ss_w, ss_h)
                    self._ss_size = (ss_w, ss_h)
                if not _is_true(gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER), gl.GL_FRAMEBUFFER_COMPLETE):
                    raise RuntimeError("supersample framebuffer is not complete")
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._ss_fbo)
                draw_w, draw_h = ss_w, ss_h
            except Exception as exc:
                self._supersample = False
                if not self._supersample_logged:
                    self._supersample_logged = True
                    logging.warning("supersampling failed (%s); drawing directly", exc)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, default_fbo)
        gl.glViewport(0, 0, draw_w, draw_h)
        gl.glClearColor(0.18, 0.22, 0.28, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(True)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Vertical gradient backdrop, drawn as a full-screen triangle with the
        # depth test off so it never occludes the model.
        gl.glUseProgram(self._background_program)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glBindVertexArray(self._background_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # The textured vertex shader maps +z (the face direction) to -z in clip
        # space, which reverses triangle winding; the default front face (CCW)
        # would therefore cull every face, so culling is left off and the depth
        # test resolves overlaps instead.
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glUseProgram(self._textured_program)

        time_loc = gl.glGetUniformLocation(self._textured_program, "u_time")
        view_loc = gl.glGetUniformLocation(self._textured_program, "u_view")
        aspect_loc = gl.glGetUniformLocation(self._textured_program, "u_aspect")
        tex_loc = gl.glGetUniformLocation(self._textured_program, "u_tex")
        base_loc = gl.glGetUniformLocation(self._textured_program, "u_base_color")
        mode_loc = gl.glGetUniformLocation(self._textured_program, "u_alpha_mode")
        cutoff_loc = gl.glGetUniformLocation(self._textured_program, "u_alpha_cutoff")
        scale_loc = gl.glGetUniformLocation(self._textured_program, "u_scale")
        offset_loc = gl.glGetUniformLocation(self._textured_program, "u_offset")
        rot_loc = gl.glGetUniformLocation(self._textured_program, "u_rot")
        pivot_loc = gl.glGetUniformLocation(self._textured_program, "u_pivot")
        cam_loc = gl.glGetUniformLocation(self._textured_program, "u_cam_dist")
        skin_loc = gl.glGetUniformLocation(self._textured_program, "u_skin")
        spec_loc = gl.glGetUniformLocation(self._textured_program, "u_spec")
        shine_loc = gl.glGetUniformLocation(self._textured_program, "u_shine")
        alpha_pass_loc = gl.glGetUniformLocation(self._textured_program, "u_alpha_pass")

        if time_loc >= 0:
            gl.glUniform1f(time_loc, self._shader_time if self._shader_time is not None else _now())
        if view_loc >= 0:
            gl.glUniformMatrix3fv(view_loc, 1, gl.GL_FALSE, list(_VIEW_MATRIX))
        if aspect_loc >= 0:
            gl.glUniform1f(aspect_loc, width / height)
        if scale_loc >= 0:
            gl.glUniform1f(scale_loc, self._framing[0])
        if offset_loc >= 0:
            gl.glUniform2f(offset_loc, self._framing[1], self._framing[2])
        if rot_loc >= 0:
            rot = self._rotation_matrix(self._face_head)
            gl.glUniformMatrix3fv(rot_loc, 1, gl.GL_FALSE, list(rot.reshape(-1, order="F")))
        if pivot_loc >= 0:
            gl.glUniform3f(pivot_loc, *self._neck_pivot)
        if cam_loc >= 0:
            gl.glUniform1f(cam_loc, 6.0)

        merged = self._merged_weights()
        for index in self._draw_indices:
            entry = self._part_gl[index]
            part = entry["part"]
            has_morphs = part.morph_deltas is not None
            eye_groups = entry["eye_groups"]
            morph_changed = has_morphs and merged != entry["merged"]
            gaze_changed = bool(eye_groups) and entry["pose_seq"] != self._pose_seq
            if morph_changed or gaze_changed:
                if has_morphs:
                    positions = self._scene.morphed_positions(part, merged)
                else:
                    positions = part.positions
                if eye_groups:
                    positions = self._apply_gaze(positions, eye_groups, self._face_gaze)
                flat = array("f", np.asarray(positions, dtype=np.float32).reshape(-1))
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, entry["pos_vbo"])
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, len(flat) * 4, flat.tobytes())
                entry["merged"] = dict(merged)
                entry["weights"] = dict(self._viseme)
                entry["pose_seq"] = self._pose_seq
            gl.glBindVertexArray(entry["vao"])
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, entry["texture"])
            if tex_loc >= 0:
                gl.glUniform1i(tex_loc, 0)
            if base_loc >= 0:
                gl.glUniform4fv(base_loc, 1, list(part.base_color))
            if mode_loc >= 0:
                mode = _ALPHA_MODE_RANK.get(part.alpha_mode, 1)
                gl.glUniform1f(mode_loc, 0.0 if mode == 0 else (1.0 if mode == 1 else 0.5))
            skin_value, spec_value, shine_value = entry["material"]
            if skin_loc >= 0:
                gl.glUniform1f(skin_loc, skin_value)
            if spec_loc >= 0:
                gl.glUniform1f(spec_loc, spec_value)
            if shine_loc >= 0:
                gl.glUniform1f(shine_loc, shine_value)
            if part.double_sided:
                gl.glDisable(gl.GL_CULL_FACE)
            # Indexed draw: the element buffer is part of this part's VAO.
            for alpha_pass in alpha_passes(part.alpha_mode):
                if cutoff_loc >= 0:
                    gl.glUniform1f(cutoff_loc, part.alpha_cutoff if alpha_pass == 0.0 else 0.5)
                if alpha_pass == 1.0:
                    gl.glDisable(gl.GL_BLEND)
                    gl.glDepthMask(True)
                elif alpha_pass == 2.0:
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                    gl.glDepthMask(False)
                if alpha_pass_loc >= 0:
                    gl.glUniform1f(alpha_pass_loc, alpha_pass)
                gl.glDrawElements(gl.GL_TRIANGLES, int(part.indices.size), gl.GL_UNSIGNED_INT, c_void_p(0))

        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(True)

        if self._supersample and self._ss_fbo is not None:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._ss_fbo)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, default_fbo)
            gl.glViewport(0, 0, width, height)
            gl.glBlitFramebuffer(0, 0, ss_w, ss_h, 0, 0, width, height, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, default_fbo)

    def _create_supersample_gl(self, gl, ss_w: int, ss_h: int) -> None:
        """(Re)create the offscreen colour texture and depth renderbuffer."""
        self._ss_tex = _single(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._ss_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            ss_w,
            ss_h,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self._ss_rbo = _single(gl.glGenRenderbuffers(1))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._ss_rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, ss_w, ss_h)
        self._ss_fbo = _single(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._ss_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._ss_tex, 0
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._ss_rbo
        )

    def _delete_supersample_gl(self, gl) -> None:
        """Release the offscreen supersample buffers."""
        if self._ss_tex is not None:
            gl.glDeleteTextures([self._ss_tex])
        if self._ss_rbo is not None:
            gl.glDeleteRenderbuffers([self._ss_rbo])
        if self._ss_fbo is not None:
            gl.glDeleteFramebuffers(1, [self._ss_fbo])
        self._ss_tex = None
        self._ss_rbo = None
        self._ss_fbo = None
        self._ss_size = None

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

        if self._scene is not None:
            self._render_textured(gl, area)
            return

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
            triangles = _build_mesh(self.mouth_level(), speaking=self._speaking, avatar=self._avatar)
            self._upload_mesh(gl, triangles)

        gl.glUseProgram(self._program)

        time = _now()
        time_loc = gl.glGetUniformLocation(self._program, "u_time")
        audio_loc = gl.glGetUniformLocation(self._program, "u_audio")
        jaw_loc = gl.glGetUniformLocation(self._program, "u_jaw")
        is_model_loc = gl.glGetUniformLocation(self._program, "u_is_model")
        if time_loc >= 0:
            gl.glUniform1f(time_loc, time)
        if audio_loc >= 0:
            gl.glUniform1f(audio_loc, self._audio_level)
        if jaw_loc >= 0:
            gl.glUniform1f(jaw_loc, self.mouth_level())
        if is_model_loc >= 0:
            gl.glUniform1f(is_model_loc, 1.0 if self._model_vertices is not None else 0.0)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(triangles))

    @property
    def audio_level(self) -> float:
        return self._audio_level

    def mouth_level(self) -> float:
        if self._speaking_since is not None:
            return speech_level(_now() - self._speaking_since)
        return 0.0

    def queue_render(self) -> None:
        self._area.queue_draw()

    def _ensure_render_tick(self) -> None:
        if self._tick_added:
            return
        self._tick_added = True
        self._area.add_tick_callback(self._on_tick)

    def _speech_visemes(self) -> dict[str, float]:
        """Viseme weights for the current playback time.

        Prefers the word timeline (Edge TTS word boundaries plus the speech
        clock) when both are present; otherwise falls back to the speech-like
        rhythm keyed to when speaking started.
        """
        if self._word_timeline and self._speech_clock is not None:
            return weights_at(self._word_timeline, self._speech_clock())
        return viseme_weights(_now() - self._speaking_since)

    def _on_tick(self, area, *args):
        if self._scene is not None:
            self._update_face_pose()
            if self._speaking_since is not None:
                self._viseme = blend(self._viseme, self._speech_visemes(), 0.45)
            elif self._viseme:
                self._viseme = blend(self._viseme, {}, 0.45)
            now = _now()
            interval = 1.0 / 60.0 if self._speaking_since is not None else 1.0 / 30.0
            if now - self._last_redraw >= interval:
                self._last_redraw = now
                self.queue_render()
            return GLib.SOURCE_CONTINUE
        if self._speaking_since is not None:
            self._viseme = blend(self._viseme, self._speech_visemes(), 0.45)
            self.queue_render()
            return GLib.SOURCE_CONTINUE
        if self._viseme:
            self._viseme = blend(self._viseme, {}, 0.45)
            self.queue_render()
            return GLib.SOURCE_CONTINUE
        self._tick_added = False
        return GLib.SOURCE_REMOVE

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        self._state = state
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
        self._listening = bool(active)
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
            self._speaking_since = _now()
            self._ensure_render_tick()
        else:
            self._view.remove_css_class("speaking")
            self._speaking_since = None
            self._raw_words = []
            self._word_timeline = []

    def set_word_timeline(self, words: list[tuple[str, float, float]]) -> None:
        """Accumulate Edge TTS word boundaries and rebuild the viseme timeline.

        ``words`` is the ``on_words`` payload: ``(word, start_s, duration_s)``
        relative to the start of the utterance's audio.  Each call extends the
        stored words and recomputes the spread timeline used by
        :meth:`_speech_visemes`.
        """
        self._raw_words.extend(words)
        self._word_timeline = timeline(self._raw_words)

    def set_speech_clock(self, clock) -> None:
        """Install the callable that reports current playback seconds."""
        self._speech_clock = clock

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
