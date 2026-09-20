"""Minimal GLB/glTF loader for downloaded 3D avatar models.

Small on purpose: it reads the first mesh of a self-contained .glb file and
returns a triangle soup in exactly the (x, y, z, r, g, b, kind) vertex shape the
GLArea renderer already uploads, so no new shader or buffer code is needed.

Textures, materials, skinning and animations are ignored in this first pass -
the geometry is vertex-lit flat, the same visual sophistication as the
procedural sphere face.  Anything the loader cannot handle raises GltfLoadError
with a specific reason so the caller can log it and fall back to the
procedural face instead of crashing.
"""

from __future__ import annotations

import os
import struct
from array import array
from typing import Any

# glTF component types and primitive modes, kept as plain integers (the spec
# values) so importing this module never requires pygltflib to be present: the
# actual parse happens inside load_mesh, and a missing library becomes a
# GltfLoadError the renderer can fall back on.
FLOAT = 5126
UNSIGNED_INT = 5125
UNSIGNED_SHORT = 5123
UNSIGNED_BYTE = 5121
SHORT = 5122
BYTE = 5120

_COMPONENT_FORMAT = {FLOAT: "f", UNSIGNED_INT: "I", UNSIGNED_SHORT: "H", UNSIGNED_BYTE: "B", SHORT: "h", BYTE: "b"}
_COMPONENT_SIZE = {component: struct.calcsize(code) for component, code in _COMPONENT_FORMAT.items()}
_ELEMENT_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

TRIANGLES = 4

# A downloaded model is normalized to this half-extent so it fits the same
# viewport as the procedural face, and huge meshes are refused rather than
# silently eating GL memory.
MODEL_RADIUS = 0.85
MAX_MODEL_VERTICES = 150_000

_LIGHT = (0.0, 0.242535625, 0.970142547)
_NORMAL_SHADE_MIN = 0.55
_NORMAL_SHADE_SPAN = 0.45


class GltfLoadError(Exception):
    """Raised when a .glb avatar model cannot be read into a usable mesh."""


def load_mesh(
    path: str,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_radius: float = MODEL_RADIUS,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Load the first mesh of a .glb file as a flat, normalized triangle list.

    Returns vertices as (x, y, z, r, g, b, kind) tuples, matching the output of
    the procedural mesh builders in avatar_3d.py.  Indices are expanded into a
    triangle soup so the existing glDrawArrays path works unchanged.
    """
    gltf = _open_glb(path)
    blob = _binary_blob(gltf)

    if not gltf.meshes:
        raise GltfLoadError("the GLB model contains no meshes")
    if len(gltf.meshes) > 1:
        raise GltfLoadError(f"multi-mesh models are not supported yet (found {len(gltf.meshes)} meshes)")

    mesh = gltf.meshes[0]
    if not mesh.primitives:
        raise GltfLoadError("mesh 0 has no primitives")

    vertices: list[tuple[float, float, float, float, float, float, float]] = []
    for index, primitive in enumerate(mesh.primitives):
        vertices.extend(_primitive_vertices(gltf, blob, index, primitive, color))

    if not vertices:
        raise GltfLoadError("mesh 0 produced no triangles")
    if len(vertices) > MAX_MODEL_VERTICES:
        raise GltfLoadError(
            f"model has {len(vertices)} vertices, more than the {MAX_MODEL_VERTICES} the renderer supports"
        )

    return _normalize(vertices, target_radius)


def _open_glb(path: str):
    try:
        import pygltflib
    except ImportError as exc:
        raise GltfLoadError("pygltflib is not installed, so .glb avatar models cannot be loaded") from exc

    path = os.path.expanduser(path)
    if not path:
        raise GltfLoadError("no avatar model path was given")
    if not path.lower().endswith(".glb"):
        raise GltfLoadError(f"only self-contained .glb models are supported, got {path}")
    if not os.path.isfile(path):
        raise GltfLoadError(f"avatar model file does not exist: {path}")

    try:
        gltf = pygltflib.GLTF2().load(path)
    except Exception as exc:
        raise GltfLoadError(f"could not parse GLB model {path}: {exc}") from exc

    if not gltf.buffers:
        raise GltfLoadError(f"GLB model {path} has no buffers")
    return gltf


def _binary_blob(gltf) -> bytes:
    """Return the single binary chunk of a .glb file.

    pygltflib leaves buffer 0's uri unset for GLB files and keeps the payload
    in the binary blob; anything else (data URIs, external bin files) is not
    supported in this pass.
    """
    if len(gltf.buffers) > 1:
        raise GltfLoadError(f"models with {len(gltf.buffers)} buffers are not supported yet")

    buffer = gltf.buffers[0]
    if buffer.uri is not None:
        raise GltfLoadError(f"buffer 0 references external data ({buffer.uri}); only self-contained .glb files are supported")

    try:
        blob = gltf.binary_blob()
    except Exception as exc:
        raise GltfLoadError(f"could not read the GLB binary chunk: {exc}") from exc
    if not blob:
        raise GltfLoadError("the GLB file has no binary chunk")
    return blob


def _read_accessor(gltf, blob: bytes, accessor_index: int, expected_type: str, what: str) -> array:
    """Read one accessor as a flat array of the expected element type."""
    try:
        accessor = gltf.accessors[accessor_index]
        view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[view.buffer]
    except (IndexError, TypeError) as exc:
        raise GltfLoadError(f"{what} points at an accessor that does not exist") from exc

    if accessor.sparse is not None:
        raise GltfLoadError(f"{what} uses a sparse accessor, which is not supported")
    if accessor.type != expected_type:
        raise GltfLoadError(f"{what} must be a {expected_type} accessor, got {accessor.type}")
    if accessor.componentType not in _COMPONENT_FORMAT:
        raise GltfLoadError(f"{what} has unsupported component type {accessor.componentType}")
    if view.byteStride:
        raise GltfLoadError(f"{what} uses an interleaved buffer view, which is not supported")
    if buffer.uri is not None:
        raise GltfLoadError(f"{what} lives in an external buffer ({buffer.uri}); only self-contained .glb files are supported")

    component = _COMPONENT_FORMAT[accessor.componentType]
    elements = accessor.count * _ELEMENT_COUNT[expected_type]
    start = view.byteOffset + accessor.byteOffset
    end = start + elements * _COMPONENT_SIZE[accessor.componentType]
    if end > len(blob):
        raise GltfLoadError(f"{what} reads past the end of the GLB binary chunk ({end} > {len(blob)})")

    values = array(component)
    values.frombytes(blob[start:end])
    return values


def _primitive_vertices(gltf, blob: bytes, index: int, primitive, color: tuple[float, float, float]):
    """Expand one primitive into flat (x, y, z, r, g, b, kind) triangle vertices."""
    mode = primitive.mode if primitive.mode is not None else TRIANGLES
    if mode != TRIANGLES:
        raise GltfLoadError(f"primitive {index} draws {mode}, only triangle primitives are supported")

    attributes = primitive.attributes
    if attributes.POSITION is None:
        raise GltfLoadError(f"primitive {index} has no POSITION attribute")

    positions = _read_accessor(gltf, blob, attributes.POSITION, "VEC3", f"primitive {index} positions")
    if len(positions) < 3:
        raise GltfLoadError(f"primitive {index} has no position vertices")

    normals = None
    if attributes.NORMAL is not None:
        normals = _read_accessor(gltf, blob, attributes.NORMAL, "VEC3", f"primitive {index} normals")
        if len(normals) != len(positions):
            raise GltfLoadError(f"primitive {index} has {len(normals)} normals for {len(positions)} positions")

    # Texcoords are parsed and validated but unused: materials and textures are
    # out of scope for the first pass, yet a broken accessor should still be loud.
    if attributes.TEXCOORD_0 is not None:
        _read_accessor(gltf, blob, attributes.TEXCOORD_0, "VEC2", f"primitive {index} texcoords")

    if primitive.indices is None:
        triangle_count = len(positions) // 3
        if len(positions) % 3:
            raise GltfLoadError(f"primitive {index} has a position count that is not a multiple of 3")
        indices = range(triangle_count * 3)
    else:
        indices = _read_accessor(gltf, blob, primitive.indices, "SCALAR", f"primitive {index} indices")
        if len(indices) % 3:
            raise GltfLoadError(f"primitive {index} has {len(indices)} indices, not a multiple of 3")

    vertex_count = len(positions) // 3
    vertices = []
    for position_index in indices:
        if position_index >= vertex_count:
            raise GltfLoadError(f"primitive {index} index {position_index} is outside the {vertex_count} vertices")
        offset = position_index * 3
        x, y, z = positions[offset], positions[offset + 1], positions[offset + 2]
        r, g, b = _shaded_color(normals, offset, color)
        vertices.append((x, y, z, r, g, b, 0.0))
    return vertices


def _shaded_color(normals, offset: int, color: tuple[float, float, float]):
    """Bake a simple normal-based light term into the flat color."""
    if normals is None:
        return color
    nx, ny, nz = normals[offset], normals[offset + 1], normals[offset + 2]
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length <= 0.0:
        return color
    dot = (nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2]) / length
    shade = max(0.0, min(1.0, dot))
    factor = _NORMAL_SHADE_MIN + _NORMAL_SHADE_SPAN * shade
    return (color[0] * factor, color[1] * factor, color[2] * factor)


def _normalize(vertices, target_radius: float):
    """Center the model on the origin and scale it into the avatar viewport."""
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    min_z = min(v[2] for v in vertices)
    max_z = max(v[2] for v in vertices)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0
    extent = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
    if extent <= 0.0:
        raise GltfLoadError("the model has no extent, so it cannot be scaled into the viewport")

    scale = target_radius / extent
    return [
        ((x - center_x) * scale, (y - center_y) * scale, (z - center_z) * scale, r, g, b, kind)
        for x, y, z, r, g, b, kind in vertices
    ]


def model_vertex_count(vertices: list[Any]) -> int:
    """Number of triangles in a loaded mesh (vertices are stored per triangle)."""
    return len(vertices) // 3
