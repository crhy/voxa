"""Minimal GLB/glTF loader for downloaded 3D avatar models.

Small on purpose: it reads every mesh reachable from the default scene of a
self-contained .glb file, applies each node's world matrix, and returns one
merged triangle soup in exactly the (x, y, z, r, g, b, kind) vertex shape the
GLArea renderer already uploads, so no new shader or buffer code is needed.

Textures, materials, skinning and animations are ignored in this first pass -
the geometry is vertex-lit flat, the same visual sophistication as the
procedural sphere face.  Anything the loader cannot handle raises GltfLoadError
with a specific reason so the caller can log it and fall back to the
procedural face instead of crashing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

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

_COMPONENT_DTYPE = {
    FLOAT: np.dtype("float32"),
    UNSIGNED_INT: np.dtype("uint32"),
    UNSIGNED_SHORT: np.dtype("uint16"),
    UNSIGNED_BYTE: np.dtype("uint8"),
    SHORT: np.dtype("int16"),
    BYTE: np.dtype("int8"),
}
_ELEMENT_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

TRIANGLES = 4

# A downloaded model is normalized to this half-extent so it fits the same
# viewport as the procedural face, and huge meshes are decimated down to this
# vertex budget instead of being refused.
MODEL_RADIUS = 0.85
MAX_MODEL_VERTICES = 150_000

# Above this triangle count the transform, decimation and shading run in numpy
# instead of Python loops.
_NUMPY_THRESHOLD = 20_000

_LIGHT = (0.0, 0.242535625, 0.970142547)
_NORMAL_SHADE_MIN = 0.55
_NORMAL_SHADE_SPAN = 0.45

log = logging.getLogger(__name__)

IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class GltfLoadError(Exception):
    """Raised when a .glb avatar model cannot be read into a usable mesh."""


def load_mesh(
    path: str,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_radius: float = MODEL_RADIUS,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Load every mesh of a .glb file as one flat, normalized triangle list.

    Returns vertices as (x, y, z, r, g, b, kind) tuples, matching the output of
    the procedural mesh builders in avatar_3d.py.  Indices are expanded into a
    triangle soup so the existing glDrawArrays path works unchanged.  Each mesh
    is transformed by the world matrix of the scene node that references it;
    files without scenes fall back to loading every mesh with the identity.
    """
    gltf = _open_glb(path)
    blob = _binary_blob(gltf)

    if not gltf.meshes:
        raise GltfLoadError("the GLB model contains no meshes")

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, list[float]]] = []
    for mesh_index, world in _scene_meshes(gltf):
        mesh = gltf.meshes[mesh_index]
        for index, primitive in enumerate(mesh.primitives):
            part = _primitive_arrays(gltf, blob, index, primitive, world)
            if part is not None:
                parts.append(part)

    if not parts:
        raise GltfLoadError("the model produced no triangles")

    triangle_count = sum(len(triangles) // 3 for _, triangles, _, _ in parts)
    if triangle_count <= MAX_MODEL_VERTICES // 3 and triangle_count <= _NUMPY_THRESHOLD:
        vertices: list[tuple[float, float, float, float, float, float, float]] = []
        for positions, triangles, normals, world in parts:
            vertices.extend(_python_vertices(positions, triangles, normals, world, color))
        return _normalize(vertices, target_radius)

    if len(parts) == 1:
        positions, triangles, normals, world = parts[0]
        if normals is not None:
            normals = normals.astype(np.float64) @ _rotation_matrix(world).T
    else:
        positions = np.concatenate([part[0] for part in parts])
        triangles = []
        offset = 0
        normals_parts = []
        for part_positions, part_triangles, part_normals, _part_world in parts:
            triangles.append(part_triangles + offset)
            offset += len(part_positions)
            normals_parts.append(part_normals)
        triangles = np.concatenate(triangles)
        if all(part_normals is not None for part_normals in normals_parts):
            normals = np.concatenate(
                [part_normals.astype(np.float64) @ _rotation_matrix(part_world).T for _, _, part_normals, part_world in parts]
            )
        else:
            normals = None

    decimated = False
    if len(triangles) * 3 > MAX_MODEL_VERTICES:
        positions, triangles = decimate(positions, triangles, MAX_MODEL_VERTICES)
        decimated = True

    if decimated:
        normals = None

    return _normalize_numpy(positions, triangles, normals, color, target_radius)


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


def _mat_mul(a: list[float], b: list[float]) -> list[float]:
    """Multiply two 16-float column-major 4x4 matrices, returning a * b."""
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3]
    return out


def _transform_point(m: list[float], x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply a 16-float column-major matrix to a point."""
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )


def _rotation_matrix(world: list[float]) -> np.ndarray:
    """Return the 3x3 linear part of a 16-float column-major 4x4 matrix."""
    return np.array(
        [
            [world[0], world[4], world[8]],
            [world[1], world[5], world[9]],
            [world[2], world[6], world[10]],
        ],
        dtype=np.float64,
    )


def _node_matrix(node) -> list[float]:
    """Local matrix of a node: its explicit 16-float matrix, or TRS if absent."""
    if node.matrix is not None:
        return [float(value) for value in node.matrix]

    matrix = list(IDENTITY)
    scale = node.scale if node.scale is not None else None
    if scale is not None:
        local = list(IDENTITY)
        local[0] = float(scale[0])
        local[5] = float(scale[1])
        local[10] = float(scale[2])
        matrix = _mat_mul(matrix, local)

    rotation = node.rotation
    if rotation is not None:
        x, y, z, w = (float(value) for value in rotation)
        local = [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0,
            2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0,
            2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        matrix = _mat_mul(matrix, local)

    translation = node.translation
    if translation is not None:
        local = list(IDENTITY)
        local[12] = float(translation[0])
        local[13] = float(translation[1])
        local[14] = float(translation[2])
        matrix = _mat_mul(matrix, local)

    return matrix


def _scene_meshes(gltf) -> list[tuple[int, list[float]]]:
    """Walk the default scene's nodes and yield (mesh_index, world_matrix) pairs.

    A node's world matrix is its parent's world matrix times its local matrix.
    Files with no scenes fall back to every mesh with the identity matrix.
    """
    scenes = gltf.scenes or []
    if not scenes:
        return [(mesh_index, list(IDENTITY)) for mesh_index in range(len(gltf.meshes))]

    scene_index = gltf.scene if gltf.scene is not None else 0
    if scene_index >= len(scenes):
        scene_index = 0

    found: list[tuple[int, list[float]]] = []
    stack: list[tuple[int, list[float]]] = [(node_index, list(IDENTITY)) for node_index in (scenes[scene_index].nodes or [])]
    while stack:
        node_index, world = stack.pop()
        try:
            node = gltf.nodes[node_index]
        except (IndexError, TypeError):
            continue
        local_world = _mat_mul(world, _node_matrix(node))
        if node.mesh is not None:
            found.append((node.mesh, local_world))
        for child in node.children or []:
            stack.append((child, local_world))
    return found


def _read_accessor(gltf, blob: bytes, accessor_index: int, expected_type: str, what: str) -> np.ndarray:
    """Read one accessor as a numpy array: (count, components) for vectors, (count,) for scalars.

    Buffer views with a byteStride larger than the element size are read as
    interleaved data via numpy strides; a stride smaller than the element size
    is still an error.
    """
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
    if accessor.componentType not in _COMPONENT_DTYPE:
        raise GltfLoadError(f"{what} has unsupported component type {accessor.componentType}")
    if buffer.uri is not None:
        raise GltfLoadError(f"{what} lives in an external buffer ({buffer.uri}); only self-contained .glb files are supported")

    dtype = _COMPONENT_DTYPE[accessor.componentType]
    components = _ELEMENT_COUNT[expected_type]
    element_size = components * dtype.itemsize
    start = view.byteOffset + accessor.byteOffset
    stride = view.byteStride or 0
    if stride:
        if stride < element_size:
            raise GltfLoadError(
                f"{what} has buffer view byteStride {stride}, smaller than its {element_size}-byte element size"
            )
        end = start + (accessor.count - 1) * stride + element_size if accessor.count else start
    else:
        end = start + accessor.count * element_size
    if end > len(blob):
        raise GltfLoadError(f"{what} reads past the end of the GLB binary chunk ({end} > {len(blob)})")

    if stride:
        values = np.ndarray(
            shape=(accessor.count, components),
            dtype=dtype,
            buffer=blob,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()
    else:
        values = np.frombuffer(blob, dtype=dtype, count=accessor.count * components, offset=start)
    if components == 1:
        values = values.reshape(accessor.count)
    else:
        values = values.reshape(accessor.count, components)
    return values


def _primitive_arrays(
    gltf,
    blob: bytes,
    index: int,
    primitive,
    world: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[float]] | None:
    """Expand one primitive into transformed positions and triangle indices.

    Returns (positions, triangles, normals, world) where positions is (N, 3),
    triangles is (M, 3) indices into positions, and normals is (N, 3) or None.
    Returns None for primitives that are not triangles.
    """
    mode = primitive.mode if primitive.mode is not None else TRIANGLES
    if mode != TRIANGLES:
        log.debug("skipping primitive %d: mode %s is not triangles", index, mode)
        return None

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
        if len(positions) % 3:
            raise GltfLoadError(f"primitive {index} has a position count that is not a multiple of 3")
        triangles = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
    else:
        indices = _read_accessor(gltf, blob, primitive.indices, "SCALAR", f"primitive {index} indices")
        if len(indices) % 3:
            raise GltfLoadError(f"primitive {index} has {len(indices)} indices, not a multiple of 3")
        if len(indices) and int(indices.max()) >= len(positions):
            bad = int(indices.max())
            raise GltfLoadError(f"primitive {index} index {bad} is outside the {len(positions)} vertices")
        triangles = indices.reshape(-1, 3).astype(np.int64)

    return positions, triangles, normals, world


def _python_vertices(
    positions: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray | None,
    world: list[float],
    color: tuple[float, float, float],
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Expand a primitive into flat (x, y, z, r, g, b, kind) triangle vertices in Python."""
    vertices = []
    for triangle in triangles:
        for position_index in triangle:
            px, py, pz = positions[position_index]
            x, y, z = _transform_point(world, float(px), float(py), float(pz))
            r, g, b = _shaded_color(normals[position_index], world, color) if normals is not None else color
            vertices.append((x, y, z, r, g, b, 0.0))
    return vertices


def _shaded_color(normal, world: list[float], color: tuple[float, float, float]):
    """Bake a simple normal-based light term into the flat color."""
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    nx, ny, nz = (
        world[0] * nx + world[4] * ny + world[8] * nz,
        world[1] * nx + world[5] * ny + world[9] * nz,
        world[2] * nx + world[6] * ny + world[10] * nz,
    )
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length <= 0.0:
        return color
    dot = (nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2]) / length
    shade = max(0.0, min(1.0, dot))
    factor = _NORMAL_SHADE_MIN + _NORMAL_SHADE_SPAN * shade
    return (color[0] * factor, color[1] * factor, color[2] * factor)


def decimate(
    positions: np.ndarray,
    triangles: np.ndarray,
    target_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a triangle soup to at most target_vertices via vertex clustering.

    positions is (N, 3), triangles is (M, 3) indices into positions.  Each grid
    cell of the bounding box collapses to the mean position of its vertices,
    triangle indices are remapped, and triangles whose corners merge are
    dropped.  The grid starts at 256 cells per axis and halves until the
    triangle budget is met.
    """
    grid = 256
    while True:
        min_corner = positions.min(axis=0)
        extent = positions.max(axis=0) - min_corner
        extent = np.where(extent > 0.0, extent, 1.0)
        cell = np.floor((positions - min_corner) / extent * grid).astype(np.int64)
        cell = np.clip(cell, 0, grid - 1)
        keys = (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]
        _, inverse = np.unique(keys, return_inverse=True)

        sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
        counts = np.bincount(inverse)
        np.add.at(sums, inverse, positions)
        new_positions = (sums / counts[:, None]).astype(np.float32)

        remapped = inverse[triangles]
        keep = (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
        new_triangles = remapped[keep]

        if 3 * len(new_triangles) <= target_vertices or grid <= 1:
            return new_positions, new_triangles
        grid = max(1, grid // 2)


def _normalize_numpy(
    positions: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray | None,
    color: tuple[float, float, float],
    target_radius: float,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Center, scale and shade a merged triangle soup in numpy, then flatten it."""
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    extent = (max_corner - min_corner).max() / 2.0
    if extent <= 0.0:
        raise GltfLoadError("the model has no extent, so it cannot be scaled into the viewport")
    scale = target_radius / extent

    light = np.array(_LIGHT)
    if normals is None:
        face_normals = np.cross(
            positions[triangles[:, 1]] - positions[triangles[:, 0]],
            positions[triangles[:, 2]] - positions[triangles[:, 0]],
        )
        lengths = np.linalg.norm(face_normals, axis=1)
        valid = lengths > 0.0
        face_normals = face_normals[valid]
        triangles = triangles[valid]
        dot = face_normals @ light / lengths[valid]
        factor = _NORMAL_SHADE_MIN + _NORMAL_SHADE_SPAN * np.clip(dot, 0.0, 1.0)
        vertex_rgb = np.array(color) * factor[:, None, None]
    else:
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        dot = np.zeros(len(normals))
        dot[valid] = (normals[valid] @ light) / lengths[valid]
        factor = _NORMAL_SHADE_MIN + _NORMAL_SHADE_SPAN * np.clip(dot, 0.0, 1.0)
        vertex_rgb = np.array(color) * factor[triangles][:, :, None]

    xyz = (positions[triangles] - center) * scale
    out = np.empty((len(triangles), 3, 7), dtype=np.float64)
    out[:, :, :3] = xyz
    out[:, :, 3:6] = vertex_rgb
    out[:, :, 6] = 0.0
    return [tuple(float(value) for value in vertex) for vertex in out.reshape(-1, 7)]


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
