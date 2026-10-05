"""Scene loader that keeps textures, UVs, normals and morph targets.

Unlike ``gltf.load_mesh``, which flattens a model to a coloured triangle soup,
this module keeps one :class:`Part` per mesh primitive with its positions,
normals, UVs, indices, material, base-colour texture bytes and morph-target
deltas, all normalised into the shared avatar viewport.  The textured renderer
and the viseme morphs need exactly this, so nothing is thrown away.

The low-level GLB plumbing (opening the file, reading the binary chunk, reading
accessors, walking the scene graph and computing node matrices) is reused
verbatim from :mod:`voxa.ui.gltf`; only the per-primitive bookkeeping and the
single whole-scene normalisation are new here.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

import numpy as np

from .gltf import (
    _COMPONENT_DTYPE,
    MAX_MODEL_VERTICES,
    MODEL_RADIUS,
    TRIANGLES,
    GltfLoadError,
    _binary_blob,
    _open_glb,
    _read_accessor,
    _rotation_matrix,
    _scene_meshes,
)

log = logging.getLogger(__name__)

_IDENTITY_TRANSLATION = np.zeros(3, dtype=np.float64)


@dataclass(slots=True)
class Part:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray | None
    indices: np.ndarray
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    image: bytes | None = None
    image_mime: str = ""
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    morph_names: tuple[str, ...] = ()
    morph_deltas: np.ndarray | None = None


@dataclass(slots=True)
class Scene:
    parts: list[Part]

    def vertex_count(self) -> int:
        return sum(len(part.positions) for part in self.parts)

    def morph_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for part in self.parts:
            names.update(part.morph_names)
        return tuple(sorted(names))

    def morphed_positions(self, part: Part, weights: dict[str, float]) -> np.ndarray:
        """Return ``part.positions`` plus the weighted morph deltas this part has.

        The original array object is returned (no copy) when no applicable
        weight is non-zero.
        """
        names = part.morph_names
        accumulated: np.ndarray | None = None
        for name, weight in weights.items():
            if weight == 0.0 or name not in names:
                continue
            if part.morph_deltas is None:
                continue
            term = weight * part.morph_deltas[names.index(name)]
            accumulated = term if accumulated is None else accumulated + term
        if accumulated is None:
            return part.positions
        return part.positions + accumulated


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return ``vectors`` rescaled to unit length, leaving zero rows at zero."""
    lengths = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors)
    valid = lengths > 0.0
    out[valid] = vectors[valid] / lengths[valid, None]
    return out


def _computed_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Per-vertex smooth normals from face cross products, no Python loops."""
    first = positions[indices[:, 0]]
    second = positions[indices[:, 1]]
    third = positions[indices[:, 2]]
    face_normals = np.cross(second - first, third - first)
    accumulated = np.zeros_like(positions)
    np.add.at(accumulated, indices[:, 0], face_normals)
    np.add.at(accumulated, indices[:, 1], face_normals)
    np.add.at(accumulated, indices[:, 2], face_normals)
    return accumulated


def _image_bytes(gltf, blob: bytes, base_dir: str, material) -> tuple[bytes | None, str]:
    """Return the base-colour texture's encoded bytes and its mime type.

    A missing or unreadable image yields ``(None, "")`` and is only logged,
    never raised: a texture the renderer cannot find is not a fatal error.
    """
    if material is None:
        return None, ""
    texture_info = material.pbrMetallicRoughness.baseColorTexture
    if texture_info is None or texture_info.index is None:
        return None, ""

    try:
        texture = gltf.textures[texture_info.index]
        image = gltf.images[texture.source]
    except (IndexError, TypeError):
        log.debug("base colour texture points at a texture or image that does not exist")
        return None, ""

    if image.bufferView is not None:
        try:
            view = gltf.bufferViews[image.bufferView]
            data = blob[view.byteOffset : view.byteOffset + view.byteLength]
        except (IndexError, TypeError):
            log.debug("base colour image buffer view does not exist")
            return None, ""
        return bytes(data), image.mimeType or ""

    if image.uri:
        uri = image.uri
        if uri.startswith("data:"):
            header, _, payload = uri[5:].partition(";")
            encoding, _, payload = payload.partition(",")
            if encoding != "base64":
                log.debug("base colour image data URI is not base64 encoded")
                return None, ""
            try:
                return base64.b64decode(payload), header
            except ValueError:
                log.debug("base colour image data URI could not be decoded")
                return None, ""
        try:
            with open(os.path.join(base_dir, uri), "rb") as handle:
                return handle.read(), image.mimeType or ""
        except OSError:
            log.debug("base colour image external file %s could not be read", uri)
            return None, ""

    log.debug("base colour image has neither a buffer view nor a uri")
    return None, ""


def _read_view(gltf, blob: bytes, view, component_type: int, components: int) -> np.ndarray:
    """Read a raw buffer view as ``(count, components)`` (or ``(count,)`` for scalars)."""
    dtype = _COMPONENT_DTYPE[component_type]
    buffer = gltf.buffers[view.buffer]
    if buffer.uri is not None:
        raise GltfLoadError("buffer view lives in an external buffer")
    element_size = components * dtype.itemsize
    stride = view.byteStride or 0
    if stride:
        count = (view.byteLength - element_size) // stride + 1 if view.byteLength >= element_size else 0
    else:
        count = view.byteLength // element_size
    start = view.byteOffset
    if start + (count - 1) * (stride or element_size) + element_size > len(blob) and count:
        raise GltfLoadError("buffer view reads past the end of the GLB binary chunk")
    if stride:
        values = np.ndarray(
            shape=(count, components),
            dtype=dtype,
            buffer=blob,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()
    else:
        values = np.frombuffer(blob, dtype=dtype, count=count * components, offset=start).reshape(count, components)
    if components == 1:
        values = values.reshape(count)
    return values


def _read_sparse_vec3(gltf, blob: bytes, accessor_index: int) -> np.ndarray:
    """Densify a sparse VEC3 accessor into ``(count, 3)`` float32 deltas."""
    accessor = gltf.accessors[accessor_index]
    sparse = accessor.sparse
    index_view = gltf.bufferViews[sparse.indices.bufferView]
    value_view = gltf.bufferViews[sparse.values.bufferView]
    indices = _read_view(gltf, blob, index_view, sparse.indices.componentType, 1).astype(np.int64)
    values = _read_view(gltf, blob, value_view, accessor.componentType, 3)
    dense = np.zeros((accessor.count, 3), dtype=np.float32)
    dense[indices] = values
    return dense


def _read_primitive(gltf, blob: bytes, primitive, rotation: np.ndarray, translation: np.ndarray) -> tuple | None:
    """Expand one primitive into world-space arrays plus its material handle.

    Returns ``(positions, normals_or_None, uvs_or_None, indices, material,
    deltas_or_None)`` where positions and deltas are float64 in world space
    (deltas carry only the node's rotation/scale, not its translation), or
    ``None`` for primitives that are not triangles.
    """
    mode = primitive.mode if primitive.mode is not None else TRIANGLES
    if mode != TRIANGLES:
        return None

    attributes = primitive.attributes
    if attributes.POSITION is None:
        raise GltfLoadError("primitive has no POSITION attribute")

    positions = _read_accessor(gltf, blob, attributes.POSITION, "VEC3", "primitive positions").astype(np.float64)
    if len(positions) < 3:
        raise GltfLoadError("primitive has no position vertices")
    positions_world = positions @ rotation.T + translation

    if attributes.NORMAL is not None:
        normals = _read_accessor(gltf, blob, attributes.NORMAL, "VEC3", "primitive normals").astype(np.float64)
        if len(normals) != len(positions):
            raise GltfLoadError("primitive has a normal count that does not match its positions")
        normals_world = normals @ rotation.T
    else:
        normals_world = None

    uvs = None
    if attributes.TEXCOORD_0 is not None:
        uvs = _read_accessor(gltf, blob, attributes.TEXCOORD_0, "VEC2", "primitive texcoords").astype(np.float64)
        if len(uvs) != len(positions):
            raise GltfLoadError("primitive has a texcoord count that does not match its positions")

    if primitive.indices is None:
        if len(positions) % 3:
            raise GltfLoadError("primitive has a position count that is not a multiple of 3")
        indices = np.arange(len(positions), dtype=np.uint32).reshape(-1, 3)
    else:
        raw_indices = _read_accessor(gltf, blob, primitive.indices, "SCALAR", "primitive indices")
        if len(raw_indices) % 3:
            raise GltfLoadError("primitive has an index count that is not a multiple of 3")
        if len(raw_indices) and int(raw_indices.max()) >= len(positions):
            bad = int(raw_indices.max())
            raise GltfLoadError(f"primitive index {bad} is outside the {len(positions)} vertices")
        indices = raw_indices.reshape(-1, 3).astype(np.uint32)

    deltas = None
    if primitive.targets:
        stacked: list[np.ndarray] = []
        for target in primitive.targets:
            accessor = target.get("POSITION") if isinstance(target, dict) else None
            if accessor is None:
                continue
            accessor_obj = gltf.accessors[accessor]
            if accessor_obj is not None and accessor_obj.sparse is not None:
                delta = _read_sparse_vec3(gltf, blob, accessor).astype(np.float64)
            else:
                delta = _read_accessor(gltf, blob, accessor, "VEC3", "morph target").astype(np.float64)
            if len(delta) != len(positions):
                raise GltfLoadError("morph target has a count that does not match its positions")
            stacked.append(delta @ rotation.T)
        if stacked:
            deltas = np.stack(stacked)

    material = None
    if primitive.material is not None:
        try:
            material = gltf.materials[primitive.material]
        except (IndexError, TypeError):
            material = None

    return positions_world, normals_world, uvs, indices, material, deltas


def load_scene(path: str, target_radius: float = MODEL_RADIUS) -> Scene:
    """Load every mesh of a .glb file as a textured, morph-aware :class:`Scene`.

    One :class:`Part` is produced per triangle primitive of every scene node.
    Positions and morph deltas are pushed through each node's world matrix, then
    the whole scene is centred and scaled once (the same rule as
    ``gltf._normalize``) so all parts share one normalised space.  Raises
    :class:`GltfLoadError` when the scene exceeds the textured renderer's vertex
    budget, so the caller can fall back to ``gltf.load_mesh`` (which decimates).
    """
    gltf = _open_glb(path)
    blob = _binary_blob(gltf)
    base_dir = os.path.dirname(os.path.expanduser(path))

    raw: list[tuple] = []
    for mesh_index, world in _scene_meshes(gltf):
        mesh = gltf.meshes[mesh_index]
        rotation = _rotation_matrix(world)
        translation = np.array([world[12], world[13], world[14]], dtype=np.float64)
        for primitive in mesh.primitives:
            part = _read_primitive(gltf, blob, primitive, rotation, translation)
            if part is not None:
                raw.append((part, mesh))

    if not raw:
        raise GltfLoadError("the scene produced no triangles")

    total_vertices = sum(len(entry[0][0]) for entry in raw)
    if total_vertices > MAX_MODEL_VERTICES:
        raise GltfLoadError("model too large for the textured renderer, use load_mesh instead")

    all_positions = np.concatenate([entry[0][0] for entry in raw])
    min_corner = all_positions.min(axis=0)
    max_corner = all_positions.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    extent = (max_corner - min_corner).max() / 2.0
    if extent <= 0.0:
        raise GltfLoadError("the scene has no extent, so it cannot be scaled into the viewport")
    scale = target_radius / extent

    parts: list[Part] = []
    for (positions_world, normals_world, uvs, indices, material, deltas_world), mesh in raw:
        positions = ((positions_world - center) * scale).astype(np.float32)

        if normals_world is None:
            normals_world = _computed_normals(positions_world, indices)
        normals = _unit_vectors(normals_world).astype(np.float32)

        morph_deltas = (deltas_world * scale).astype(np.float32) if deltas_world is not None else None
        if morph_deltas is not None:
            target_count = len(morph_deltas)
            names = mesh.extras.get("targetNames") if isinstance(mesh.extras, dict) else None
            if names is not None and len(names) == target_count:
                morph_names = tuple(str(name) for name in names)
            else:
                morph_names = tuple(f"target_{index}" for index in range(target_count))
        else:
            morph_names = ()

        image, image_mime = _image_bytes(gltf, blob, base_dir, material)
        if material is not None:
            base_color = tuple(float(value) for value in (material.pbrMetallicRoughness.baseColorFactor or [1.0, 1.0, 1.0, 1.0]))
            alpha_mode = material.alphaMode or "OPAQUE"
            alpha_cutoff = 0.5 if material.alphaCutoff is None else float(material.alphaCutoff)
            double_sided = bool(material.doubleSided)
        else:
            base_color = (1.0, 1.0, 1.0, 1.0)
            alpha_mode = "OPAQUE"
            alpha_cutoff = 0.5
            double_sided = False

        parts.append(
            Part(
                name=mesh.name or "",
                positions=positions,
                normals=normals,
                uvs=uvs.astype(np.float32) if uvs is not None else None,
                indices=indices,
                base_color=base_color,
                image=image,
                image_mime=image_mime,
                alpha_mode=alpha_mode,
                alpha_cutoff=alpha_cutoff,
                double_sided=double_sided,
                morph_names=morph_names,
                morph_deltas=morph_deltas,
            )
        )

    return Scene(parts=parts)
