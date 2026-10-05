from __future__ import annotations

import os
import struct
import zlib

import numpy as np
import pytest

from voxa.ui.gltf import MAX_MODEL_VERTICES, MODEL_RADIUS, GltfLoadError
from voxa.ui.gltf_scene import load_scene

pygltflib = pytest.importorskip("pygltflib")

FLOAT = 5126
UNSIGNED_SHORT = 5123

GRACE = "/home/rhy/voxa-characters/out/grace-long01.glb"


def _pack_floats(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _pack_shorts(values: list[int]) -> bytes:
    return struct.pack(f"{len(values)}H", *values)


def _pad_binary(data: bytes) -> bytes:
    return data + b"\x00" * (-len(data) % 4)


def _png_2x2() -> bytes:
    """A real 2x2 RGBA PNG, used as an embedded base-colour texture."""
    raw = b""
    for _ in range(2):
        raw += b"\x00" + bytes([255, 0, 0, 255, 0, 255, 0, 255])

    def chunk(typ: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    header = chunk(b"IHDR", struct.pack(">IIBBBB", 2, 2, 8, 0, 0, 0)[:11])
    return b"\x89PNG\r\n\x1a\n" + header + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _write_scene_glb(
    path,
    *,
    positions: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]] | None = None,
    texcoords: list[tuple[float, float]] | None = None,
    indices: list[int] | None = None,
    morph_targets: list[list[tuple[float, float, float]]] | None = None,
    target_names: list[str] | None = None,
    material: dict | None = None,
    nodes: list | None = None,
    scene: list[int] | None = None,
    mesh_count: int = 1,
) -> None:
    binary = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    def add_accessor(component_type: int, type_name: str, count: int, data: bytes) -> int:
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=len(binary), byteLength=len(data), target=2))
        binary.extend(_pad_binary(data))
        accessors.append(
            pygltflib.Accessor(bufferView=len(buffer_views) - 1, componentType=component_type, count=count, type=type_name)
        )
        return len(accessors) - 1

    position_index = add_accessor(FLOAT, "VEC3", len(positions), _pack_floats([v for xyz in positions for v in xyz]))
    attributes = pygltflib.Attributes(POSITION=position_index)

    if normals is not None:
        attributes.NORMAL = add_accessor(FLOAT, "VEC3", len(normals), _pack_floats([v for xyz in normals for v in xyz]))
    if texcoords is not None:
        attributes.TEXCOORD_0 = add_accessor(FLOAT, "VEC2", len(texcoords), _pack_floats([v for uv in texcoords for v in uv]))

    index_index = None
    if indices is not None:
        index_index = add_accessor(UNSIGNED_SHORT, "SCALAR", len(indices), _pack_shorts(indices))

    targets = None
    if morph_targets is not None:
        targets = []
        for deltas in morph_targets:
            accessor = add_accessor(FLOAT, "VEC3", len(deltas), _pack_floats([v for xyz in deltas for v in xyz]))
            targets.append({"POSITION": accessor})

    materials: list[pygltflib.Material] = []
    textures: list[pygltflib.Texture] = []
    images: list[pygltflib.Image] = []
    if material is not None:
        pbr = pygltflib.PbrMetallicRoughness(baseColorFactor=material.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))
        if material.get("image") is not None:
            image_bytes = material["image"]
            buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=len(binary), byteLength=len(image_bytes)))
            binary.extend(_pad_binary(image_bytes))
            images.append(pygltflib.Image(bufferView=len(buffer_views) - 1, mimeType=material.get("mime", "image/png")))
            textures.append(pygltflib.Texture(source=0))
            pbr.baseColorTexture = pygltflib.TextureInfo(index=0, texCoord=0)
        mat = pygltflib.Material(
            pbrMetallicRoughness=pbr,
            alphaMode=material.get("alphaMode", "OPAQUE"),
            alphaCutoff=material.get("alphaCutoff"),
            doubleSided=material.get("doubleSided", False),
        )
        materials.append(mat)

    primitive = pygltflib.Primitive(attributes=attributes, indices=index_index, material=0 if material is not None else None)
    if targets is not None:
        primitive.targets = targets

    mesh = pygltflib.Mesh(primitives=[primitive])
    if target_names is not None:
        mesh.extras = {"targetNames": target_names}
    glb = pygltflib.GLTF2()
    glb.buffers = [pygltflib.Buffer(byteLength=len(binary))]
    glb.bufferViews = buffer_views
    glb.accessors = accessors
    glb.meshes = [mesh for _ in range(mesh_count)]
    if materials:
        glb.materials = materials
    if textures:
        glb.textures = textures
    if images:
        glb.images = images
    if nodes is not None:
        glb.nodes = nodes
    if scene is not None:
        glb.scenes = [pygltflib.Scene(nodes=scene)]
        glb.scene = 0
    glb._glb_data = bytes(binary)
    assert glb.save(str(path))


def test_two_part_scene_normalised_jointly(tmp_path) -> None:
    path = tmp_path / "two-parts.glb"
    nodes = [
        pygltflib.Node(mesh=0, children=[1]),
        pygltflib.Node(mesh=1, translation=[2.0, 0.0, 0.0]),
    ]
    _write_scene_glb(
        path,
        positions=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        texcoords=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        indices=[0, 1, 2],
        nodes=nodes,
        scene=[0],
        mesh_count=2,
    )

    scene = load_scene(str(path))
    assert len(scene.parts) == 2
    for part in scene.parts:
        assert part.uvs is not None
        np.testing.assert_allclose(part.uvs, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], atol=1e-6)
        np.testing.assert_array_equal(part.indices, [[0, 1, 2]])

    all_positions = np.concatenate([part.positions for part in scene.parts])
    assert np.abs(all_positions).max() == pytest.approx(MODEL_RADIUS, abs=1e-4)


def test_embedded_png_texture_round_trips(tmp_path) -> None:
    path = tmp_path / "textured.glb"
    png = _png_2x2()
    _write_scene_glb(
        path,
        positions=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        material={"image": png, "mime": "image/png", "alphaMode": "MASK", "alphaCutoff": 0.3, "doubleSided": True},
    )

    scene = load_scene(str(path))
    part = scene.parts[0]
    assert part.image == png
    assert part.image_mime == "image/png"
    assert part.alpha_mode == "MASK"
    assert part.alpha_cutoff == pytest.approx(0.3)
    assert part.double_sided is True


def test_morph_targets_and_morphed_positions(tmp_path) -> None:
    path = tmp_path / "morphs.glb"
    _write_scene_glb(
        path,
        positions=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        morph_targets=[[(1.0, 0.0, 0.0)] * 3, [(0.0, 1.0, 0.0)] * 3],
        target_names=["viseme_aa", "viseme_E"],
    )

    scene = load_scene(str(path))
    part = scene.parts[0]
    assert part.morph_names == ("viseme_aa", "viseme_E")
    assert part.morph_deltas is not None
    assert part.morph_deltas.shape == (2, 3, 3)

    morphed = scene.morphed_positions(part, {"viseme_aa": 1.0})
    np.testing.assert_allclose(morphed, part.positions + part.morph_deltas[0], atol=1e-6)

    unchanged = scene.morphed_positions(part, {})
    assert unchanged is part.positions


def test_normals_computed_when_absent_are_unit_length(tmp_path) -> None:
    path = tmp_path / "no-normals.glb"
    _write_scene_glb(path, positions=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)])

    scene = load_scene(str(path))
    normals = scene.parts[0].normals
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths[lengths > 0.0], 1.0, atol=1e-5)
    assert np.all(lengths <= 1.0 + 1e-5)


def test_scene_over_vertex_budget_raises(tmp_path) -> None:
    path = tmp_path / "huge.glb"
    count = MAX_MODEL_VERTICES + 3
    positions = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    positions[:, 0] = np.arange(count, dtype=np.float32)
    _write_scene_glb(
        path,
        positions=[tuple(row) for row in positions.tolist()],
    )

    with pytest.raises(GltfLoadError) as exc:
        load_scene(str(path))
    assert "too large" in str(exc.value)


@pytest.mark.skipif(not os.path.isfile(GRACE), reason="sample character not present")
def test_grace_character_scene() -> None:
    scene = load_scene(GRACE)
    assert len(scene.parts) == 7

    body = next(part for part in scene.parts if len(part.morph_names) == 15)
    assert all(name.startswith("viseme_") for name in body.morph_names)

    with_image = sum(1 for part in scene.parts if part.image is not None)
    assert with_image >= 5
