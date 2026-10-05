from __future__ import annotations

import struct

import pytest

from voxa.ui.gltf import GltfLoadError, _mat_mul, _node_matrix, _transform_point, load_mesh, model_vertex_count

pygltflib = pytest.importorskip("pygltflib")

FLOAT = 5126
UNSIGNED_SHORT = 5123


def _pack_floats(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _pack_shorts(values: list[int]) -> bytes:
    return struct.pack(f"{len(values)}H", *values)


def _pad_binary(data: bytes) -> bytes:
    return data + b"\x00" * (-len(data) % 4)


def _write_synthetic_glb(
    path,
    *,
    positions: list[tuple[float, float, float]] | None = None,
    normals: list[tuple[float, float, float]] | None = None,
    texcoords: list[tuple[float, float]] | None = None,
    indices: list[int] | None = None,
    mesh_count: int = 1,
    nodes: list | None = None,
    scene: list[int] | None = None,
) -> None:
    if positions is None:
        positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    if indices is None:
        indices = [0, 1, 2]

    binary = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    def add_accessor(component_type: int, type_name: str, count: int, data: bytes) -> int:
        view_index = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(buffer=0, byteOffset=len(binary), byteLength=len(data), target=2)
        )
        binary.extend(_pad_binary(data))
        accessors.append(
            pygltflib.Accessor(
                bufferView=view_index,
                componentType=component_type,
                count=count,
                type=type_name,
            )
        )
        return len(accessors) - 1

    position_index = add_accessor(
        FLOAT,
        "VEC3",
        len(positions),
        _pack_floats([value for xyz in positions for value in xyz]),
    )

    attributes = pygltflib.Attributes(POSITION=position_index)
    normal_index = None
    if normals is not None:
        normal_index = add_accessor(
            FLOAT,
            "VEC3",
            len(normals),
            _pack_floats([value for xyz in normals for value in xyz]),
        )
        attributes.NORMAL = normal_index

    if texcoords is not None:
        attributes.TEXCOORD_0 = add_accessor(
            FLOAT,
            "VEC2",
            len(texcoords),
            _pack_floats([value for uv in texcoords for value in uv]),
        )

    index_index = None
    if indices is not None:
        index_index = add_accessor(UNSIGNED_SHORT, "SCALAR", len(indices), _pack_shorts(indices))

    glb = pygltflib.GLTF2()
    glb.buffers = [pygltflib.Buffer(byteLength=len(binary))]
    glb.bufferViews = buffer_views
    glb.accessors = accessors
    primitive = pygltflib.Primitive(attributes=attributes, indices=index_index)
    glb.meshes = [pygltflib.Mesh(primitives=[primitive]) for _ in range(mesh_count)]
    if nodes is not None:
        glb.nodes = nodes
    if scene is not None:
        glb.scenes = [pygltflib.Scene(nodes=scene)]
        glb.scene = 0
    glb._glb_data = bytes(binary)
    assert glb.save(str(path))


def test_synthetic_triangle_glb_loads_as_triangle_soup(tmp_path) -> None:
    path = tmp_path / "triangle.glb"
    _write_synthetic_glb(
        path,
        normals=[(0.0, 0.0, 1.0)] * 3,
        texcoords=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    vertices = load_mesh(path)
    assert len(vertices) == 3
    assert model_vertex_count(vertices) == 1
    assert all(len(vertex) == 7 for vertex in vertices)

    x_values = [vertex[0] for vertex in vertices]
    y_values = [vertex[1] for vertex in vertices]
    assert max(x_values) - min(x_values) == pytest.approx(1.7)
    assert max(y_values) - min(y_values) == pytest.approx(1.7)
    assert max(abs(vertex[0]) for vertex in vertices) <= 0.85
    assert max(abs(vertex[1]) for vertex in vertices) <= 0.85

    red, green, blue = vertices[0][3], vertices[0][4], vertices[0][5]
    assert 0.9 < red < 1.0
    assert 0.9 < green < 1.0
    assert 0.9 < blue < 1.0


def test_synthetic_glb_without_normals_keeps_color(tmp_path) -> None:
    path = tmp_path / "no-normals.glb"
    _write_synthetic_glb(path)
    vertices = load_mesh(path, color=(0.25, 0.5, 0.75))
    assert all(vertex[3:6] == (0.25, 0.5, 0.75) for vertex in vertices)


def test_synthetic_glb_without_indices_expands_position_count(tmp_path) -> None:
    path = tmp_path / "indices-implicit.glb"
    _write_synthetic_glb(path, indices=None)
    vertices = load_mesh(path)
    assert len(vertices) == 3


def test_missing_glb_raises_specific_error(tmp_path) -> None:
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(tmp_path / "missing.glb"))
    assert "does not exist" in str(exc.value)


def test_non_glb_path_raises_specific_error(tmp_path) -> None:
    path = tmp_path / "model.gltf"
    path.write_bytes(b"not a glb")
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(path))
    assert "self-contained .glb" in str(exc.value)


def test_invalid_glb_raises_specific_error(tmp_path) -> None:
    path = tmp_path / "bad.glb"
    path.write_bytes(b"not a glb")
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(path))
    assert "could not parse" in str(exc.value)


def test_glb_with_no_meshes_raises_specific_error(tmp_path) -> None:
    path = tmp_path / "no-mesh.glb"
    _write_synthetic_glb(path, mesh_count=0)
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(path))
    assert "no meshes" in str(exc.value)


def test_two_mesh_nodes_are_merged_with_world_transforms(tmp_path) -> None:
    path = tmp_path / "two-meshes.glb"
    nodes = [
        pygltflib.Node(mesh=0, children=[1]),
        pygltflib.Node(mesh=1, translation=[2.0, 0.0, 0.0]),
    ]
    _write_synthetic_glb(path, mesh_count=2, nodes=nodes, scene=[0])

    vertices = load_mesh(path)
    assert len(vertices) == 6
    assert model_vertex_count(vertices) == 2
    # The second mesh sits at x in [2, 4] before normalisation, so the merged
    # box spans x in [-0.85, 0.85] after normalisation.
    assert max(vertex[0] for vertex in vertices) == pytest.approx(0.85)
    assert min(vertex[0] for vertex in vertices) == pytest.approx(-0.85)


def test_transform_point_applies_translation() -> None:
    translation = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0]
    assert _transform_point(translation, 0.0, 0.0, 0.0) == (2.0, 0.0, 0.0)
    assert _transform_point(translation, 1.0, 2.0, 3.0) == (3.0, 2.0, 3.0)


def test_node_rotation_quaternion_rotates_x_to_y() -> None:
    node = pygltflib.Node(rotation=[0.0, 0.0, 0.7071, 0.7071])
    x, y, z = _transform_point(_node_matrix(node), 1.0, 0.0, 0.0)
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(1.0, abs=1e-3)
    assert z == pytest.approx(0.0, abs=1e-3)


def test_child_world_matrix_inherits_parent_translation() -> None:
    parent = pygltflib.Node(translation=[1.0, 0.0, 0.0])
    child = pygltflib.Node(translation=[0.0, 1.0, 0.0])
    world = _mat_mul(_node_matrix(parent), _node_matrix(child))
    assert _transform_point(world, 0.0, 0.0, 0.0) == pytest.approx((1.0, 1.0, 0.0))


def test_glb_without_scenes_loads_every_mesh(tmp_path) -> None:
    path = tmp_path / "no-scene.glb"
    _write_synthetic_glb(path, mesh_count=2)
    vertices = load_mesh(path)
    assert len(vertices) == 6


def test_out_of_range_indices_raises_specific_error(tmp_path) -> None:
    path = tmp_path / "bad-indices.glb"
    _write_synthetic_glb(path, indices=[0, 1, 5])
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(path))
    assert "outside" in str(exc.value)


def test_index_count_not_multiple_of_three_raises_specific_error(tmp_path) -> None:
    path = tmp_path / "bad-index-count.glb"
    _write_synthetic_glb(path, indices=[0, 1])
    with pytest.raises(GltfLoadError) as exc:
        load_mesh(str(path))
    assert "multiple of 3" in str(exc.value)
