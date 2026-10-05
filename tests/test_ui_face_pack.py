from __future__ import annotations

import json
from pathlib import Path

from voxa.ui.face_pack import VISEME_SHAPES, blink_patch, load_pack, mouth_target, pick_frame


def _write_pack(tmp_path: Path, character_id: str = "fake") -> Path:
    pack_dir = tmp_path / character_id
    pack_dir.mkdir(parents=True)
    data = {
        "id": character_id,
        "size": 512,
        "fps": 25,
        "eye_box": [10, 20, 30, 40],
        "eye_open": 0.05,
        "mouth": [
            {"file": "m000.jpg", "open": 0.0, "width": 0.5},
            {"file": "m001.jpg", "open": 0.5, "width": 0.5},
            {"file": "m002.jpg", "open": 1.0, "width": 0.6},
        ],
        "blink": ["b0.jpg", "b1.jpg", "b2.jpg"],
    }
    (pack_dir / "index.json").write_text(json.dumps(data))
    return tmp_path


def test_load_pack_valid(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert pack is not None
    assert pack.id == "fake"
    assert pack.size == 512
    assert pack.eye_box == (10, 20, 30, 40)
    assert len(pack.mouth) == 3
    assert pack.blink == ("b0.jpg", "b1.jpg", "b2.jpg")


def test_load_pack_missing_returns_none(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    assert load_pack("nope", root) is None


def test_load_pack_corrupt_json_returns_none(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    (root / "fake" / "index.json").write_text("{not json")
    assert load_pack("fake", root) is None


def test_load_pack_missing_keys_returns_none(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    (root / "fake" / "index.json").write_text(json.dumps({"id": "fake"}))
    assert load_pack("fake", root) is None


def test_mouth_target_rest_when_empty() -> None:
    assert mouth_target({}) == (0.0, 0.5)


def test_mouth_target_single_viseme_matches_table() -> None:
    assert mouth_target({"viseme_aa": 1.0}) == VISEME_SHAPES["viseme_aa"]


def test_mouth_target_weighted_average() -> None:
    got = mouth_target({"viseme_sil": 1.0, "viseme_aa": 1.0})
    sil = VISEME_SHAPES["viseme_sil"]
    aa = VISEME_SHAPES["viseme_aa"]
    expected = ((sil[0] + aa[0]) / 2.0, (sil[1] + aa[1]) / 2.0)
    assert got == expected


def test_pick_frame_nearest(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert pick_frame(pack, (1.0, 0.6)) == 2
    assert pick_frame(pack, (0.0, 0.5)) == 0


def test_pick_frame_open_weighted_twice(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    # target open 0.45 width 0.5: frame 1 (open 0.5) is nearer than frame 0
    assert pick_frame(pack, (0.45, 0.5)) == 1


def test_pick_frame_stickiness_keeps_previous(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    # best for (0.5,0.5) is frame 1; from previous 0 the gain is small
    assert pick_frame(pack, (0.5, 0.5), previous=0, stickiness=0.03) == 1


def test_pick_frame_stickiness_blocks_tiny_switch(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    # frame 1 is only marginally nearer than frame 0 -> stickiness keeps 0
    assert pick_frame(pack, (0.26, 0.5), previous=0, stickiness=0.03) == 0


def test_pick_frame_large_change_switches(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert pick_frame(pack, (1.0, 0.6), previous=0, stickiness=0.03) == 2


def test_blink_patch_open_is_none(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert blink_patch(pack, 0.0) is None
    assert blink_patch(pack, 0.14) is None


def test_blink_patch_nearly_open(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert blink_patch(pack, 0.2) == "b2.jpg"


def test_blink_patch_half(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert blink_patch(pack, 0.5) == "b1.jpg"


def test_blink_patch_closed(tmp_path: Path) -> None:
    root = _write_pack(tmp_path)
    pack = load_pack("fake", root)
    assert blink_patch(pack, 0.9) == "b0.jpg"
    assert blink_patch(pack, 1.0) == "b0.jpg"
