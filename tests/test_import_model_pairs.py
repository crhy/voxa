from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from tools.import_model_pairs import (  # noqa: E402
    PAIR_CHARACTERS,
    PORTRAIT_TOP,
    import_pairs,
    split_pair,
)


def test_split_pair_removes_divider_and_preserves_pixels() -> None:
    image = Image.new("RGB", (26, PORTRAIT_TOP + 9), "white")
    image.paste((255, 0, 0), (0, 0, 10, image.height))
    image.paste((0, 0, 255), (16, 0, 26, image.height))

    left, right = split_pair(image)

    assert left.size == (9, 9)
    assert right.size == (9, 9)
    assert left.getpixel((5, 5)) == (255, 0, 0)
    assert right.getpixel((5, 5)) == (0, 0, 255)


def test_split_pair_rejects_unexpected_geometry() -> None:
    with pytest.raises(ValueError, match="6px divider"):
        split_pair(Image.new("RGB", (20, 20)))


def test_import_pairs_requires_the_complete_cast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AmericanCouple.png"):
        import_pairs(tmp_path, tmp_path / "out")


def test_pair_mapping_covers_every_character_once() -> None:
    character_ids = [item for pair in PAIR_CHARACTERS.values() for item in pair]
    assert len(character_ids) == 22
    assert len(set(character_ids)) == 22
