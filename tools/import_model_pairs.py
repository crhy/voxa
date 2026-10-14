#!/usr/bin/env python3
"""Split Voxa's paired character renders into named, square portraits.

The source directory contains one composite PNG per culture.  Each composite
has the female character on the left, the male character on the right, and a
six-pixel divider between two equally-sized 624-pixel panels.  The generated
divider is antialiased into the adjacent column on both panels, so that inner
column is excluded.  The square is also taken lower in the tall source panels
to put the faces in the visual centre instead of anchoring them at the bottom.

Example::

    python tools/import_model_pairs.py SOURCE_DIR OUTPUT_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

PAIR_CHARACTERS: dict[str, tuple[str, str]] = {
    "AmericanCouple.png": ("grace", "jack"),
    "ArabCouple.png": ("layla", "omar"),
    "EnglishCouple.png": ("charlotte", "oliver"),
    "FrenchCouple.png": ("camille", "etienne"),
    "GermanCouple.png": ("greta", "klaus"),
    "IndianCouple.png": ("priya", "arjun"),
    "IrishCouple.png": ("aoife", "seamus"),
    "JapaneseCouple.png": ("sakura", "hiroshi"),
    "MexicanCouple.png": ("valentina", "juan"),
    "NativeCouple.png": ("aiyana", "koda"),
    "NigerianCouple.png": ("amara", "chidi"),
}

DIVIDER_WIDTH = 6
INNER_EDGE_TRIM = 1
PORTRAIT_TOP = 112


def split_pair(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return centred square portraits without the divider's white fringe."""
    width, height = image.size
    panel_width = (width - DIVIDER_WIDTH) // 2
    portrait_size = panel_width - INNER_EDGE_TRIM
    right_start = panel_width + DIVIDER_WIDTH + INNER_EDGE_TRIM
    if (
        width != 2 * panel_width + DIVIDER_WIDTH
        or height < PORTRAIT_TOP + portrait_size
    ):
        raise ValueError(
            f"expected two portrait panels and a {DIVIDER_WIDTH}px divider, got {width}x{height}"
        )
    return (
        image.crop((0, PORTRAIT_TOP, portrait_size, PORTRAIT_TOP + portrait_size)),
        image.crop(
            (right_start, PORTRAIT_TOP, right_start + portrait_size, PORTRAIT_TOP + portrait_size)
        ),
    )


def import_pairs(source_dir: Path, output_dir: Path) -> list[Path]:
    """Import every expected pair into ``output_dir`` as lossless RGB PNGs."""
    missing = [name for name in PAIR_CHARACTERS if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source pairs: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, character_ids in PAIR_CHARACTERS.items():
        with Image.open(source_dir / filename) as source:
            left, right = split_pair(source.convert("RGB"))
        for character_id, portrait in zip(character_ids, (left, right), strict=True):
            destination = output_dir / f"{character_id}.png"
            portrait.save(destination, format="PNG", optimize=True)
            written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    written = import_pairs(args.source_dir, args.output_dir)
    print(f"Imported {len(written)} portraits into {args.output_dir}")


if __name__ == "__main__":
    main()
