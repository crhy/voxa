"""Issue #23: the panel/menu icon is a two-tone symbolic SVG, the colour icon stays for the grid.

Same style of guard as tests/test_release_metadata.py: the files in the checkout are
checked against the contract the issue sets, so a later edit that hard-codes a colour
or drops the install line fails here instead of on someone's desktop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "icons" / "io.github.crhy.voxa-symbolic.svg"
MANIFEST = ROOT / "io.github.crhy.voxa.yml"
DESKTOP = ROOT / "io.github.crhy.voxa.desktop"

PAINT_ATTRIBUTES = ("fill", "stroke", "stop-color", "flood-color", "lighting-color")
FORBIDDEN_TAGS = ("image", "filter", "linearGradient", "radialGradient", "pattern", "mesh", "raster")


def _elements() -> list[ET.Element]:
    return list(ET.parse(SVG).iter())


def test_symbolic_icon_exists_and_parses() -> None:
    assert SVG.is_file()
    root = ET.parse(SVG).getroot()
    assert root.tag.endswith("svg")


def test_symbolic_icon_uses_a_square_viewbox() -> None:
    view_box = ET.parse(SVG).getroot().get("viewBox")
    assert view_box is not None
    parts = [float(value) for value in view_box.replace(",", " ").split()]
    assert len(parts) == 4
    assert parts[2] == parts[3] > 0


def test_symbolic_icon_is_painted_only_with_currentcolor() -> None:
    """Spaced apps take colours from the desktop theme, so nothing may be hard-coded."""
    paints: set[str] = set()
    for element in _elements():
        for attribute in PAINT_ATTRIBUTES:
            value = element.get(attribute)
            if value is not None:
                paints.add(value.strip())

    assert "currentColor" in paints
    assert len(paints) <= 2
    assert paints <= {"currentColor", "none"}

    text = SVG.read_text(encoding="utf-8")
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", text)
    assert "rgb(" not in text


def test_symbolic_icon_carries_no_raster_or_effects() -> None:
    tags = {element.tag.split("}")[-1] for element in _elements()}
    assert tags.isdisjoint(FORBIDDEN_TAGS)

    text = SVG.read_text(encoding="utf-8")
    assert "base64" not in text
    assert "data:image" not in text
    assert "<filter" not in text and "url(#" not in text


def test_the_mark_stays_inside_the_icon_box() -> None:
    """The V and its ring must not be clipped: every coordinate sits in the viewBox."""
    root = ET.parse(SVG).getroot()
    size = float(root.get("viewBox").split()[2])

    numbers: list[float] = []
    for element in _elements():
        for attribute in ("cx", "cy", "r", "x", "y", "width", "height", "stroke-width"):
            value = element.get(attribute)
            if value is not None:
                numbers.append(float(value))
        path = element.get("d")
        if path is not None:
            numbers.extend(float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path))

    assert numbers
    assert all(0.0 <= value <= size for value in numbers)


def test_flatpak_manifest_installs_the_symbolic_icon() -> None:
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "path: icons" in manifest and "dest: icons" in manifest
    assert (
        "install -Dm644 icons/io.github.crhy.voxa-symbolic.svg "
        "/app/share/icons/hicolor/scalable/symbolic/apps/io.github.crhy.voxa-symbolic.svg"
    ) in manifest


def test_flatpak_manifest_still_installs_the_colour_icon_unchanged() -> None:
    """The app grid keeps the full-colour medallion; only menus and panels change."""
    manifest = MANIFEST.read_text(encoding="utf-8")

    for target in (
        "/app/share/icons/hicolor/256x256/apps/io.github.crhy.voxa.png",
        "/app/share/icons/hicolor/256x256/apps/voxa.png",
    ):
        assert f"install -Dm644 icons/io.github.crhy.voxa.png {target}" in manifest


def test_desktop_entry_keeps_the_colour_icon_for_the_app_grid() -> None:
    desktop = DESKTOP.read_text(encoding="utf-8")

    assert re.search(r"^Icon=voxa$", desktop, re.MULTILINE)
    assert "symbolic" not in desktop
    # The rename marker from voice2textai must survive (issue #7 section 22).
    assert "X-Flatpak-RenamedFrom=io.github.crhy.voice2textai.desktop;" in desktop


def test_desktop_entry_still_validates() -> None:
    validator = shutil.which("desktop-file-validate")
    if validator is None:
        return

    result = subprocess.run([validator, str(DESKTOP)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
