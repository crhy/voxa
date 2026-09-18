from __future__ import annotations

import re
import struct
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from voxa import APP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metainfo = ET.parse(ROOT / "io.github.crhy.voxa.metainfo.xml").getroot()
    release = metainfo.find("./releases/release")

    assert project["project"]["version"] == APP_VERSION
    assert release is not None
    assert release.attrib["version"] == APP_VERSION


def test_license_metadata_matches_license_file() -> None:
    metainfo = ET.parse(ROOT / "io.github.crhy.voxa.metainfo.xml").getroot()
    assert metainfo.findtext("project_license") == "MIT"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_flatpak_manifest_and_launcher_agree() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")
    launcher = (ROOT / "packaging/flatpak/voxa").read_text(encoding="utf-8")

    command = re.search(r"^command:\s*(\S+)", manifest, re.MULTILINE)
    assert command is not None
    assert command.group(1) == "voxa"
    assert "/app/lib/voxa/voxa.py" in launcher


def test_release_bundle_name_includes_version_and_arch() -> None:
    workflow = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # voxa#1: release bundles must carry the version and arch so older and
    # newer downloads are distinguishable on disk.
    assert "Voxa-${{ steps.version.outputs.version }}-x86_64.flatpak" in workflow
    assert "Voxa-*-x86_64.flatpak" in workflow
    assert "\n          Voxa.flatpak\n" not in workflow
    assert "Voxa-*-x86_64.flatpak" in readme
    assert "Voxa.flatpak" not in readme.replace("Voxa-*-x86_64.flatpak", "")


def test_cuda_payload_stays_below_ostree_safety_limit() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")

    # CUDA 12.8's libcublasLt shared object exceeds OSTree's hard 512 MiB
    # decompressed-object limit and produces a bundle that current Flatpak
    # cannot import.  CUDA 12.6.4.1 provides the same libcublas.so.12 ABI and
    # its largest installed shared object is 491,106,832 bytes.
    assert "libcublas-linux-x86_64-12.6.4.1-archive.tar.xz" in manifest
    assert "libcublasLt.so.12.6.4.1" in manifest
    assert "libcublas-linux-x86_64-12.8" not in manifest


def test_flatpak_uses_supported_runtime_and_python_wheels() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")
    requirements = (ROOT / "python3-requirements-flatpak.json").read_text(
        encoding="utf-8"
    )

    assert "runtime-version: '50'" in manifest
    assert "cp313" in requirements
    assert "cp312" not in requirements


def test_application_icon_pack_is_complete() -> None:
    master = ROOT / "icons" / "io.github.crhy.voxa-1024.png"
    assert master.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        icon = ROOT / "icons" / "hicolor" / f"{size}x{size}" / "apps" / (
            "io.github.crhy.voxa.png"
        )
        data = icon.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack(">II", data[16:24]) == (size, size)

    exported = ROOT / "icons" / "io.github.crhy.voxa.png"
    assert exported.read_bytes() == (
        ROOT / "icons" / "io.github.crhy.voxa-256.png"
    ).read_bytes()


def test_rhubarb_module_is_pinned_and_installs_to_app_bin() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")

    # Avatar lip-sync (docs/AVATAR.md): upstream release binary, pinned by
    # URL + hash, with its res/ models next to the installed binary.
    assert "rhubarb-lip-sync-1.9.1-linux.zip" in manifest
    assert "f55dc23ac75017b0ce5f1a84a92fbfa89720ff8962972321dcbcf554c80f9594" in manifest
    assert "extracted/rhubarb-lip-sync-1.9.1-linux/rhubarb /app/bin/rhubarb" in manifest
    assert "cp -r extracted/rhubarb-lip-sync-1.9.1-linux/res/sphinx /app/bin/res/sphinx" in manifest
    assert "EsotericSoftwareSpine" not in manifest


def test_flatpak_installs_ui_subpackage() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")

    # voxa/ui/ must ship or the installed app cannot import the new view.
    assert "voxa/ui/*.py /app/lib/voxa/voxa/ui/" in manifest


def test_launcher_sets_no_rejected_gdk_flags() -> None:
    launcher = (ROOT / "packaging" / "flatpak" / "voxa").read_text(encoding="utf-8")

    # GDK_DISABLE=incremental-present is fatal on current GTK (the app
    # exits before showing a window); GSK full-redraw covers the old
    # NVIDIA damage-fragment workaround instead.
    assert "incremental-present" not in launcher.replace(
        "Do NOT set GDK_DISABLE=incremental-present here", ""
    )
