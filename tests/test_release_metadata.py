from __future__ import annotations

import re
import struct
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from tools import release_check
from voxa import APP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_match() -> None:
    """One source of truth: APP_VERSION. pyproject derives it; AppStream must agree."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metainfo = ET.parse(ROOT / "io.github.crhy.voxa.metainfo.xml").getroot()
    release = metainfo.find("./releases/release")

    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "voxa.__init__.APP_VERSION"
    }
    assert release is not None
    assert release.attrib["version"] == APP_VERSION
    assert release_check.app_version() == APP_VERSION


def test_release_checker_accepts_the_repository_as_it_is() -> None:
    assert release_check.check() == []
    assert release_check.check(tag=f"v{APP_VERSION}") == []


def test_release_checker_rejects_a_mismatched_tag_and_a_stale_metainfo(tmp_path) -> None:
    assert any("does not match" in problem for problem in release_check.check(tag="v99.0.0"))

    # A copy of the repository files with an out-of-date AppStream release.
    (tmp_path / "voxa").mkdir()
    (tmp_path / "voxa" / "__init__.py").write_text('APP_VERSION = "2.0.0"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "io.github.crhy.voxa.metainfo.xml").write_text(
        (ROOT / "io.github.crhy.voxa.metainfo.xml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert any("AppStream" in problem for problem in release_check.check(tmp_path))


def test_release_checker_flags_a_static_pyproject_version(tmp_path) -> None:
    (tmp_path / "voxa").mkdir()
    (tmp_path / "voxa" / "__init__.py").write_text('APP_VERSION = "0.1.1"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "voxa"\nversion = "0.1.1"\n', encoding="utf-8")
    (tmp_path / "io.github.crhy.voxa.metainfo.xml").write_text(
        (ROOT / "io.github.crhy.voxa.metainfo.xml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert any("static" in problem for problem in release_check.check(tmp_path))


def test_release_artifact_name_carries_the_version_and_architecture() -> None:
    assert release_check.artifact_name("0.6.0") == "Voxa-0.6.0-x86_64.flatpak"
    assert release_check.artifact_name("1.2.3", "aarch64") == "Voxa-1.2.3-aarch64.flatpak"


def test_ci_uses_versioned_bundles_and_the_voxa_repository() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")

    # The bundle name is derived from APP_VERSION (issue #1), never a fixed "Voxa.flatpak".
    assert "Voxa.flatpak" not in workflow
    assert "tools/release_check.py" in workflow
    assert "needs.test.outputs.artifact" in workflow
    # The update URL must not point at another project (issue #7 section 22).
    assert "spacedbazaar" not in workflow
    assert "https://crhy.github.io/voxa/flatpak-repo/" in workflow


def test_ci_validates_metadata_and_runs_the_ui_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    for expected in ("appstreamcli validate", "desktop-file-validate", "xvfb-run", "tests/test_ui_*.py"):
        assert expected in workflow


def test_ci_compiles_and_lints_voxatest() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")

    assert "compileall -q voxa voxatest voxa.py" in workflow
    assert "ruff check voxa voxatest tests voxa.py" in workflow


def test_package_exposes_voxatest_and_includes_its_cases() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["voxatest"] == "voxatest.__main__:main"
    assert project["tool"]["setuptools"]["package-data"]["voxatest"] == ["data/*.json"]


def test_license_metadata_matches_license_file() -> None:
    metainfo = ET.parse(ROOT / "io.github.crhy.voxa.metainfo.xml").getroot()
    assert metainfo.findtext("project_license") == "MIT"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_voxa_declares_itself_the_successor_of_voice2textai() -> None:
    """Flatpak only migrates the old app's data when both files declare the rename."""
    desktop = (ROOT / "io.github.crhy.voxa.desktop").read_text(encoding="utf-8")
    assert "X-Flatpak-RenamedFrom=io.github.crhy.voice2textai.desktop;" in desktop

    metainfo = ET.parse(ROOT / "io.github.crhy.voxa.metainfo.xml").getroot()
    replaced = [element.text for element in metainfo.findall("./replaces/id")]
    assert replaced == ["io.github.crhy.voice2textai"]
    assert metainfo.find("./launchable") is not None
    assert metainfo.find("./replaces") is not None


def test_flatpak_manifest_copies_the_package_recursively() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")

    # A bare voxa/*.py glob would drop subpackages such as voxa/ui/ from the
    # Flatpak payload and crash the app on import.
    assert "voxa/*.py" not in manifest
    assert "cp -r voxa /app/lib/voxa/" in manifest
    assert "__pycache__" in manifest


def test_flatpak_manifest_and_launcher_agree() -> None:
    manifest = (ROOT / "io.github.crhy.voxa.yml").read_text(encoding="utf-8")
    launcher = (ROOT / "packaging/flatpak/voxa").read_text(encoding="utf-8")

    command = re.search(r"^command:\s*(\S+)", manifest, re.MULTILINE)
    assert command is not None
    assert command.group(1) == "voxa"
    assert "/app/lib/voxa/voxa.py" in launcher


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


def test_ci_publishes_the_update_repository_to_pages_on_release_tags() -> None:
    import yaml

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8"))
    pages = workflow["jobs"]["pages"]
    assert pages["if"] == "startsWith(github.ref, 'refs/tags/v')"  # never on ordinary pushes
    assert pages["permissions"] == {"pages": "write", "id-token": "write"}
    text = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    assert "actions/deploy-pages" in text and "site/flatpak-repo" in text and "voxa.flatpakrepo" in text
    assert "https://crhy.github.io/voxa/flatpak-repo/" in text
