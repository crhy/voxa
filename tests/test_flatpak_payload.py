"""Issue #75, first step: the Flatpak must not ship the debug-symbol payload.

The owner measured 38 MB of ``lib/debug`` in the installed app and named dropping
it as the first step of the debloating plan. The three decisions that plan is
waiting on — CUDA as an add-on, which characters ship in the core, whether the
3D avatars stay — are deliberately left untouched here, and the wheel pins must
not move.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = "io.github.crhy.voxa.yml"
WHEELS = "python3-requirements-flatpak.json"

# Modules in the manifest that install compiled binaries, i.e. the ones whose
# ELF files carry the debug symbols.
COMPILED_MODULES = ("espeak-ng", "nvidia-cuda-runtime", "llama-cpp", "voxa")

# Wheel modules that carry compiled extension code. hf-xet and PyYAML ride with
# the hub; ctranslate2, onnxruntime, numpy and tokenizers ride with faster-whisper.
COMPILED_WHEELS = (
    "python3-huggingface-hub",
    "python3-av",
    "python3-faster-whisper",
    "python3-numpy",
    "python3-PyOpenGL",
)


def _manifest() -> dict:
    return yaml.safe_load((ROOT / MANIFEST).read_text(encoding="utf-8"))


def _manifest_modules() -> dict[str, dict]:
    return {module["name"]: module for module in _manifest()["modules"] if isinstance(module, dict)}


def _wheel_modules() -> list[dict]:
    return json.loads((ROOT / WHEELS).read_text(encoding="utf-8"))["modules"]


def _drops_debug_symbols(module: dict) -> bool:
    """True when the module cleans a debug directory or prunes one itself."""
    cleanup = {str(pattern) for pattern in module.get("cleanup", [])}
    if cleanup & {"/lib/debug", "/share/debug"}:
        return True
    commands = " ".join(str(command) for command in module.get("post-install", []))
    return re.search(r"-type\s+d\s+-name\s+[\"']?debug\b", commands) is not None


def _pip_spec(command: str) -> str:
    match = re.search(r'--prefix=\$\{FLATPAK_DEST\}\s+"([^"]+)"', command)
    assert match is not None, command
    return match.group(1)


def _requirement_lines(path: Path) -> set[str]:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_manifest_and_generated_wheel_file_still_parse() -> None:
    manifest = _manifest()

    assert manifest["id"] == "io.github.crhy.voxa"
    assert manifest["command"] == "voxa"
    assert manifest["runtime-version"] == "50"

    # The generated wheel module is included by file name, so it must stay listed.
    assert WHEELS in _manifest()["modules"]
    assert _wheel_modules()
    assert json.loads((ROOT / WHEELS).read_text(encoding="utf-8"))["name"] == "python3-requirements-flatpak"


def test_compiled_manifest_modules_drop_the_debug_symbol_payload() -> None:
    modules = _manifest_modules()

    for name in COMPILED_MODULES:
        module = modules[name]
        assert _drops_debug_symbols(module), f"{name} still installs debug symbols"
        # cleanup: is the established mechanism in this file; the leading slash
        # matters, a pattern without it only matches a basename and never fires.
        assert "/lib/debug" in module["cleanup"], f"{name} does not clean /lib/debug"
        assert any("debug" in str(command) for command in module.get("post-install", [])), (
            f"{name} has no post-install debug prune"
        )


def test_compiled_wheel_modules_drop_the_debug_symbol_payload() -> None:
    for module in _wheel_modules():
        if module["name"] not in COMPILED_WHEELS:
            continue
        assert "/lib/debug" in module.get("cleanup", []), f"{module['name']} still installs debug symbols"


def test_cuda_still_ships_until_the_owner_decides_otherwise() -> None:
    modules = _manifest_modules()

    cuda = modules["nvidia-cuda-runtime"]
    assert cuda.get("disabled") is not True
    sources = " ".join(str(source["url"]) for source in cuda["sources"])
    assert "libcublas-linux-x86_64-12.6.4.1-archive.tar.xz" in sources
    assert "cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz" in sources


def test_every_character_still_ships_until_the_owner_decides_otherwise() -> None:
    modules = _manifest_modules()

    faces = modules["voxa-faces"]
    assert faces.get("disabled") is not True
    # The whole pack, not a subset of characters: the archive is copied as-is.
    assert faces["sources"][0]["url"].endswith("voxa-faces-2.tar.gz")
    assert "cp -r . /app/share/voxa/faces/" in faces["build-commands"]


def test_the_3d_avatar_library_stays_until_the_owner_decides_otherwise() -> None:
    assert "PyOpenGL>=3.1,<4" in _requirement_lines("requirements.txt")
    assert "PyOpenGL>=3.1,<4" in {
        _pip_spec(command)
        for module in _wheel_modules()
        for command in module.get("build-commands", [])
    }


def test_no_wheel_pin_changed() -> None:
    specs = {
        _pip_spec(command)
        for module in _wheel_modules()
        for command in module.get("build-commands", [])
    }

    # packaging/flatpak/requirements.txt is what the generator was run on.
    assert specs == _requirement_lines("packaging/flatpak/requirements.txt")
    # The application's own requirements must still be covered by those wheels.
    assert _requirement_lines("requirements.txt") <= specs
