"""Installed Ollama models must be listed from disk with the server stopped (crhy/voxa#27).

Ollama keeps one JSON manifest per tag under ``<models dir>/manifests/<registry>/<library>/<name>/<tag>``
(the default ``registry.ollama.ai`` and ``library`` are not part of the displayed name). These fixtures build
that tree in a temp folder: no network, no Ollama install, no server, and the disk reader must never raise
whatever it finds.
"""

from __future__ import annotations

import os

import pytest

from voxa.backup import BackupPaths, _parse_ollama_models
from voxa.modelwait import models_to_show
from voxa.ollama import disk_models, ollama_models_dir


def _manifest(models_dir, *segments: str) -> None:
    path = models_dir.joinpath("manifests", *segments)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schemaVersion": 1}', encoding="utf-8")


def test_library_family_shows_as_name_and_tag(tmp_path) -> None:
    _manifest(tmp_path, "library", "qwen2.5", "0.5b")
    assert disk_models(tmp_path) == ["qwen2.5:0.5b"]


def test_default_registry_prefix_is_not_part_of_the_name(tmp_path) -> None:
    # The tree Ollama 0.32 actually writes: <registry>/<library>/<name>/<tag>.
    _manifest(tmp_path, "registry.ollama.ai", "library", "qwen2.5", "0.5b")
    assert disk_models(tmp_path) == ["qwen2.5:0.5b"]


def test_other_family_shows_as_family_name_and_tag(tmp_path) -> None:
    _manifest(tmp_path, "ghcr.io", "me/models", "latest")
    assert disk_models(tmp_path) == ["ghcr.io/me/models:latest"]


def test_a_hugging_face_gguf_keeps_its_full_name(tmp_path) -> None:
    # Same manifest path and expected name as tests/test_backup.py asserts for
    # the backup inventory, so the picker and the backup agree on one name.
    _manifest(tmp_path, "hf.co", "ISTA-DASLab", "Qwen3.8-27B-GSQ-RCO-GGUF", "IQ3_S")
    assert disk_models(tmp_path) == ["hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S"]


def test_missing_store_is_no_models(tmp_path, monkeypatch) -> None:
    assert disk_models(tmp_path / "nowhere") == []
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert disk_models() == []


def test_blobs_and_stray_files_are_ignored(tmp_path) -> None:
    _manifest(tmp_path, "library", "qwen2.5", "0.5b")
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "sha256-abc").write_text("binary", encoding="utf-8")
    (tmp_path / "manifests" / "loose.txt").write_text("stray", encoding="utf-8")
    (tmp_path / "manifests" / "library").mkdir(exist_ok=True)
    (tmp_path / "manifests" / "library" / "loose.txt").write_text("stray", encoding="utf-8")
    (tmp_path / "manifests" / "library" / "qwen2.5" / ".0.5b.tmp").write_text("partial", encoding="utf-8")
    assert disk_models(tmp_path) == ["qwen2.5:0.5b"]


def test_names_are_sorted_and_de_duped(tmp_path) -> None:
    _manifest(tmp_path, "library", "llama3", "latest")
    _manifest(tmp_path, "library", "qwen2.5", "32b")
    _manifest(tmp_path, "library", "qwen2.5", "0.5b")
    _manifest(tmp_path, "library", "deep/nested", "latest")
    _manifest(tmp_path, "library", "qwen2.5", "0.5b")  # written twice, listed once
    names = disk_models(tmp_path)
    assert names == sorted(names)
    assert names == sorted(set(names))
    assert names == ["deep/nested:latest", "llama3:latest", "qwen2.5:0.5b", "qwen2.5:32b"]


def test_unreadable_store_is_no_models(tmp_path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads anything; the permission case cannot be forced")
    _manifest(tmp_path, "library", "qwen2.5", "0.5b")
    (tmp_path / "manifests").chmod(0o000)
    try:
        assert disk_models(tmp_path) == []
    finally:
        (tmp_path / "manifests").chmod(0o755)


def test_manifests_being_a_file_is_no_models(tmp_path) -> None:
    (tmp_path / "manifests").write_text("not a folder", encoding="utf-8")
    assert disk_models(tmp_path) == []


def test_models_dir_follows_ollamas_own_rule(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "elsewhere"))
    assert ollama_models_dir() == tmp_path / "elsewhere"
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ollama_models_dir() == tmp_path / ".ollama" / "models"
    assert not (tmp_path / ".ollama").exists()  # listing must never create folders


def test_the_picker_and_the_backup_name_models_alike(tmp_path) -> None:
    # voxa/backup.py inventories the same tree (issue #26). Both must print the
    # name ``ollama list`` prints, or the picker would offer an unusable name.
    store = tmp_path / ".ollama" / "models"
    _manifest(store, "registry.ollama.ai", "library", "qwen2.5", "0.5b")
    _manifest(store, "hf.co", "ISTA-DASLab", "Qwen3.8-27B-GSQ-RCO-GGUF", "IQ3_S")
    infos = _parse_ollama_models(BackupPaths(tmp_path), hash_blobs=False)
    assert sorted(info.name for info in infos) == disk_models(store)
    assert disk_models(store) == [
        "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S",
        "qwen2.5:0.5b",
    ]


def test_a_live_list_replaces_the_disk_one() -> None:
    # The picker seeds from disk, then the server's own list wins when it answers.
    assert models_to_show(["qwen2.5:0.5b"], ["qwen2.5:0.5b", "llama3"]) == ["qwen2.5:0.5b", "llama3"]
    assert models_to_show(["qwen2.5:0.5b"], []) == []  # up and truly empty: say so


def test_the_disk_list_survives_an_offline_server() -> None:
    assert models_to_show(["qwen2.5:0.5b"], None) == ["qwen2.5:0.5b"]


def test_nothing_on_disk_either_still_says_so() -> None:
    assert models_to_show([], None) is None
