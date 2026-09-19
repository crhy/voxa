"""Every configuration shape Voxa (or Voice2Text AI before it) has shipped must keep loading.

Issue #7 section 23: a user upgrading must never have to delete their settings. Each fixture
below is the JSON a released version wrote; loading it must keep every saved value, leave users
of older versions on the Ollama backend they were using, and survive a save/load round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxa.config import ConfigStore

# What each released version actually wrote (fields as of that release).
RELEASED_SCHEMAS: dict[str, dict] = {
    # The first dictation app: only two keys.
    "voice2text-pre-0.4": {"selected_model": "qwen2.5:0.5b", "microphone_name": "Blue Yeti"},
    # Voice2Text AI 0.5.x, the native GTK rebuild, after conversation mode was added.
    "voice2text-0.5": {
        "microphone_id": "alsa_input.usb-Blue_Yeti",
        "microphone_name": "Blue Yeti",
        "whisper_model": "small",
        "ollama_model": "llama3.1:8b",
        "ollama_url": "http://127.0.0.1:11434",
        "language": "en",
        "tts_rate": 170,
        "tts_voice": "en-GB-SoniaNeural",
        "appearance": "dark",
        "auto_speak": False,
        "silence_ms": 800,
        "voice_threshold": 500,
        "max_segment_seconds": 8.0,
    },
    # Voxa 0.1.0 and 0.1.1 (the wake word is configurable).
    "voxa-0.1.1": {
        "microphone_id": "alsa_input.pci-0000_00_1f.3",
        "microphone_name": "Built-in Audio",
        "whisper_model": "medium",
        "ollama_model": "qwen3:14b",
        "ollama_url": "http://192.168.1.20:11434",
        "language": "en",
        "tts_rate": 200,
        "tts_voice": "en-US-AriaNeural",
        "appearance": "light",
        "auto_speak": True,
        "silence_ms": 1000,
        "voice_threshold": 420,
        "max_segment_seconds": 7.5,
        "wake_word": "computer",
    },
}


def _store(tmp_path: Path, payload: dict) -> ConfigStore:
    path = tmp_path / "voxa" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ConfigStore(path)


@pytest.mark.parametrize("name", list(RELEASED_SCHEMAS))
def test_released_configs_keep_every_saved_value(tmp_path: Path, name: str) -> None:
    payload = RELEASED_SCHEMAS[name]
    loaded = _store(tmp_path, payload).load()

    expected = dict(payload)
    if "selected_model" in expected:  # renamed in 0.4
        expected["ollama_model"] = expected.pop("selected_model")
    for key, value in expected.items():
        assert getattr(loaded, key) == value, f"{name}: {key} was not kept"


@pytest.mark.parametrize("name", list(RELEASED_SCHEMAS))
def test_users_of_older_versions_stay_on_ollama(tmp_path: Path, name: str) -> None:
    loaded = _store(tmp_path, RELEASED_SCHEMAS[name]).load()
    assert loaded.ai_backend == "ollama"  # the backend they were using; llama.cpp is opt-in for them
    assert loaded.llamacpp_url == "http://127.0.0.1:8080"


@pytest.mark.parametrize("name", list(RELEASED_SCHEMAS))
def test_released_configs_survive_a_save_and_reload(tmp_path: Path, name: str) -> None:
    store = _store(tmp_path, RELEASED_SCHEMAS[name])
    first = store.load()
    store.save(first)
    assert store.load() == first


def test_missing_fields_fall_back_to_defaults_not_to_a_reset(tmp_path: Path) -> None:
    loaded = _store(tmp_path, {"ollama_model": "llama3", "wake_word": "jarvis"}).load()
    assert loaded.ollama_model == "llama3" and loaded.wake_word == "jarvis"
    assert loaded.whisper_model == "base" and loaded.tts_voice == "en-US-AriaNeural"


def test_a_config_from_a_newer_version_loads_and_ignores_unknown_keys(tmp_path: Path) -> None:
    payload = {**RELEASED_SCHEMAS["voxa-0.1.1"], "some_future_setting": {"nested": True}, "avatar": "porcelain-grace"}
    loaded = _store(tmp_path, payload).load()
    assert loaded.wake_word == "computer" and loaded.ollama_model == "qwen3:14b"
    assert not hasattr(loaded, "some_future_setting")


def test_a_new_install_defaults_to_llamacpp(tmp_path: Path) -> None:
    path = tmp_path / "voxa" / "config.json"
    assert ConfigStore(path).load().ai_backend == "llamacpp"
