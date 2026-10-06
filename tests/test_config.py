from __future__ import annotations

import json
from pathlib import Path

from voxa.config import FACE_MODES, WEB_SEARCH_MODES, ConfigStore, Settings


def test_defaults_when_config_is_missing(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load() == Settings()


def test_round_trip_and_normalization(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings(tts_rate=999, silence_ms=10, ollama_url="http://localhost:11434/")
    store.save(settings)

    loaded = store.load()
    assert loaded.tts_rate == 350
    assert loaded.silence_ms == 300
    assert loaded.ollama_url == "http://localhost:11434"


def test_legacy_keys_are_migrated_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"selected_model": "qwen3:8b", "whisper_model": "small"}), encoding="utf-8")
    loaded = ConfigStore(path).load()
    assert loaded.ollama_model == "qwen3:8b"
    assert loaded.whisper_model == "small"


def test_appearance_and_voice_are_normalized(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings(appearance="sepia", tts_voice="")
    store.save(settings)

    loaded = store.load()
    assert loaded.appearance == "system"
    assert loaded.tts_voice == "en-US-AriaNeural"


def test_character_id_is_normalized_and_unknown_ids_become_classic(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(character_id="Grace"))
    assert store.load().character_id == "grace"

    store.save(Settings(character_id="PORCELAIN-GRACE"))
    assert store.load().character_id == ""

    store.save(Settings(character_id=""))
    assert store.load().character_id == ""


def test_wake_word_falls_back_to_default_when_blank(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(wake_word="   "))

    assert store.load().wake_word == "voxa"


def test_ai_backend_defaults_to_llamacpp(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().ai_backend == "llamacpp"
    assert store.load().llamacpp_url == "http://127.0.0.1:8080"
    assert Settings().ai_backend == "llamacpp"


def test_existing_config_without_ai_backend_keeps_ollama(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ollama_model": "qwen3:8b"}), encoding="utf-8")
    loaded = ConfigStore(path).load()
    assert loaded.ai_backend == "ollama"
    assert loaded.ollama_model == "qwen3:8b"


def test_ai_backend_is_validated_and_llamacpp_url_normalized(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(ai_backend="mistral", llamacpp_url="http://box:8080/"))
    loaded = store.load()
    assert loaded.ai_backend == "llamacpp"
    assert loaded.llamacpp_url == "http://box:8080"
    store.save(Settings(ai_backend="ollama", llamacpp_url="http://box:8080"))
    assert store.load().ai_backend == "ollama"


def test_one_bad_field_keeps_the_other_saved_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"tts_rate": "fast", "ollama_url": "http://box:11434", "whisper_model": "small"}),
        encoding="utf-8",
    )

    loaded = ConfigStore(path).load()
    assert loaded.tts_rate == 180
    assert loaded.ollama_url == "http://box:11434"
    assert loaded.whisper_model == "small"


def test_legacy_config_dir_is_migrated(tmp_path: Path) -> None:
    old_dir = tmp_path / "config" / "voice2text-ai"
    old_dir.mkdir(parents=True)
    (old_dir / "config.json").write_text(json.dumps({"tts_rate": 200}), encoding="utf-8")

    store = ConfigStore(tmp_path / "config" / "voxa" / "config.json")
    loaded = store.load()

    assert loaded.tts_rate == 200
    assert (tmp_path / "config" / "voxa" / "config.json").is_file()


def test_early_silence_ms_defaults_and_clamps(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().early_silence_ms == 300
    assert Settings().early_silence_ms == 300

    store.save(Settings(early_silence_ms=50))
    assert store.load().early_silence_ms == 150
    store.save(Settings(early_silence_ms=5000))
    assert store.load().early_silence_ms == 800
    store.save(Settings(early_silence_ms=400))
    assert store.load().early_silence_ms == 400


def test_early_final_pass_round_trips(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().early_final_pass is False
    assert Settings().early_final_pass is False

    store.save(Settings(early_final_pass=True))
    assert store.load().early_final_pass is True


def test_suggestions_enabled_round_trips(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().suggestions_enabled is True
    assert Settings().suggestions_enabled is True

    store.save(Settings(suggestions_enabled=False))
    assert store.load().suggestions_enabled is False


def test_strata_backend_is_accepted(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert Settings().strata_url == "http://127.0.0.1:8080"

    store.save(Settings(ai_backend="strata", strata_url="http://localhost:9000/"))
    loaded = store.load()
    assert loaded.ai_backend == "strata"
    assert loaded.strata_url == "http://localhost:9000"


def test_unknown_backend_falls_back_to_llamacpp(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(ai_backend="not-real"))
    assert store.load().ai_backend == "llamacpp"


def test_followup_seconds_defaults_and_clamps(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().followup_seconds == 6.0
    assert Settings().followup_seconds == 6.0

    store.save(Settings(followup_seconds=-5.0))
    assert store.load().followup_seconds == 0.0
    store.save(Settings(followup_seconds=99.0))
    assert store.load().followup_seconds == 20.0
    store.save(Settings(followup_seconds=3.5))
    assert store.load().followup_seconds == 3.5


def test_face_mode_defaults_to_prerendered(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().face_mode == "prerendered"
    assert Settings().face_mode == "prerendered"


def test_face_mode_keeps_each_valid_value(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    for mode in FACE_MODES:
        store.save(Settings(face_mode=mode))
        assert store.load().face_mode == mode


def test_face_mode_normalises_invalid_value(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(face_mode="hologram"))
    assert store.load().face_mode == "prerendered"
    assert Settings(face_mode="hologram").normalized().face_mode == "prerendered"


def test_face_mode_round_trips(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(face_mode="live"))
    assert store.load().face_mode == "live"


def test_web_search_defaults_to_auto(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    assert store.load().web_search == "auto"
    assert Settings().web_search == "auto"


def test_web_search_keeps_each_valid_value(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    for mode in WEB_SEARCH_MODES:
        store.save(Settings(web_search=mode))
        assert store.load().web_search == mode


def test_web_search_normalises_invalid_value(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(web_search="sometimes"))
    assert store.load().web_search == "auto"
    assert Settings(web_search="sometimes").normalized().web_search == "auto"


def test_web_search_round_trips(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(web_search="never"))
    assert store.load().web_search == "never"
