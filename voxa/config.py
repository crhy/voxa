from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from voxa.ui.avatars import get_avatar

FACE_MODES = ("live", "prerendered", "still")
WEB_SEARCH_MODES = ("auto", "always", "never")
ECHO_MODES = ("voxa", "system", "off")


@dataclass(slots=True)
class Settings:
    microphone_id: str = ""
    microphone_name: str = ""
    whisper_model: str = "base"
    ollama_model: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ai_backend: str = "llamacpp"
    llamacpp_url: str = "http://127.0.0.1:8080"
    llamacpp_model: str = ""
    strata_url: str = "http://127.0.0.1:8080"
    language: str = "en"
    character_id: str = ""
    face_mode: str = "prerendered"
    # Location override set by saying "I'm in X"; empty means detect it from the web.
    home_location: str = ""
    # Subtract the computer's own sound (Voxa's voice, music) from the microphone.
    echo_cancel: bool = True
    # How to remove the computer's own sound: "voxa" (our canceller), "system" (PulseAudio module-echo-cancel),
    # "off". ``echo_cancel`` is a deprecated alias: False maps to "off".
    # "system" is the default: PulseAudio's WebRTC canceller was measured removing 17 dB on real hardware
    # (down to the room noise). Voxa's own canceller ("voxa") is experimental until it is proven in the app.
    echo_mode: str = "system"
    # Milliseconds the lips run ahead of the reported audio position (compensates sound-card and display delay).
    lip_sync_lead_ms: int = 160
    tts_rate: int = 180
    tts_voice: str = "en-US-AriaNeural"
    appearance: str = "system"
    auto_speak: bool = True
    web_search: str = "auto"
    silence_ms: int = 700
    voice_threshold: int = 450
    # The longest single request Voxa will listen to. 6 s (the old default) cut people off mid-sentence.
    max_segment_seconds: float = 20.0
    wake_word: str = "voxa"
    early_silence_ms: int = 300
    early_final_pass: bool = False
    suggestions_enabled: bool = True
    followup_seconds: float = 6.0
    home_assistant_url: str = ""
    home_assistant_token: str = ""

    def normalized(self) -> Settings:
        self.tts_rate = max(80, min(350, int(self.tts_rate)))
        if self.appearance not in {"system", "light", "dark"}:
            self.appearance = "system"
        self.tts_voice = (self.tts_voice or "en-US-AriaNeural").strip()
        self.silence_ms = max(300, min(4000, int(self.silence_ms)))
        self.early_silence_ms = max(150, min(800, int(self.early_silence_ms)))
        self.voice_threshold = max(50, min(5000, int(self.voice_threshold)))
        if float(self.max_segment_seconds) == 6.0:
            self.max_segment_seconds = 20.0  # the old default, saved in existing profiles
        self.max_segment_seconds = max(2.0, min(30.0, float(self.max_segment_seconds)))
        self.followup_seconds = max(0.0, min(20.0, float(self.followup_seconds)))
        self.ollama_url = self.ollama_url.rstrip("/") or "http://127.0.0.1:11434"
        self.llamacpp_url = self.llamacpp_url.rstrip("/") or "http://127.0.0.1:8080"
        self.strata_url = self.strata_url.rstrip("/") or "http://127.0.0.1:8080"
        if self.ai_backend not in {"llamacpp", "ollama", "strata"}:
            self.ai_backend = "llamacpp"
        self.language = (self.language or "en").strip()[:16]
        self.wake_word = (self.wake_word or "voxa").strip()[:32] or "voxa"
        self.character_id = (self.character_id or "").strip().lower()
        if self.character_id and get_avatar(self.character_id) is None:
            self.character_id = ""
        if self.face_mode not in FACE_MODES:
            self.face_mode = "prerendered"
        if self.web_search not in WEB_SEARCH_MODES:
            self.web_search = "auto"
        if self.echo_mode not in ECHO_MODES:
            self.echo_mode = "system"
        if not self.echo_cancel:
            self.echo_mode = "off"
        return self


class ConfigStore:
    """Small JSON settings store using the XDG configuration directory."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            path = base / "voxa" / "config.json"
        self.path = path
        self.legacy_path = Path.home() / ".voice_config.json"

    def load(self, *, migrate: bool = True) -> Settings:
        if migrate:
            self._migrate_legacy_file()
            self._migrate_legacy_config_dir()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return Settings()
        if not isinstance(payload, dict):
            return Settings()

        accepted = {field.name for field in fields(Settings)}
        clean: dict[str, Any] = {key: value for key, value in payload.items() if key in accepted}

        # Compatibility with the pre-0.4 configuration keys.
        if "selected_model" in payload and "ollama_model" not in clean:
            clean["ollama_model"] = payload["selected_model"]
        if "avatar" in payload and "character_id" not in clean:
            clean["character_id"] = payload["avatar"]
        if "microphone_name" in payload:
            clean["microphone_name"] = payload["microphone_name"]
        # Configs written before the llama.cpp backend existed have no
        # ``ai_backend`` key; those users stay on Ollama, today's behaviour.
        if "ai_backend" not in clean:
            clean["ai_backend"] = "ollama"

        # Apply fields one at a time so a single invalid value (e.g.
        # "tts_rate": "fast") only resets that field to its default instead
        # of discarding every saved setting.
        settings = Settings()
        for name, value in clean.items():
            try:
                candidate = Settings(**asdict(settings))
                setattr(candidate, name, value)
                settings = candidate.normalized()
            except (TypeError, ValueError, AttributeError):
                continue
        return settings

    def save(self, settings: Settings) -> None:
        settings.normalized()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def _migrate_legacy_file(self) -> None:
        if self.path.exists() or not self.legacy_path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.legacy_path, self.path)
        except OSError:
            # Migration is best-effort; load() will fall back to defaults.
            return

    def _migrate_legacy_config_dir(self) -> None:
        """Move settings from the pre-0.6 ``.config/voice2text-ai`` directory."""
        if self.path.exists() or self.path.parent.name != "voxa":
            return
        old_path = self.path.parent.parent / "voice2text-ai" / "config.json"
        if not old_path.exists() or old_path == self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, self.path)
        except OSError:
            # Migration is best-effort; load() will fall back to defaults.
            return
