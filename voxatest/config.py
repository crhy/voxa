"""Harness settings.

The grading model defaults to the llama.cpp server known to be running on this
machine (qwen38-flash-next at 127.0.0.1:8080). Voxa's own config file is read
best-effort so the same backend/model never has to be configured twice; the
VOXATEST_BACKEND, VOXATEST_MODEL and VOXATEST_URL environment variables always
win.

Device notes:
* ``input_device`` is a microphone identifier from ``voxa.audio.AudioCapture``
  (empty string means the system default).
* ``output_device`` is a GStreamer sink element name (e.g. ``pipewiresink`` or
  ``alsasink``); it is exported as ``GST_AUDIOSINK`` so Voxa's own
  ``SpeechService`` routes through it. An empty string keeps the default sink.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BACKEND = "llamacpp"
DEFAULT_MODEL = "qwen38-flash-next"
DEFAULT_URL = "http://127.0.0.1:8080"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_DIR / "data" / "test_cases.json"
DEFAULT_REPORTS_DIR = PACKAGE_DIR / "reports"


@dataclass(slots=True)
class HarnessSettings:
    backend: str = DEFAULT_BACKEND
    model: str = DEFAULT_MODEL
    url: str = DEFAULT_URL
    input_device: str = ""
    output_device: str = ""
    wake_word: str = "voxa"
    whisper_model: str = "base"
    language: str = "en"
    tts_rate: int = 180
    tts_voice: str = "en-US-AriaNeural"
    voice_threshold: int = 450
    silence_ms: int = 900
    max_segment_seconds: float = 12.0
    reply_wait_seconds: float = 20.0
    reasoning_effort: str | None = "none"
    end_silence_seconds: float = 3.0
    whisper_load_timeout_seconds: float = 120.0
    speak_timeout_seconds: float = 60.0
    run_deadline_seconds: float = 180.0
    data_path: Path = DEFAULT_DATA_PATH
    reports_dir: Path = DEFAULT_REPORTS_DIR

    def make_client(self) -> Any:
        """Return the configured local-model client (llama.cpp or Ollama)."""
        if self.backend == "ollama":
            from voxa.ollama import OllamaClient

            return OllamaClient(base_url=self.url)
        from voxa.llamacpp import LlamaCppClient

        return LlamaCppClient(base_url=self.url, reasoning_effort=self.reasoning_effort)


def load_settings() -> HarnessSettings:
    settings = HarnessSettings()

    try:
        from voxa.config import ConfigStore

        voxa_settings = ConfigStore().load()
    except Exception:  # noqa: BLE001 - reading Voxa's config is best-effort
        voxa_settings = None

    settings.backend = os.environ.get("VOXATEST_BACKEND", DEFAULT_BACKEND)

    if voxa_settings is not None:
        settings.model = os.environ.get("VOXATEST_MODEL", voxa_settings.ollama_model or DEFAULT_MODEL)
        settings.input_device = voxa_settings.microphone_id
        settings.wake_word = voxa_settings.wake_word or "voxa"
        settings.whisper_model = voxa_settings.whisper_model or "base"
        settings.language = voxa_settings.language or "en"
        settings.tts_rate = voxa_settings.tts_rate
        settings.tts_voice = voxa_settings.tts_voice
        settings.voice_threshold = voxa_settings.voice_threshold
        settings.silence_ms = voxa_settings.silence_ms

        if settings.backend == "ollama":
            settings.url = voxa_settings.ollama_url or DEFAULT_URL
        elif settings.backend == "llamacpp":
            settings.url = voxa_settings.llamacpp_url or DEFAULT_URL

    settings.model = os.environ.get("VOXATEST_MODEL", settings.model)
    settings.url = os.environ.get("VOXATEST_URL", settings.url)
    settings.wake_word = os.environ.get("VOXATEST_WAKE_WORD", settings.wake_word)
    settings.reasoning_effort = os.environ.get("VOXATEST_REASONING_EFFORT", settings.reasoning_effort)
    if not settings.reasoning_effort:
        settings.reasoning_effort = None
    if settings.backend not in {"llamacpp", "ollama"}:
        settings.backend = DEFAULT_BACKEND
    return settings
