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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_BACKEND = "llamacpp"
DEFAULT_MODEL = "qwen38-flash-next"
DEFAULT_URL = "http://127.0.0.1:8080"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_DIR / "data" / "test_cases.json"


_Number = TypeVar("_Number", int, float)


def _env_number(name: str, default: _Number, cast: Callable[[str], _Number]) -> _Number:
    """Return an env override as a number, falling back to ``default`` when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def _in_flatpak() -> bool:
    return bool(os.environ.get("FLATPAK_ID"))


def voxa_config_candidates() -> list[Path]:
    """Return possible locations of Voxa's config.json, most specific first, without duplicates."""
    candidates: list[Path] = []
    override = os.environ.get("VOXATEST_VOXA_CONFIG")
    if override:
        candidates.append(Path(override).expanduser())
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        candidates.append(Path(xdg_config).expanduser() / "voxa" / "config.json")
    candidates.append(Path.home() / ".config" / "voxa" / "config.json")
    candidates.append(Path.home() / ".var" / "app" / "io.github.crhy.voxa" / "config" / "voxa" / "config.json")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def choose_voxa_config(candidates: list[Path]) -> Path | None:
    """Return the config Voxa actually uses: the explicit override, else the most recently modified file."""
    override = os.environ.get("VOXATEST_VOXA_CONFIG")
    if override:
        # An explicit choice is never second-guessed, even when the file is missing.
        path = Path(override).expanduser()
        return path if path.is_file() else None

    newest: Path | None = None
    newest_mtime = 0.0
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest_mtime:
            newest, newest_mtime = candidate, mtime
    return newest


def default_reports_dir() -> Path:
    """Return a writable, VoxaTest-specific reports directory."""
    override = os.environ.get("VOXATEST_REPORTS_DIR")
    if override:
        return Path(override).expanduser()
    if _in_flatpak():
        # The Flatpak sandbox only grants write access to xdg-documents.
        return Path.home() / "Documents" / "VoxaTest" / "reports"
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return base / "voxatest" / "reports"


def default_failure_report_path() -> Path:
    """Return the stable Markdown report intended for source control."""
    override = os.environ.get("VOXATEST_FAILURE_REPORT")
    if override:
        return Path(override).expanduser()
    if _in_flatpak():
        return Path.home() / "Documents" / "VoxaTest" / "VOXATEST_FAILURES.md"
    return Path("VOXATEST_FAILURES.md")


DEFAULT_REPORTS_DIR = default_reports_dir()
DEFAULT_FAILURE_REPORT_PATH = default_failure_report_path()


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
    run_deadline_seconds: float = 0.0
    max_reply_seconds: float = 90.0
    settle_seconds: float = 1.5
    retries: int = 1
    max_consecutive_no_reply: int = 5
    data_path: Path = DEFAULT_DATA_PATH
    reports_dir: Path = field(default_factory=default_reports_dir)
    failure_report_path: Path = field(default_factory=default_failure_report_path)
    voxa_config_path: Path | None = None

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

        config_path = choose_voxa_config(voxa_config_candidates())
        if config_path is None:
            voxa_settings = None
        else:
            settings.voxa_config_path = config_path
            voxa_settings = ConfigStore(config_path).load(migrate=False)
    except Exception:  # noqa: BLE001 - reading Voxa's config is best-effort
        voxa_settings = None

    backend_override = os.environ.get("VOXATEST_BACKEND")
    settings.backend = backend_override or (
        voxa_settings.ai_backend if voxa_settings is not None else DEFAULT_BACKEND
    )

    if voxa_settings is not None:
        settings.input_device = voxa_settings.microphone_id
        settings.wake_word = voxa_settings.wake_word or "voxa"
        settings.whisper_model = voxa_settings.whisper_model or "base"
        settings.language = voxa_settings.language or "en"
        settings.tts_rate = voxa_settings.tts_rate
        settings.tts_voice = voxa_settings.tts_voice
        settings.voice_threshold = voxa_settings.voice_threshold
        settings.silence_ms = voxa_settings.silence_ms

        if settings.backend == "ollama":
            settings.model = voxa_settings.ollama_model or DEFAULT_MODEL
            settings.url = voxa_settings.ollama_url or DEFAULT_URL
        elif settings.backend == "llamacpp":
            settings.model = voxa_settings.llamacpp_model or DEFAULT_MODEL
            settings.url = voxa_settings.llamacpp_url or DEFAULT_URL

    settings.model = os.environ.get("VOXATEST_MODEL", settings.model)
    settings.url = os.environ.get("VOXATEST_URL", settings.url)
    settings.wake_word = os.environ.get("VOXATEST_WAKE_WORD", settings.wake_word)
    settings.reasoning_effort = os.environ.get("VOXATEST_REASONING_EFFORT", settings.reasoning_effort)
    settings.run_deadline_seconds = _env_number("VOXATEST_RUN_DEADLINE", settings.run_deadline_seconds, float)
    settings.end_silence_seconds = _env_number("VOXATEST_END_SILENCE", settings.end_silence_seconds, float)
    settings.reply_wait_seconds = _env_number("VOXATEST_REPLY_WAIT", settings.reply_wait_seconds, float)
    settings.retries = _env_number("VOXATEST_RETRIES", settings.retries, int)
    settings.max_consecutive_no_reply = _env_number("VOXATEST_MAX_SILENT", settings.max_consecutive_no_reply, int)
    settings.reports_dir = default_reports_dir()
    settings.failure_report_path = default_failure_report_path()
    if not settings.reasoning_effort:
        settings.reasoning_effort = None
    if settings.backend not in {"llamacpp", "ollama"}:
        settings.backend = DEFAULT_BACKEND
    return settings
