from __future__ import annotations

import sys
import threading
import types

import pytest

from voxa.transcription import TranscriptionError, WhisperService, _cuda_compute_usable


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    def __init__(self, model_name: str, **_kwargs) -> None:
        if model_name == "broken-model":
            raise RuntimeError("model weights missing")
        self.model_name = model_name

    def transcribe(self, _audio, **_kwargs):
        return [FakeSegment(f"transcribed by {self.model_name}")], None


def install_fake_whisper(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_DEVICE", "cpu")
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )


def test_failed_load_keeps_previous_model_usable(monkeypatch) -> None:
    install_fake_whisper(monkeypatch)
    service = WhisperService()

    ready = threading.Event()
    ready2 = threading.Event()
    errors: list[str] = []

    service.load_async(
        "base",
        on_ready=lambda _name, _backend: ready.set(),
        on_error=errors.append,
    )
    assert ready.wait(5)
    assert service.ready
    assert service.model_name == "base"

    service.load_async(
        "broken-model",
        on_ready=lambda _name, _backend: ready2.set(),
        on_error=errors.append,
    )
    assert not ready2.wait(1)
    assert errors
    assert service.ready
    assert service.model_name == "base"
    assert service.transcribe(b"\x00\x00\x00\x00") == "transcribed by base"


def test_successful_load_swaps_in_new_model(monkeypatch) -> None:
    install_fake_whisper(monkeypatch)
    service = WhisperService()

    ready = threading.Event()
    service.load_async(
        "small",
        on_ready=lambda _name, _backend: ready.set(),
        on_error=lambda _error: None,
    )
    assert ready.wait(5)
    assert service.model_name == "small"
    assert service.transcribe(b"\x00\x00\x00\x00") == "transcribed by small"


def test_transcribe_before_any_load_raises(monkeypatch) -> None:
    install_fake_whisper(monkeypatch)
    with pytest.raises(TranscriptionError):
        WhisperService().transcribe(b"\x00\x00\x00\x00")


def test_cuda_force_env_var_prefers_new_name_with_legacy_fallback(monkeypatch) -> None:
    monkeypatch.delenv("VOXA_FORCE_CUDA", raising=False)
    monkeypatch.delenv("VOICE2TEXT_FORCE_CUDA", raising=False)
    monkeypatch.setenv("VOXA_FORCE_CUDA", "1")
    assert _cuda_compute_usable()
    monkeypatch.delenv("VOXA_FORCE_CUDA")
    monkeypatch.setenv("VOICE2TEXT_FORCE_CUDA", "1")
    assert _cuda_compute_usable()
