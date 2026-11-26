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


def test_transcribe_stashes_segment_confidence_for_the_barge_guard(monkeypatch) -> None:
    """Issue #72, task v11d (Claude's 18:35 note): the hallucination guard was fed no
    confidence at all. The service keeps the LAST pass's own segment figures - max
    no_speech_prob, mean avg_logprob, max compression_ratio - so the words the barge-in
    judge reads arrive with the bar Whisper gave those very words."""

    class ConfSegment(FakeSegment):
        def __init__(self, text: str, nsp: float, avg: float, ratio: float) -> None:
            super().__init__(text)
            self.no_speech_prob = nsp
            self.avg_logprob = avg
            self.compression_ratio = ratio

    class ConfModel:
        def __init__(self, _model_name: str, **_kwargs) -> None:
            pass

        def transcribe(self, _audio, **_kwargs):
            return [ConfSegment("the link is in the description.", 0.04, -1.7, 2.9),
                    ConfSegment("second bit", 0.02, -0.5, 1.2)], None

    monkeypatch.setenv("VOXA_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=ConfModel))
    service = WhisperService()
    ready = threading.Event()
    service.load_async("base", lambda *_a: ready.set(), lambda _e: None)
    assert ready.wait(5)
    service.transcribe(b"\x00\x00\x00\x00" * 16)
    stats = service.last_stats
    assert stats is not None
    assert stats["no_speech_prob"] == pytest.approx(0.04)
    assert stats["avg_logprob"] == pytest.approx((-1.7 + -0.5) / 2.0)
    assert stats["compression_ratio"] == pytest.approx(2.9)

    # A service (or fake) whose segments report nothing leaves the figures None rather than
    # crashing the pass that produced the words.
    install_fake_whisper(monkeypatch)
    plain = WhisperService()
    ready2 = threading.Event()
    plain.load_async("base", lambda *_a: ready2.set(), lambda _e: None)
    assert ready2.wait(5)
    plain.transcribe(b"\x00\x00\x00\x00" * 16)
    assert plain.last_stats is not None
    assert plain.last_stats["no_speech_prob"] is None


def test_cuda_force_env_var_prefers_new_name_with_legacy_fallback(monkeypatch) -> None:
    monkeypatch.delenv("VOXA_FORCE_CUDA", raising=False)
    monkeypatch.delenv("VOICE2TEXT_FORCE_CUDA", raising=False)
    monkeypatch.setenv("VOXA_FORCE_CUDA", "1")
    assert _cuda_compute_usable()
    monkeypatch.delenv("VOXA_FORCE_CUDA")
    monkeypatch.setenv("VOICE2TEXT_FORCE_CUDA", "1")
    assert _cuda_compute_usable()


def test_whisper_skips_the_gpu_when_it_is_nearly_full() -> None:
    from voxa.hardware import GpuUsage
    from voxa.transcription import MIN_FREE_VRAM_GB, _gpu_has_room

    assert _gpu_has_room(GpuUsage(10.0, 15.3, 16.0)) is False
    assert _gpu_has_room(GpuUsage(10.0, 16.0 - MIN_FREE_VRAM_GB, 16.0)) is True
    assert _gpu_has_room(GpuUsage(0.0, 2.0, 16.0)) is True


def test_transcribe_biases_whisper_towards_the_wake_word(monkeypatch) -> None:
    """Issue #72, task v11e: the wake hint carries "Voxa" as initial_prompt, and the
    wake word is also passed as hotwords, so a quiet far mic is heard as "Voxa" rather
    than the model dropping her name and the wake gate abstaining."""

    class PromptModel:
        def __init__(self, _name: str, **_kw) -> None:
            self.kwargs: dict = {}

        def transcribe(self, _audio, **kwargs):
            self.kwargs = kwargs
            return [FakeSegment("Voxa open the map")], None

        def __call__(self, *_a, **_k):  # pragma: no cover - unused
            raise AssertionError

    monkeypatch.setenv("VOXA_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=PromptModel))
    service = WhisperService()
    ready = threading.Event()
    service.load_async("tiny", lambda *_a: ready.set(), lambda _e: None)
    assert ready.wait(5)
    service.transcribe(b"\x00\x00" * 8, "en", "Voxa.", "Voxa")
    kw = service._model.kwargs
    assert kw["initial_prompt"] == "Voxa."
    assert kw["hotwords"] == "Voxa"
    # with no hotword given the token stays None (nothing forced)
    service.transcribe(b"\x00\x00" * 8, "en", "")
    assert service._model.kwargs["hotwords"] is None
