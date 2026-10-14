from __future__ import annotations

from voxa.conversation import ConversationController
from voxa.speechfit import RtfGuard, lighter
from voxa.transcription import WhisperService


class StubModel:
    """Stand-in for a loaded Faster Whisper model: only the inference call is stubbed."""

    def transcribe(self, audio, **kwargs):
        class _Segment:
            text = "hello"

        return ([_Segment()], None)


def real_whisper(name: str) -> WhisperService:
    service = WhisperService()
    service._model = StubModel()
    service._model_name = name
    return service


class FakeWhisper:
    def __init__(self, name: str) -> None:
        self._name = name

    def transcribe(self, segment: bytes, language: str, hint: str = "") -> str:
        return "hello"

    @property
    def model_name(self) -> str:
        return self._name


def make_controller(on_transcribe_timing=None):
    wake = FakeWhisper("tiny")
    prompt = FakeWhisper("base")
    conv = ConversationController(
        wake_whisper=wake,
        prompt_whisper=prompt,
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=800,
        max_segment_seconds=10.0,
        on_woken=lambda: None,
        on_prompt=lambda text: None,
        on_status=lambda text: None,
        on_error=lambda text: None,
    )
    conv.on_transcribe_timing = on_transcribe_timing
    return conv


def test_short_clips_ignored():
    guard = RtfGuard()
    guard.add("base", 0.5, 0.4)
    assert guard.median("base") is None
    assert not guard.should_step_down("base")


def test_not_enough_clips():
    guard = RtfGuard()
    for _ in range(4):
        guard.add("base", 2.0, 1.8)
    assert not guard.should_step_down("base")


def test_five_slow_clips_step_down():
    guard = RtfGuard()
    for _ in range(5):
        guard.add("base", 2.0, 1.8)
    assert guard.should_step_down("base")


def test_five_fast_clips_stay():
    guard = RtfGuard()
    for _ in range(5):
        guard.add("base", 2.0, 0.6)
    assert not guard.should_step_down("base")


def test_one_slow_outlier_stays():
    guard = RtfGuard()
    for _ in range(4):
        guard.add("base", 2.0, 0.2)
    guard.add("base", 2.0, 1.8)
    assert not guard.should_step_down("base")


def test_models_separate():
    guard = RtfGuard()
    for _ in range(5):
        guard.add("base", 2.0, 1.8)
        guard.add("tiny", 2.0, 0.2)
    assert guard.should_step_down("base")
    assert not guard.should_step_down("tiny")


def test_reset():
    guard = RtfGuard()
    for _ in range(5):
        guard.add("base", 2.0, 1.8)
    guard.reset("base")
    assert guard.median("base") is None
    assert not guard.should_step_down("base")


def test_lighter_chain():
    assert lighter("large-v3") == "small"
    assert lighter("turbo") == "small"
    assert lighter("medium") == "small"
    assert lighter("small") == "base"
    assert lighter("base") == "tiny"
    assert lighter("tiny") is None


def test_transcribe_reports_timing():
    calls: list[tuple[str, float, float]] = []

    def hook(model: str, audio: float, took: float) -> None:
        calls.append((model, audio, took))

    conv = make_controller(hook)
    segment = b"\0" * 64000
    text = conv._transcribe(segment, conv.prompt_whisper)
    assert text == "hello"
    assert len(calls) == 1
    model, audio, took = calls[0]
    assert model == "base"
    assert audio == 2.0
    assert took > 0.0


def test_transcribe_wake_model_not_timed():
    calls: list[tuple[str, float, float]] = []
    conv = make_controller(lambda model, audio, took: calls.append((model, audio, took)))
    conv._transcribe(b"\0" * 64000, conv.wake_whisper)
    assert calls == []


def test_raising_hook_does_not_break():
    def boom(model: str, audio: float, took: float) -> None:
        raise RuntimeError("hook failed")

    conv = make_controller(boom)
    text = conv._transcribe(b"\0" * 64000, conv.prompt_whisper)
    assert text == "hello"


def make_real_controller(on_transcribe_timing=None):
    wake = FakeWhisper("tiny")
    prompt = real_whisper("base")
    conv = ConversationController(
        wake_whisper=wake,
        prompt_whisper=prompt,
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=800,
        max_segment_seconds=10.0,
        on_woken=lambda: None,
        on_prompt=lambda text: None,
        on_status=lambda text: None,
        on_error=lambda text: None,
    )
    conv.on_transcribe_timing = on_transcribe_timing
    return conv


def test_real_service_reports_timing():
    calls: list[tuple[str, float, float]] = []

    def hook(model: str, audio: float, took: float) -> None:
        calls.append((model, audio, took))

    conv = make_real_controller(hook)
    text = conv._transcribe(b"\0" * 64000, conv.prompt_whisper)
    assert text == "hello"
    assert len(calls) == 1
    model, audio, took = calls[0]
    assert model == "base"
    assert audio == 2.0
    assert took > 0.0


def test_real_service_wake_not_timed():
    calls: list[tuple[str, float, float]] = []
    conv = make_real_controller(lambda model, audio, took: calls.append((model, audio, took)))
    conv._transcribe(b"\0" * 64000, conv.wake_whisper)
    assert calls == []


def test_real_service_raising_hook_does_not_break():
    def boom(model: str, audio: float, took: float) -> None:
        raise RuntimeError("hook failed")

    conv = make_real_controller(boom)
    text = conv._transcribe(b"\0" * 64000, conv.prompt_whisper)
    assert text == "hello"
