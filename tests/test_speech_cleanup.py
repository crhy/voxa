from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import voxa.speech as speech_mod
from voxa.speech import SpeechService, _run_quietly

_ORIGINAL_EDGE_TTS = speech_mod.edge_tts


class _FakeCommunicate:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def stream(self):
        return self._gen()

    async def _gen(self):
        try:
            for _ in range(3):
                yield {"type": "audio", "data": b"x"}
        finally:
            self.closed = True


def _fake_edge_tts(created):
    mod = ModuleType("edge_tts")

    def factory(*args, **kwargs):
        obj = _FakeCommunicate()
        created.append(obj)
        return obj

    mod.Communicate = factory
    return mod


def _fake_communicate_with_cancel(cancel_event, created):
    class Communicate:
        def __init__(self, *args, **kwargs):
            self.closed = False
            created.append(self)

        def stream(self):
            return self._gen()

        async def _gen(self):
            try:
                yield {"type": "audio", "data": b"a"}
                cancel_event.set()
                yield {"type": "audio", "data": b"b"}
            finally:
                self.closed = True

    return Communicate


def test_speak_removes_asterisk_marks_before_synthesis(monkeypatch):
    service = SpeechService()
    spoken = []

    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(
        service,
        "_start_natural_stream",
        lambda text, _rate, _voice, _cancel: spoken.append(text),
    )

    service.speak(
        "**Important:** keep the rest, including an escaped \\* mark and a full-width ＊ mark.",
        180,
        "en-US-AriaNeural",
        on_started=lambda: None,
        on_done=lambda: None,
        on_error=lambda _error: None,
    )

    assert spoken == ["Important: keep the rest, including an escaped  mark and a full-width  mark."]


def test_prepared_speech_removes_asterisks_too(monkeypatch):
    service = SpeechService()
    received = []

    async def fetch(text, *_args):
        received.append(text)
        return "/tmp/natural.mp3", []

    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(service, "_fetch_to_file", fetch)

    item = service.prepare("This is *also* clean.", 180, "en-US-AriaNeural")

    assert item.ready.wait(1)
    assert received == ["This is also clean."]
    assert item.text == "This is also clean."


def test_cancel_after_first_chunk_closes_stream():
    created = []
    seen = []

    def should_stop():
        seen.append(1)
        return len(seen) > 1

    speech_mod.edge_tts = _fake_edge_tts(created)
    try:
        path, words = _run_quietly(SpeechService._fetch_to_file(None, "hi", 180, "", should_stop))
    finally:
        speech_mod.edge_tts = _ORIGINAL_EDGE_TTS
    assert path == "" and words == []
    assert created[0].closed


def test_normal_run_closes_stream():
    created = []
    speech_mod.edge_tts = _fake_edge_tts(created)
    try:
        path, words = _run_quietly(
            SpeechService._fetch_to_file(None, "hi", 180, "", lambda: False)
        )
    finally:
        speech_mod.edge_tts = _ORIGINAL_EDGE_TTS
    assert path
    assert created[0].closed


def test_cancelled_natural_fetch_closes_stream():
    cancel_event = threading.Event()
    created = []
    speech_mod.edge_tts = ModuleType("edge_tts")
    speech_mod.edge_tts.Communicate = _fake_communicate_with_cancel(cancel_event, created)
    ns = SimpleNamespace(cancel_event=cancel_event, _on_words=None)
    try:
        path = _run_quietly(SpeechService._fetch_natural_audio(ns, "hi", 180, "", cancel_event))
    finally:
        speech_mod.edge_tts = _ORIGINAL_EDGE_TTS
    assert path
    assert created[0].closed


def test_natural_file_worker_retries_before_falling_back(monkeypatch, tmp_path):
    service = SpeechService()
    cancel_event = threading.Event()
    service.cancel_event = cancel_event
    attempts = []
    played = []

    monkeypatch.setattr(service, "_fetch_natural_audio", lambda *_args: object())

    def run_with_one_transient_failure(awaitable):
        attempts.append(awaitable)
        if len(attempts) == 1:
            raise RuntimeError("temporary network failure")
        return str(tmp_path / "natural.mp3")

    monkeypatch.setattr(speech_mod, "_run_quietly", run_with_one_transient_failure)
    monkeypatch.setattr(speech_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(speech_mod, "_ui_once", lambda callback, *args: callback(*args))
    monkeypatch.setattr(service, "_play_file", lambda path, event: played.append((path, event)))

    service._natural_file_worker("hello", 180, "en-US-AriaNeural", cancel_event)

    assert len(attempts) == 2
    assert played == [(str(tmp_path / "natural.mp3"), cancel_event)]


def test_prepared_voice_clears_a_transient_error_after_retry(monkeypatch):
    service = SpeechService()
    attempts = 0

    async def fetch(*_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary network failure")
        return "/tmp/natural.mp3", []

    monkeypatch.setattr(speech_mod, "edge_tts", object())
    monkeypatch.setattr(speech_mod, "NATURAL_VOICE_RETRY_SECONDS", 0)
    monkeypatch.setattr(service, "_fetch_to_file", fetch)

    item = service.prepare("hello", 180, "en-US-AriaNeural")

    assert item.ready.wait(1)
    assert attempts == 2
    assert item.path == "/tmp/natural.mp3"
    assert item.error == ""


def test_natural_fallback_uses_google_voice_before_espeak(monkeypatch):
    service = SpeechService()
    cancel_event = threading.Event()
    service.cancel_event = cancel_event
    played = []

    monkeypatch.setattr(
        speech_mod,
        "_download_google_voice",
        lambda _text, path, *_rest: Path(path).write_bytes(b"natural audio"),
    )
    monkeypatch.setattr(speech_mod, "_ui_once", lambda callback, *args: callback(*args))
    monkeypatch.setattr(service, "_play_file", lambda path, event: played.append((path, event)))
    monkeypatch.setattr(service, "_synthesize_offline", lambda *_args: pytest.fail("eSpeak must not be used"))

    service._synthesize_fallback("hello", 180, cancel_event, "Edge HTTP 403")

    assert len(played) == 1
    assert played[0][1] is cancel_event


def test_strict_voice_mode_rejects_robotic_fallback(monkeypatch):
    service = SpeechService(allow_offline_fallback=False)
    cancel_event = threading.Event()
    service.cancel_event = cancel_event
    errors = []
    service._on_error = errors.append
    service._finished = False

    def fail_google_voice(_text, _path, *_rest):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(speech_mod, "_download_google_voice", fail_google_voice)
    monkeypatch.setattr(speech_mod, "_ui_once", lambda callback, *args: callback(*args))
    monkeypatch.setattr(service, "_synthesize_offline", lambda *_args: pytest.fail("eSpeak must not be used"))

    service._synthesize_fallback("hello", 180, cancel_event, "Edge HTTP 403")

    assert errors and "acoustic test was not run" in errors[0]


def test_strict_voice_mode_does_not_substitute_online_voice(monkeypatch):
    service = SpeechService(
        allow_online_fallback=False,
        allow_offline_fallback=False,
    )
    cancel_event = threading.Event()
    service.cancel_event = cancel_event
    errors = []
    service._on_error = errors.append
    service._finished = False

    monkeypatch.setattr(
        speech_mod,
        "_download_google_voice",
        lambda *_args: pytest.fail("the configured voice must not be substituted"),
    )
    monkeypatch.setattr(speech_mod, "_ui_once", lambda callback, *args: callback(*args))

    service._synthesize_fallback("hello", 250, cancel_event, "Edge HTTP 403")

    assert errors and "acoustic test was not run" in errors[0]
    assert "Edge HTTP 403" in errors[0]


def test_run_quietly_returns_value_and_cleans_up():
    created = []
    speech_mod.edge_tts = _fake_edge_tts(created)

    async def coro():
        stream = speech_mod.edge_tts.Communicate("hi").stream()
        try:
            async for _ in stream:
                break
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        return 42

    try:
        value = _run_quietly(coro())
    finally:
        speech_mod.edge_tts = _ORIGINAL_EDGE_TTS
    assert value == 42
    assert created[0].closed
