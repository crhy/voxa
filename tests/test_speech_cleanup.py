from __future__ import annotations

import contextlib
import threading
from types import ModuleType, SimpleNamespace

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
