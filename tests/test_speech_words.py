"""Word-boundary plumbing in :mod:`voxa.speech` without network or GStreamer.

The natural stream is faked: a fake ``edge_tts.Communicate`` yields ``WordBoundary``
and ``audio`` messages, and a fake ``Gst`` stands in for the real pipeline so the
test never touches ``gi``.  This exercises the two new pieces of P05 in speech:
the ``on_words`` callback firing once per boundary (ticks converted to seconds)
and :meth:`SpeechService.position` reporting playback seconds.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from voxa.speech import SpeechService


def _fake_gst():
    """A stand-in for the GStreamer module used by ``_push_natural_audio``."""
    flow_ok = types.SimpleNamespace(value_nick="ok")
    gst = types.ModuleType("Gst")
    gst.FlowReturn = types.SimpleNamespace(OK=flow_ok)
    gst.Buffer = types.SimpleNamespace(new_allocate=lambda *args: types.SimpleNamespace(fill=lambda *args: True))
    return gst, flow_ok


def _fake_communicate(messages: list[dict]):
    """A fake ``edge_tts.Communicate`` whose ``stream()`` yields ``messages``."""

    async def _gen():
        for message in messages:
            yield message

    class _Communicate:
        def stream(self):
            return _gen()

    return _Communicate()


def test_on_words_fires_once_per_boundary(monkeypatch) -> None:
    gst, flow_ok = _fake_gst()
    monkeypatch.setattr("voxa.speech.Gst", gst)
    monkeypatch.setattr("voxa.speech.GLib", types.SimpleNamespace(idle_add=lambda fn, *args: fn(*args)))
    monkeypatch.setattr(
        "voxa.speech.edge_tts.Communicate",
        lambda *args, **kwargs: _fake_communicate(
            [
                {"type": "WordBoundary", "text": "hello", "offset": 0, "duration": 5_000_000},
                {"type": "audio", "data": b"\x00" * 16},
                {"type": "WordBoundary", "text": "world", "offset": 5_000_000, "duration": 5_000_000},
                {"type": "audio", "data": b"\x01" * 16},
            ]
        ),
    )

    collected: list[tuple[str, float, float]] = []
    service = SpeechService()
    service._on_words = lambda words: collected.extend(words)

    appsrc = types.SimpleNamespace(emit=lambda *args: flow_ok)
    service.appsrc = appsrc
    service.cancel_event = asyncio.Event()  # only .is_set() and identity are used

    asyncio.run(
        service._push_natural_audio(
            "hello world",
            180,
            "en-US-AriaNeural",
            appsrc,
            service.cancel_event,
            lambda: None,
        )
    )

    assert collected == [("hello", 0.0, 0.5), ("world", 0.5, 0.5)]


def test_on_words_none_is_harmless(monkeypatch) -> None:
    gst, flow_ok = _fake_gst()
    monkeypatch.setattr("voxa.speech.Gst", gst)
    monkeypatch.setattr("voxa.speech.GLib", types.SimpleNamespace(idle_add=lambda fn, *args: fn(*args)))
    monkeypatch.setattr(
        "voxa.speech.edge_tts.Communicate",
        lambda *args, **kwargs: _fake_communicate(
            [
                {"type": "WordBoundary", "text": "hello", "offset": 0, "duration": 5_000_000},
                {"type": "audio", "data": b"\x00" * 16},
            ]
        ),
    )

    service = SpeechService()
    service._on_words = None
    appsrc = types.SimpleNamespace(emit=lambda *args: flow_ok)
    service.appsrc = appsrc
    service.cancel_event = asyncio.Event()

    asyncio.run(
        service._push_natural_audio(
            "hello",
            180,
            "en-US-AriaNeural",
            appsrc,
            service.cancel_event,
            lambda: None,
        )
    )


def test_position_uses_pipeline_when_known() -> None:
    service = SpeechService()
    service.pipeline = types.SimpleNamespace(get_position=lambda: 1.25)
    assert service.position() == 1.25


def test_position_falls_back_to_wall_clock() -> None:
    service = SpeechService()
    service.pipeline = None
    service._started_at = time.monotonic() - 2.0
    assert service.position() == pytest.approx(2.0)


def test_position_is_zero_before_start() -> None:
    service = SpeechService()
    assert service.position() == 0.0
