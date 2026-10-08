"""Tests for the voice-detector wiring: segment_stream's is_voice hook, the
loudness floor that keeps pure silence cheap, and the conversation-level
fallback when the detector raises."""

from __future__ import annotations

import json
import queue
import threading
import time

from voxa.config import ConfigStore
from voxa.conversation import ConversationController
from voxa.dictation import segment_stream

VOICE = b"\x01\x00" * 1000  # marked "voice" by the fake detector
MUSIC = b"\x00\x10" * 1000  # loud but not voice
QUIET = b"\x00\x00" * 1000
CHUNK_BYTES = len(VOICE)


def _stream(q: queue.Queue, chunks: list[tuple[bytes, float]], delay: float) -> threading.Thread:
    """Feed chunks in real time from a helper thread so wall-clock pauses are real."""

    def feed() -> None:
        for item in chunks:
            q.put(item)
            time.sleep(delay)

    thread = threading.Thread(target=feed, name="feeder")
    thread.start()
    return thread


def test_loud_music_does_not_block_pause_with_voice_detector() -> None:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def is_voice(pcm: bytes, level: float) -> bool:
        return pcm == VOICE

    chunks = [(VOICE, 1000.0)] + [(MUSIC, 1000.0)] * 6 + [(VOICE, 1000.0)] * 5
    feeder = _stream(q, chunks, 0.05)
    segments = list(
        segment_stream(
            q,
            stop,
            threshold=500,
            silence_seconds=0.2,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.6,
            is_voice=is_voice,
        )
    )
    feeder.join(timeout=5.0)
    assert len(segments) == 2  # music is not voice, so the pause cuts after the first word
    assert [len(seg) for seg in segments] == [5 * CHUNK_BYTES, 7 * CHUNK_BYTES]


def test_loud_music_blocks_the_pause_without_detector() -> None:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    chunks = [(VOICE, 1000.0)] + [(MUSIC, 1000.0)] * 6 + [(VOICE, 1000.0)] * 5
    feeder = _stream(q, chunks, 0.05)
    segments = list(
        segment_stream(
            q,
            stop,
            threshold=500,
            silence_seconds=0.2,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.6,
        )
    )
    feeder.join(timeout=5.0)
    assert len(segments) == 1  # the loudness rule hears the music as voice, no pause
    assert len(segments[0]) == 12 * CHUNK_BYTES


def test_pure_silence_never_reaches_the_detector() -> None:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    seen: list[bytes] = []

    def is_voice(pcm: bytes, level: float) -> bool:
        seen.append(pcm)
        return False

    for _ in range(5):
        q.put((QUIET, 0.0))
    stop.set()
    segments = list(
        segment_stream(
            q,
            stop,
            threshold=500,
            silence_seconds=0.15,
            max_segment_seconds=5.0,
            is_voice=is_voice,
        )
    )
    assert segments == []
    assert seen == []  # level < threshold/2 short-circuits before the detector


def test_raising_detector_falls_back_to_loudness_rule() -> None:
    controller = ConversationController(
        wake_whisper=object(),
        prompt_whisper=object(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=100,
        max_segment_seconds=5.0,
        voice_detector=False,
        on_woken=lambda: None,
        on_prompt=lambda _text: None,
        on_status=lambda _status: None,
        on_error=lambda _error: None,
    )
    calls: list[str] = []

    class RaisingDetector:
        def feed(self, pcm: bytes) -> bool:
            calls.append("feed")
            raise RuntimeError("model exploded")

    controller._voice_detector = RaisingDetector()
    controller._voice_fell_back = False
    assert controller._voice_check(MUSIC, 1000.0) is True  # rule: loud enough
    assert controller._voice_fell_back is True
    assert controller._voice_check(MUSIC, 1000.0) is True
    assert controller._voice_check(QUIET, 0.0) is False
    assert len(calls) == 1  # the detector is not consulted again


def test_detector_disabled_uses_loudness_rule() -> None:
    controller = ConversationController(
        wake_whisper=object(),
        prompt_whisper=object(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=100,
        max_segment_seconds=5.0,
        voice_detector=False,
        on_woken=lambda: None,
        on_prompt=lambda _text: None,
        on_status=lambda _status: None,
        on_error=lambda _error: None,
    )
    assert controller._voice_detector is None
    assert controller._voice_check(MUSIC, 1000.0) is True
    assert controller._voice_check(QUIET, 0.0) is False


def test_old_config_still_loads_with_voice_detector_default(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"whisper_model": "base", "silence_ms": 900}), encoding="utf-8")
    settings = ConfigStore(path).load(migrate=False)
    assert settings.voice_detector is True
    assert settings.silence_ms == 900
