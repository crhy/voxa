from __future__ import annotations

import queue
import threading

import pytest

from voxa.dictation import DICTATION_CHUNK_SECONDS, DictationController, segment_stream

LOUD_CHUNK = (b"\x00\x10" * 1000, 1000.0)  # 2000 bytes, above any test threshold
QUIET_CHUNK = (b"\x00\x00" * 1000, 0.0)


def test_segment_stream_yields_one_segment_after_a_pause() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 10 * len(LOUD_CHUNK[0])


def test_segment_stream_ignores_audio_below_threshold() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(5):
        q.put(QUIET_CHUNK)

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
        )
    )

    assert segments == []


def test_segment_stream_stops_immediately_when_stop_event_is_set() -> None:
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    stop_event.set()

    segments = list(
        segment_stream(
            q,
            stop_event,
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
        )
    )

    assert segments == []


def test_segment_stream_calls_on_idle_timeout() -> None:
    q: queue.Queue = queue.Queue()
    called = threading.Event()

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.2,
            on_idle_timeout=called.set,
        )
    )

    assert segments == []
    assert called.is_set()


def test_segment_stream_ends_after_trailing_silence_without_stop_event() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(5):
        q.put(LOUD_CHUNK)
    for _ in range(5):
        q.put(QUIET_CHUNK)

    stop_event = threading.Event()
    segments = list(
        segment_stream(
            q,
            stop_event,
            threshold=500,
            silence_seconds=5.0,
            max_segment_seconds=10.0,
            trailing_silence_seconds=0.3,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 10 * len(LOUD_CHUNK[0])
    assert not stop_event.is_set()


class StoppingTranscriber:
    """Sets the controller's stop event mid-"transcription"."""

    def __init__(self, controller: DictationController) -> None:
        self.controller = controller

    def transcribe(self, segment: bytes, language: str) -> str:
        self.controller.stop()
        return "hello there"


def test_stop_during_transcription_drops_text_callback() -> None:
    texts: list[str] = []
    controller = DictationController(
        whisper=object(),
        language="en",
        threshold=500,
        silence_ms=100,
        max_segment_seconds=5.0,
        on_text=texts.append,
        on_status=lambda _status: None,
        on_auto_stop=lambda: None,
        on_error=lambda _error: None,
    )
    controller.whisper = StoppingTranscriber(controller)

    controller.start()
    for _ in range(10):
        controller.feed(*LOUD_CHUNK)
    for _ in range(5):
        controller.feed(*QUIET_CHUNK)
    if controller.thread is not None:
        controller.thread.join(timeout=3.0)

    assert texts == []


def test_dictation_caps_long_general_command_segments() -> None:
    controller = DictationController(
        whisper=object(),
        language="en",
        threshold=500,
        silence_ms=100,
        max_segment_seconds=20.0,
        on_text=lambda _text: None,
        on_status=lambda _status: None,
        on_auto_stop=lambda: None,
        on_error=lambda _error: None,
    )

    assert controller.max_segment_seconds == DICTATION_CHUNK_SECONDS


def test_early_check_ends_segment_before_full_silence() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)
    for _ in range(5):
        q.put(QUIET_CHUNK)

    calls: list[int] = []

    def early_check(segment: bytes) -> bool:
        calls.append(len(segment))
        return True

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=5.0,
            max_segment_seconds=10.0,
            idle_timeout_seconds=0.3,
            early_silence_seconds=0.3,
            early_check=early_check,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 15 * len(LOUD_CHUNK[0])
    assert len(calls) == 1


def test_early_check_false_runs_to_full_silence() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)
    for _ in range(5):
        q.put(QUIET_CHUNK)

    calls: list[int] = []

    def early_check(segment: bytes) -> bool:
        calls.append(len(segment))
        return False

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.3,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
            early_silence_seconds=0.05,
            early_check=early_check,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 15 * len(LOUD_CHUNK[0])
    assert len(calls) == 1


def test_early_check_called_once_per_pause() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)
    for _ in range(5):
        q.put(QUIET_CHUNK)

    calls: list[int] = []

    def early_check(segment: bytes) -> bool:
        calls.append(len(segment))
        return False

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
            early_silence_seconds=0.05,
            early_check=early_check,
        )
    )

    assert len(segments) == 1
    assert len(calls) == 1


def test_without_early_parameters_timing_is_unchanged() -> None:
    q: queue.Queue = queue.Queue()
    for _ in range(10):
        q.put(LOUD_CHUNK)

    segments = list(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=0.1,
            max_segment_seconds=5.0,
            idle_timeout_seconds=0.3,
        )
    )

    assert len(segments) == 1
    assert len(segments[0]) == 10 * len(LOUD_CHUNK[0])



def test_segment_stream_reads_a_callable_max_segment_on_every_chunk() -> None:
    """Dictation switched on mid-utterance caps the segment already in progress."""
    q: queue.Queue = queue.Queue()
    for _ in range(40):
        q.put(LOUD_CHUNK)
    chunk_seconds = len(LOUD_CHUNK[0]) / 32000
    reads = []

    def limit() -> float:
        reads.append(1)
        # A long command limit for the first chunks, then the short dictation one.
        return 30.0 if len(reads) <= 3 else 6 * chunk_seconds

    segment = next(
        segment_stream(
            q,
            threading.Event(),
            threshold=500,
            silence_seconds=5.0,
            max_segment_seconds=limit,
        )
    )
    assert len(segment) == 6 * len(LOUD_CHUNK[0])


def _seg_probe_steps(probe, verdict_calls) -> list[float]:
    """Feed one voiced second, then silence; return (t, completed_len) per step."""
    from voxa.dictation import SpeechSegmenter
    from voxa.endpoint import EndpointPolicy

    seg = SpeechSegmenter(threshold=500, silence_seconds=0.7, max_segment_seconds=10.0,
                          endpoint_probe=probe, endpoint_policy=EndpointPolicy("default"))
    chunk = b"\x02\x00" * 1600  # 100 ms of int16 samples
    completed_at: list[float] = []
    t = 0.0
    for _ in range(10):
        seg.step(chunk, 600.0, t)
        t += 0.1
    for _ in range(28):
        out = seg.step(b"\x00\x00" * 1600, 100.0, t)
        if out is not None:
            completed_at.append(t)
            break
        t += 0.1
    return completed_at


def test_endpoint_probe_true_ends_at_the_complete_floor() -> None:
    calls: list[int] = []

    def probe(segment: bytes) -> bool:
        calls.append(len(segment))
        return True

    assert _seg_probe_steps(probe, calls) == [pytest.approx(1.4)]  # 0.5 s floor, not 0.7
    assert len(calls) == 1  # 250 ms trigger into the pause


def test_endpoint_probe_false_holds_until_the_incomplete_wait() -> None:
    calls: list[int] = []

    def probe(segment: bytes) -> bool:
        calls.append(len(segment))
        return False

    assert _seg_probe_steps(probe, calls) == [pytest.approx(2.9)]  # 2.0 s incomplete floor
    assert len(calls) == 1


def test_endpoint_probe_none_keeps_the_silence_floor() -> None:
    calls: list[int] = []

    def probe(segment: bytes) -> None:
        calls.append(len(segment))
        return None

    assert _seg_probe_steps(probe, calls) == [pytest.approx(1.6)]  # the 0.7 s the app already had
    assert len(calls) == 1


def test_endpoint_probe_reruns_on_each_new_pause() -> None:
    calls: list[int] = []

    def probe(segment: bytes) -> bool:
        calls.append(len(segment))
        return False

    from voxa.dictation import SpeechSegmenter
    from voxa.endpoint import EndpointPolicy

    seg = SpeechSegmenter(threshold=500, silence_seconds=0.7, max_segment_seconds=10.0,
                          endpoint_probe=probe, endpoint_policy=EndpointPolicy("default"))
    chunk = b"\x02\x00" * 1600
    t = 0.0
    for _ in range(10):
        seg.step(chunk, 600.0, t)
        t += 0.1
    for _ in range(4):  # 0.4 s pause: trigger fires, False holds the turn open
        seg.step(b"\x00\x00" * 1600, 100.0, t)
        t += 0.1
    for _ in range(3):  # the user resumes
        seg.step(chunk, 600.0, t)
        t += 0.1
    for _ in range(4):  # second pause: the probe runs again
        seg.step(b"\x00\x00" * 1600, 100.0, t)
        t += 0.1
    assert len(calls) == 2
