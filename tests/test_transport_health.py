"""A4 (issue #72): bounded memory, no raw bypass in the echo tail, counted drops, visible degraded state.

Everything runs offline on synthetic arrays: no microphone, no speaker, no audio device, no GTK.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals  # noqa: E402  (shared synthetic signals for the echo tests)

from voxa.anc import EchoCanceller, erle_db
from voxa.audio import AudioCapture, ReferenceRing, mix_chunk
from voxa.bargectl import BargeInController
from voxa.capture import CaptureRing
from voxa.conversation import ConversationController
from voxa.transport import TransportHealth

SR = anc_signals.SR


class FakeASR:
    """Enough of a WhisperService for the controller's constructor; never called here."""

    def transcribe(self, *args, **kwargs):
        return ""

    def transcribe_words(self, *args, **kwargs):
        return {"text": "", "words": [], "no_speech_prob": 1.0, "audio_seconds": 0.0,
                "asr_seconds": 0.0}

    ready = True
    calls = 0
    audio_seconds = 0.0
    processing_seconds = 0.0


def make_controller() -> ConversationController:
    asr = FakeASR()
    return ConversationController(
        wake_whisper=asr, prompt_whisper=asr, language="en", wake_word="voxa",
        threshold=450, silence_ms=700, max_segment_seconds=20.0, early_silence_ms=300,
        voice_detector=False,
        on_woken=lambda: None, on_prompt=lambda text: None,
        on_status=lambda text: None, on_error=lambda text: None,
    )


def feed(canceller, mic, ref, chunk=160):
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    return np.concatenate(outs)


# ---------------------------------------------------------------------------
# 1. The reference working set is bounded, and identical to the old unbounded one
#    (handoff 5.7: "the custom reference history grows as blocks are appended").
# ---------------------------------------------------------------------------


class UnboundedReference(EchoCanceller):
    """The pre-A4 canceller: every reference sample ever appended, read by absolute position."""

    def _push_reference(self, ref_b: np.ndarray) -> None:
        self._ref_hist = np.concatenate([self._ref_hist, ref_b])
        self._ref_first = 0

    def _delayed_ref(self, length: int, lookback: int = 0) -> np.ndarray:
        start = self._pos - self._delay - lookback
        span = length + lookback
        if start >= 0:
            return self._ref_hist[start:start + span]
        head = np.zeros(-start)
        return np.concatenate([head, self._ref_hist[:span + start]])


def test_bounded_working_set_gives_identical_output_over_60s() -> None:
    ref, echo, speech, delay = anc_signals.make(seconds=8)  # 8 s of the fixture, 56 s below
    mic = (echo + speech).astype(np.float32)
    ref = np.tile(ref, 7)
    mic = np.tile(mic, 7).astype(np.float32)
    assert len(ref) >= 50 * SR
    old, new = UnboundedReference(delay=delay), EchoCanceller(delay=delay)
    out_old = feed(old, mic, ref)
    out_new = feed(new, mic, ref)
    assert out_old.shape == out_new.shape
    assert np.array_equal(out_old, out_new)  # bit for bit, not "close enough"
    assert erle_db(mic, out_new) > 0.0  # and the filter really did something


def test_working_set_stays_flat_over_two_hours() -> None:
    canceller = EchoCanceller(delay=1600)
    capacity = canceller._reference_capacity()
    assert capacity < 20_000  # a fixed working set, not a session-length history
    block = np.full(1600, 0.25, dtype=np.float64)
    sizes = set()
    for index in range(2 * 3600 * SR // 1600):  # 2 h of stream position
        canceller._push_reference(block)
        canceller._pos += 1600
        if index > 5:  # after the working set has filled
            sizes.add(canceller._ref_hist.size)
    assert sizes == {capacity}
    # The old code would have held all 2 h: 115.2M float64 samples, ~737 MB.
    assert 2 * 3600 * SR * 8 > 700_000_000


def test_working_set_still_covers_the_oldest_echo_it_can_be_asked_about() -> None:
    canceller = EchoCanceller(delay=1600)
    block = np.arange(160, dtype=np.float64)
    for _ in range(400):  # far past the capacity
        canceller._push_reference(block)
        canceller._pos += 160
    tail = canceller._echo_tail_samples()
    window = canceller._delayed_ref(160, tail)
    assert window.size == 160 + tail
    assert np.any(window != 0.0)  # not zero-padded away


# ---------------------------------------------------------------------------
# 2. No raw bypass in the echo tail: voxa/audio.py:62-63 and the canceller's paths
# ---------------------------------------------------------------------------


class RecordingCanceller:
    def __init__(self) -> None:
        self.calls = 0
        self.refs: list[np.ndarray] = []

    def process(self, mic, ref):
        self.calls += 1
        self.refs.append(np.asarray(ref, dtype=np.float32))
        return np.zeros(len(mic), dtype=np.float32)  # a canceller that removes everything


def test_mix_chunk_feeds_the_canceller_even_when_the_reference_window_is_silent() -> None:
    ring = ReferenceRing(3 * SR)
    ring.append(np.zeros(SR, dtype=np.float32))  # an idle monitor
    canceller = RecordingCanceller()
    raw = (np.arange(1600, dtype="<i2")).tobytes()
    out = mix_chunk(raw, ring, 0, canceller, 0)
    assert canceller.calls == 1
    assert np.array_equal(canceller.refs[0], np.zeros(1600, dtype=np.float32))
    assert out != raw  # the chunk went through the canceller, not around it


def test_echo_tail_of_a_clip_is_cancelled_not_returned_raw() -> None:
    """The bug as a clip: her reply stops, its echo is still arriving in the microphone."""
    ref, _echo, _speech, delay = anc_signals.make()
    n = 6 * SR
    ref_cut = ref[:n].copy()
    ref_cut[4 * SR:] = 0.0  # the speakers go silent at 4 s
    mic = anc_signals.echo_path(ref_cut, delay)  # ... and the echo of what they DID play keeps arriving
    ring = ReferenceRing(3 * SR)
    canceller = EchoCanceller(delay=delay)
    raw_all, cleaned_all = [], []
    for i in range(0, n, 320):
        ring.append(ref_cut[i:i + 320])
        # The microphone travels as the int16 PCM the pipeline moves, not as floats.
        chunk = np.rint(np.asarray(mic[i:i + 320], dtype=np.float32) * 32767.0).astype("<i2").tobytes()
        raw_all.append(chunk)
        cleaned_all.append(mix_chunk(chunk, ring, i, canceller, 0))
    start, stop = 4 * SR + delay, 4 * SR + delay + int(0.05 * SR)
    raw = np.frombuffer(b"".join(raw_all)[start * 2:stop * 2], dtype="<i2")
    out = np.frombuffer(b"".join(cleaned_all)[start * 2:stop * 2], dtype="<i2")
    assert raw.size == out.size > 0
    assert float(np.mean((raw / 32768.0) ** 2)) > 1e-6  # the tail is audible in the microphone
    assert erle_db(raw, out) >= 15.0  # and the pipeline removed it


def test_the_canceller_still_refuses_to_adapt_with_nothing_in_flight() -> None:
    ref, echo, _speech, delay = anc_signals.make()
    canceller = EchoCanceller(delay=delay)
    feed(canceller, echo[:2 * SR], ref[:2 * SR])
    weights = canceller.weights.copy()
    silent = np.zeros(SR, dtype=np.float32)
    feed(canceller, silent, silent)  # 1 s: the echo tail finishes inside this
    assert np.array_equal(canceller.weights, weights)
    assert canceller.pass_through_blocks > 0


def test_near_end_speech_after_the_tail_is_untouched() -> None:
    ref, echo, speech, delay = anc_signals.make()
    canceller = EchoCanceller(delay=delay)
    feed(canceller, echo[:2 * SR], ref[:2 * SR])
    silent = np.zeros(SR, dtype=np.float32)
    feed(canceller, silent, silent)  # past the tail
    mic = speech[:SR].astype(np.float32)
    out = feed(canceller, mic, np.zeros(SR, dtype=np.float32))
    assert np.array_equal(out, mic)  # the user's voice is never cancelled


# ---------------------------------------------------------------------------
# 3. Counted drops: capture buffers, queue overruns, canceller fallbacks
# ---------------------------------------------------------------------------


def test_capture_buffer_gaps_are_counted_as_dropped_buffers() -> None:
    health = TransportHealth()
    for start in range(0, 10 * 1000, 1000):  # 1 ms buffers, no gap
        assert health.note_capture_buffer(1000.0, float(start)) == 0
    assert health.capture_buffers_dropped == 0
    assert health.note_capture_buffer(1000.0, 13_000.0) == 2  # two buffers missing
    assert health.capture_buffers_dropped == 2
    assert health.drops == 2


def test_the_capture_callback_counts_a_hole_in_the_microphone_stream() -> None:
    class Sample:
        def __init__(self, timestamp: float, duration: float) -> None:
            self._timestamp, self._duration = timestamp, duration

        def get_default_duration(self) -> float:
            return self._duration

        def get_timestamp(self) -> float:
            return self._timestamp

    capture = AudioCapture()
    capture._note_capture_buffer(Sample(0.0, 1000.0))
    capture._note_capture_buffer(Sample(3000.0, 1000.0))  # one buffer vanished
    assert capture.health.capture_buffers_dropped == 1
    capture._note_capture_buffer(Sample(-1.0, 1000.0))  # Gst.NO_TIMESTAMP: ignored
    assert capture.health.capture_buffers_dropped == 1
    capture.health.reset_capture_clock()
    capture._note_capture_buffer(Sample(0.0, 1000.0))  # a restarted source is not a drop
    assert capture.health.capture_buffers_dropped == 1


def test_queue_overruns_are_counted() -> None:
    conversation = make_controller()
    conversation.health = TransportHealth()
    for _ in range(200):  # the queue holds 80
        conversation._enqueue(b"\x00\x01", 0.0)
    assert conversation.health.queue_overruns > 0
    assert conversation.health.snapshot()["total"] == conversation.health.queue_overruns


def test_a_canceller_that_raises_is_a_counted_fallback_and_capture_keeps_running() -> None:
    class Broken:
        def process(self, mic, ref):
            raise RuntimeError("filter overflow")

    capture = AudioCapture()
    capture.canceller = Broken()
    capture._ref_ring = ReferenceRing(100)
    capture._ref_ring.append(np.ones(100, dtype=np.float32))
    capture._mic_counter = 0
    capture._ref_anchor = 0
    raw = (np.arange(160, dtype="<i2")).tobytes()
    out = capture._mix_or_passthrough(raw)
    assert out == raw  # the microphone is kept and recording never breaks
    assert capture._mic_counter == 160  # the stream position keeps advancing
    assert capture.health.canceller_fallbacks == 1
    assert capture.health.degraded is True
    assert capture.echo_qualified is False


def test_the_health_snapshot_adds_the_caller_s_own_counters() -> None:
    health = TransportHealth()
    health.note_queue_overrun()
    health.note_canceller_fallback("boom")
    snapshot = health.snapshot(pinned_frames_dropped=3, hearing_drops=4)
    assert snapshot["queue_overruns"] == 1
    assert snapshot["canceller_fallbacks"] == 1
    assert snapshot["pinned_frames_dropped"] == 3
    assert snapshot["hearing_drops"] == 4
    assert snapshot["total"] == 9
    assert snapshot["degraded"] is True


def test_the_capture_ring_counts_only_losses_inside_a_span() -> None:
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], rate=SR, window_seconds=0.2,
                       max_pinned_seconds=0.3)
    for index in range(40):  # 40 x 10 ms, nothing pinned: a rolling window, not a loss
        clock[0] = index * 0.01
        ring.add(b"\x00\x01", 0.1, seq=index, t_s=index * 0.01)
    assert ring.dropped_frames > 0
    assert ring.dropped_pinned_frames == 0
    ring.pin(35)
    for index in range(40, 120):  # the pinned span runs past its bound
        ring.add(b"\x00\x01", 0.1, seq=index, t_s=index * 0.01)
    assert ring.dropped_pinned_frames > 0


# ---------------------------------------------------------------------------
# 4. Degraded state: no automatic command from unqualified audio, audio kept
# ---------------------------------------------------------------------------


def unqualified(controller: BargeInController) -> object:
    return controller.observe(
        0.9, speaking=True, muted=True, wake_word="voxa",
        candidate=lambda: ("open the window", "open the window"),
        qualified=False, dt_ms=20.0,
    )


def test_unqualified_audio_holds_instead_of_becoming_a_command() -> None:
    outcome = unqualified(BargeInController())
    assert outcome.action == "hold"
    assert outcome.reason == "unqualified-audio"


def test_qualified_audio_still_decides() -> None:
    controller = BargeInController()
    outcome = None
    for _ in range(80):
        outcome = controller.observe(0.9, speaking=True, muted=True, wake_word="voxa",
                                     candidate=lambda: ("open the window", "open the window"),
                                     dt_ms=20.0)
    assert outcome.reason != "unqualified-audio"


def test_the_degraded_state_keeps_capturing_and_keeps_the_audio() -> None:
    conversation = make_controller()
    conversation.health = TransportHealth()
    conversation.health.note_canceller_fallback("filter overflow")
    conversation.mute()
    for index in range(20):
        conversation.feed(b"\x00\x01" * 160, 0.5, seq=index)
    conversation.note_onset(seq=19)
    before = conversation.capture.snapshot()
    outcome = unqualified(BargeInController())
    assert outcome.action == "hold"
    assert conversation.capture.snapshot() == before  # nothing was discarded
    assert conversation.capture.pin_from is not None  # the span is still pinned
    assert conversation.health.degraded is True  # and the state is visible to the caller


def test_echo_switched_off_by_choice_is_not_a_failure() -> None:
    capture = AudioCapture()
    capture.health.note_canceller_fallback("boom")
    assert capture.echo_qualified is False
    capture.set_echo_mode("off")
    assert capture.echo_qualified is True
    capture.set_echo_mode("voxa", reference_source="pulse:monitor")
    assert capture.echo_qualified is True
    assert capture.health.canceller_fallbacks == 1  # the counter stays, the flag clears


# ---------------------------------------------------------------------------
# 5. The bench reports the counters instead of guessing them
# ---------------------------------------------------------------------------


def test_the_transport_metric_sums_the_runner_s_counters() -> None:
    from voxa.bench import metrics as metrics_module

    facts = [
        {"summary": {"transport": {"total": 0, "capture_buffers_dropped": 0,
                                   "queue_overruns": 0, "canceller_fallbacks": 0,
                                   "pinned_frames_dropped": 0, "hearing_drops": 0}}},
        {"summary": {"transport": {"total": 2, "capture_buffers_dropped": 2,
                                   "queue_overruns": 0, "canceller_fallbacks": 0,
                                   "pinned_frames_dropped": 0, "hearing_drops": 0}}},
    ]
    assert metrics_module.transport_drop_samples(facts)["values"] == [0.0, 2.0]
    row = metrics_module.measure(metrics_module.transport_drop_samples(facts), seed=0)
    assert row["value"] == 2.0
    # A clip run before the counters existed has no number, and is not counted as a zero.
    assert metrics_module.transport_drop_samples([{"summary": {}}])["values"] == []


def test_transport_drop_count_is_no_longer_reported_unmeasurable() -> None:
    from voxa.bench import metrics as metrics_module

    assert "transport_drop_count" not in metrics_module.UNMEASURED
    builder = {name: b for name, _unit, _direction, b in metrics_module.METRICS
               if name == "transport_drop_count"}
    assert builder["transport_drop_count"] is not None
