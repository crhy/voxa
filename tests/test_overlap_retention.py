"""Milestone A: the words of an interruption become the request (voxa/capture.py,
voxa/conversation.py, voxa/bargectl.py, voxa/bench/runner.py).

Handoff 5.1 (vetted) says the old ``_barge_buf`` was DESTROYED by the check that judged it:
``candidate_text()`` joined and cleared it, ``mute()``/``unmute()`` cleared it, and the accepted
interruption never transferred that audio into the new request, so the user had to repeat
themselves. Handoff 8.4 asks for these fixtures specifically: "Test frame identifiers for loss
and duplication, not just transcript strings."

Everything is offline: generated WAVs, a stub recogniser whose words come from the audio it was
handed, a stub voice detector. No model, no network, no sound device, no sleeping.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from voxa.bench.render import write_wav
from voxa.bench.runner import CHUNK_SAMPLES, ClipRunner, load_clip
from voxa.capture import CaptureRing
from voxa.conversation import ConversationController

ECHO_LEVEL = 800  # her own voice back from the speakers, as the capture path reports it
USER_LEVEL = 3000  # the user over it: the level gate's floor is 2 x the echo baseline
FRAME_BYTES = CHUNK_SAMPLES * 2  # one 20 ms frame of 16-bit mono


class StubDetector:
    """A fake Silero detector: always voice, so the loudness rule alone decides."""

    threshold = 0.5

    def __init__(self) -> None:
        self.last_prob = 0.9

    def reset(self) -> None:
        self.last_prob = 0.9

    def feed(self, pcm: bytes) -> bool:
        self.last_prob = 0.9
        return True


@pytest.fixture(autouse=True)
def _stub_detector(monkeypatch):
    monkeypatch.setattr("voxa.voicedetect.VoiceDetector", StubDetector)


class AudioReadingASR:
    """A recogniser whose words come from the audio it was handed, one per 20 ms frame.

    It reports what the pipeline actually delivered: a frame of the user's voice is
    ``userword``, a frame of her echo is ``herword``, silence is nothing. Counting the words
    therefore counts the frames that reached the recogniser - which is exactly the loss/duplication
    question, and it cannot be gamed by a transcript that was never backed by audio.
    """

    model_name = "stub"

    def __init__(self) -> None:
        self.calls = 0
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        self.calls += 1
        samples = np.frombuffer(pcm, dtype="<i2")
        self.audio_seconds += len(samples) / 16000.0
        words = []
        for start in range(0, len(samples), CHUNK_SAMPLES):
            frame = samples[start:start + CHUNK_SAMPLES]
            if frame.size == 0:
                continue
            peak = int(np.max(np.abs(frame.astype(np.int64))))
            word = ("userword" if peak >= 2000
                    else "herword" if peak >= 400 else None)
            if word is None:
                continue
            words.append({"word": word,
                          "onset_s": round(start / 16000.0, 3),
                          "offset_s": round((start + CHUNK_SAMPLES) / 16000.0, 3)})
        return {
            "text": " ".join(item["word"] for item in words),
            "words": words,
            "no_speech_prob": 0.02,
            "audio_seconds": round(len(samples) / 16000.0, 3),
            "asr_seconds": 0.0,
            "segments": 1 if words else 0,
        }

    def transcribe(self, pcm_s16le: bytes, language: str = "en", hint: str = "") -> str:
        return self.transcribe_words(pcm_s16le, hint)["text"]


def signal(seconds: float, level: int, rate: int = 16000) -> np.ndarray:
    """A constant-amplitude stretch as a WAV carries it (``write_wav`` scales by 32767)."""
    samples = int(round(seconds * rate))
    wave = np.tile([level, -level], samples // 2 + 1)
    return (wave[:samples] / 32768.0).astype(np.float32)


def pcm(seconds: float, level: int, rate: int = 16000) -> bytes:
    """The same stretch as the int16 bytes the capture path moves."""
    samples = int(round(seconds * rate))
    wave = np.tile([level, -level], samples // 2 + 1)
    return wave[:samples].astype("<i2").tobytes()


def word_for(peak: int) -> str | None:
    """What the stub recogniser reports for one frame of that amplitude."""
    return "userword" if peak >= 2000 else "herword" if peak >= 400 else None


def frame_peaks(stream: np.ndarray) -> list[int]:
    """The peak of every 20 ms frame of a clip's microphone stream, by frame id."""
    return [int(np.max(np.abs(stream[index:index + CHUNK_SAMPLES].astype(np.int64))))
            for index in range(0, len(stream), CHUNK_SAMPLES)]


def frames_of(seconds: float) -> int:
    return int(round(seconds * 16000 / CHUNK_SAMPLES))


def write_clip(root, suite: str, clip_id: str, mic: np.ndarray, ref: np.ndarray, truth: dict):
    clip_dir = root / suite / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    write_wav(clip_dir / "mic.wav", mic)
    write_wav(clip_dir / "ref.wav", ref)
    (clip_dir / "truth.json").write_text(json.dumps(truth))
    return clip_dir


def short_over_reply_clip(root):
    """A short request spoken ENTIRELY while Voxa is reading a reply (0.8-1.4 s of a 2.4 s reply)."""
    mic = np.concatenate([signal(0.8, ECHO_LEVEL), signal(0.6, USER_LEVEL),
                          signal(1.0, ECHO_LEVEL), signal(1.6, 0)])
    ref = np.concatenate([signal(2.4, ECHO_LEVEL), signal(1.6, 0)])
    return write_clip(root, "tiny", "tiny-0000-short-00", mic, ref, {
        "id": "tiny-0000-short-00", "scenario": "short", "expect": "keep",
        "playback": {"text": "The weather is clear today.", "onset_s": 0.0, "offset_s": 2.4},
        "user": {"onset_s": 0.8, "offset_s": 1.4,
                 "words": [{"word": "yes", "onset_s": 0.8, "offset_s": 1.4}]},
    })


def long_over_reply_clip(root):
    """A 12 s request that starts under a reply and runs long past the five-second window."""
    mic = np.concatenate([signal(1.5, ECHO_LEVEL), signal(12.0, USER_LEVEL), signal(1.5, 0)])
    ref = np.concatenate([signal(3.0, ECHO_LEVEL), signal(12.0, 0)])
    return write_clip(root, "tiny", "tiny-0001-dt_onset-00", mic, ref, {
        "id": "tiny-0001-dt_onset-00", "scenario": "dt_onset", "expect": "keep",
        "playback": {"text": "Reminding you at three to call Bob.", "onset_s": 0.0,
                     "offset_s": 3.0},
        "user": {"onset_s": 1.5, "offset_s": 13.5,
                 "words": [{"word": "open", "onset_s": 1.5, "offset_s": 1.9}]},
    })


def events_of(runner, name: str) -> list[dict]:
    return [event for event in runner.events if event["event"] == name]


def make_controller(**overrides) -> tuple[ConversationController, AudioReadingASR]:
    asr = AudioReadingASR()
    kwargs = dict(
        wake_whisper=asr,
        prompt_whisper=asr,
        language="en",
        wake_word="voxa",
        threshold=450,
        silence_ms=700,
        max_segment_seconds=20.0,
        early_silence_ms=300,
        voice_detector=False,
        on_woken=lambda: None,
        on_prompt=lambda text: None,
        on_status=lambda text: None,
        on_error=lambda text: None,
    )
    kwargs.update(overrides)
    return ConversationController(**kwargs), asr


# ------------------------------------------------------------------ sequence ids and pinning


def test_every_captured_frame_gets_a_sequence_id_and_never_exists_twice() -> None:
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0])
    for index in range(5):
        clock[0] += 0.02
        frame = ring.add(b"\x01\x02" * (index + 1), 100.0 + index)
        assert frame.seq == index
        assert frame.t_s == pytest.approx(clock[0])
    same = ring.add(b"\x01\x02", 999.0, seq=2)
    assert same.seq == 2 and same.level == 102.0, "re-adding an id must not create a second copy"
    assert ring.frames_retained == 5


def test_onset_pins_from_onset_minus_the_half_second_pre_roll() -> None:
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0])
    for _ in range(100):  # 2 s of 20 ms frames
        clock[0] += 0.02
        ring.add(b"\x00" * FRAME_BYTES, 100.0)
    assert ring.pin(seq=80) is True
    # Frame 80 ends at 1.62 s; the span must start half a second before that onset.
    onset_t = ring.pinned()[0].t_s  # not the onset frame: the pre-roll start
    last = ring.pinned()[-1]
    pinned = ring.pinned()
    onset_frame = next(frame for frame in pinned if frame.seq == 80)
    assert 0.48 <= onset_frame.t_s - onset_t <= 0.52, (onset_frame.t_s, onset_t)
    assert pinned[0].seq < 80 and pinned[-1].seq == 99


def test_the_pin_walks_back_over_the_run_of_speech_the_onset_belongs_to() -> None:
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0])
    levels = ([ECHO_LEVEL] * 41 + [10] * 20 + [USER_LEVEL] * 20)
    for level in levels:
        clock[0] += 0.02
        ring.add(b"\x00" * FRAME_BYTES, float(level))
    assert ring.pin(seq=70, level_floor=1000.0, pre_roll_seconds=0.0) is True
    # The quiet run at frames 41-60 stops the walk-back: the span starts where THIS speech began.
    assert ring.pin_from == 61


def a_pinned_ring():
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], window_seconds=0.4, pre_roll_seconds=0.0)
    for index in range(20):  # 0.4 s
        clock[0] += 0.02
        ring.add(pcm(0.02, USER_LEVEL), float(USER_LEVEL))
    ring.pin(seq=10)
    return ring, clock


def test_the_raw_ring_keeps_at_least_one_second_of_the_pre_onset_region() -> None:
    """Roadmap task 6: the turn is first onset MINUS at least 300 ms (a 1 s raw ring), so the
    raw capture survives the decision about what to do with it. A cold start that begins under
    a reply must still be able to hand the recogniser the user's first word even though the
    microphone was muted and the echo stages were running when it arrived."""
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0])
    # 1.2 s of frames while muted (her playback), raw capture carried on every frame.
    for index in range(60):
        clock[0] += 0.02
        ring.add(pcm(0.02, ECHO_LEVEL), float(ECHO_LEVEL), raw=pcm(0.02, USER_LEVEL))
    # The onset lands on the last frame; the pinned span must reach back at least 300 ms and
    # its raw audio (not just the cleaned/echo-cancelled copy) must still be readable.
    ring.pin(seq=59)
    pinned = ring.pinned()
    assert pinned, "onset-minus-pre-roll pins a span"
    assert pinned[-1].t_s - pinned[0].t_s >= 0.30, "at least 300 ms of pre-roll"
    raw_span = ring.snapshot(raw=True)
    assert raw_span, "the pre-onset raw capture is still there for one pass over the whole turn"
    assert len(raw_span) >= int(0.30 * 16000) * 2, "a full second ring covers the pre-roll"


def test_the_pin_never_walks_back_past_the_onset_minus_pre_roll_floor() -> None:
    """Without a level floor the span is exactly onset minus pre-roll: a cold start's pin cannot
    reach arbitrarily far back into her own echo and hand the turn a prefix that is only her voice."""
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], pre_roll_seconds=0.3)
    for index in range(80):  # 1.6 s, every frame "loud" (echo during playback)
        clock[0] += 0.02
        ring.add(b"\x00" * FRAME_BYTES, float(ECHO_LEVEL))
    ring.pin(seq=60)
    pinned = ring.pinned()
    onset_frame = next(frame for frame in pinned if frame.seq == 60)
    assert pinned[0].seq < 60
    assert onset_frame.t_s - pinned[0].t_s <= 0.30 + 0.021, "the span is onset minus pre-roll, not a free walk-back"


# ------------------------------------------------------------------ f177: POST-AEC pre-roll for the barge-in case


def _post_aec_preroll_ring():
    """Her playback, then the user, with the CLEANED (post-AEC) and the RAW capture carried apart.

    Frames 0-39: she is playing and the user is talking over her. The canceller + mask DESTROY the
    user's first word there - the cleaned frame is quiet (echo cancelled, user masked), while the
    RAW microphone still carries the user (this is exactly what f133 measured: recovering the first
    word from that raw audio re-introduces her echo). Frames 40-59: the user is clearly heard, so the
    onset rule confirms here - 400 ms of unexplained sound after the first word was masked.
    """
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0])
    for _ in range(40):  # 0.8 s under playback: cleaned ~quiet, raw carries the user
        clock[0] += 0.02
        ring.add(pcm(0.02, 120), 120.0, raw=pcm(0.02, USER_LEVEL))
    for _ in range(20):  # the user heard clearly: the confirming onset lands at seq 59
        clock[0] += 0.02
        ring.add(pcm(0.02, USER_LEVEL), float(USER_LEVEL))
    return ring


def test_a_barge_in_turn_starts_at_the_suspect_onset_minus_pre_roll_from_cleaned_audio() -> None:
    """f177: pinning from the suspect onset (the run of unexplained sound the onset confirmed),
    not the late confirming frame, starts the barge-in turn on the cleaned stream earlier - so the
    first word that arrived under her playback is inside the span."""
    ring = _post_aec_preroll_ring()
    plain_ring = _post_aec_preroll_ring()
    # Control: pinning the same onset with NO run length starts at onset minus pre-roll only.
    plain_ring.pin(seq=59, onset_audio_ms=None)
    plain = plain_ring.pin_from
    ring.pin(seq=59, onset_audio_ms=400.0)
    # The suspect onset (seq 59 - 0.4 s = seq 39) minus the 0.5 s pre-roll reaches earlier.
    assert ring.pin_from is not None and ring.pin_from < plain, (ring.pin_from, plain)
    assert ring.pin_from <= 20, "the span starts at the suspect onset minus the pre-roll"
    span = ring.pinned()
    assert span[0].seq == ring.pin_from and span[-1].seq == 59


def test_the_barge_in_pre_roll_uses_cleaned_audio_never_the_raw_ring() -> None:
    """f177: the pre-roll is the CLEANED (post-AEC, post-mask) stream, NOT the raw ring: the raw
    audio under playback carries her echo, and promoting it would pull an echo word into the turn
    (f133's re-introduced echo). The promoted span's bytes must be the cleaned copies."""
    ring = _post_aec_preroll_ring()
    ring.pin(seq=59, onset_audio_ms=400.0)
    span = ring.promote_pinned()
    assert span, "the barge-in turn owes its audio to the record"
    for frame in span:
        # the promoted bytes are the cleaned frame, whose pre-onset region is echo-cancelled quiet
        peak = int(np.max(np.abs(np.frombuffer(frame.pcm, dtype="<i2").astype(np.int64))))
        if frame.seq < 40:
            assert peak < 2000, "cleaned pre-roll frames are the cancelled stream, not the raw copy"
            assert frame.raw is not None, "the raw user-under-echo capture is kept but not used"
            raw_peak = int(np.max(np.abs(np.frombuffer(frame.raw, dtype="<i2").astype(np.int64))))
            assert raw_peak >= 2000, "the raw copy WOULD have carried an echo/user word"
        else:
            assert peak >= 2000, "the clear user frames are still in the span"
    cleaned = b"".join(frame.pcm for frame in span)
    raw = b"".join(frame.raw for frame in span if frame.raw is not None)
    assert cleaned and raw and cleaned != raw, "the turn is the cleaned stream, never the raw ring"
    assert ring.snapshot() == b"", "promotion consumed the span once"


def test_the_pre_roll_never_pulls_an_echo_word_into_the_whole_turn_pass() -> None:
    """Assembling the whole turn with the post-AEC pre-roll and running ONE pass over it must not
    produce her echo word: the cleaned pre-roll region carries no echo, so the stub recogniser sees
    only the user's words (peak >= 2000) and silence, never herword (400 <= peak < 2000)."""
    ring = _post_aec_preroll_ring()
    ring.pin(seq=59, onset_audio_ms=400.0)
    span = b"".join(frame.pcm for frame in ring.promote_pinned())
    asr = AudioReadingASR()
    heard = asr.transcribe_words(span, "")["text"].split()
    assert "herword" not in heard, "the pre-roll re-introduced an echo word into the turn"
    assert "userword" in heard, "the user's own words are still in the assembled whole turn"


def test_the_degraded_none_onset_audio_ms_path_is_unchanged() -> None:
    """A caller that cannot supply the onset run length (None: the degraded joint path, no onset
    rule) pins exactly as before - onset minus pre-roll - and the suspect-onset wiring changes
    nothing about it."""
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], pre_roll_seconds=0.3)
    for _ in range(80):
        clock[0] += 0.02
        ring.add(b"\x00" * FRAME_BYTES, float(ECHO_LEVEL))
    ring.pin(seq=60, onset_audio_ms=None)
    pinned = ring.pinned()
    onset_frame = next(frame for frame in pinned if frame.seq == 60)
    assert onset_frame.t_s - pinned[0].t_s <= 0.30 + 0.021, "None onset_audio_ms = unchanged behaviour"


def test_a_raw_ring_cold_start_is_not_used_for_the_first_word() -> None:
    """f133's cold-start recovery from the RAW ring stays OUT: the turn's audio is always the
    cleaned stream, so a cold first word is never recovered by reading the raw microphone back."""
    ring = _post_aec_preroll_ring()
    ring.pin(seq=59, onset_audio_ms=400.0)
    span = ring.promote_pinned()
    pre_onset = [frame for frame in span if frame.seq < 40]
    assert pre_onset, "the span reaches back before the clear-user frames"
    for frame in pre_onset:
        peak = int(np.max(np.abs(np.frombuffer(frame.pcm, dtype="<i2").astype(np.int64))))
        assert peak < 2000, "the pre-onset region is the cancelled stream, not the raw cold-start audio"
        assert frame.raw is not None and int(np.max(np.abs(np.frombuffer(
            frame.raw, dtype="<i2").astype(np.int64)))) >= 2000, "raw is kept but never promoted"


def test_a_rolling_trim_can_never_overwrite_a_pinned_span() -> None:
    ring, clock = a_pinned_ring()
    for index in range(20, 120):  # two more seconds: the window is only 0.4 s
        clock[0] += 0.02
        ring.add(pcm(0.02, USER_LEVEL), float(USER_LEVEL))
    assert ring.pinned()[0].seq == 10, "the pinned span was trimmed away"
    assert ring.snapshot().startswith(pcm(0.02, USER_LEVEL))
    assert ring.dropped_frames > 0, "unpinned frames outside the window are still bounded and counted"


def test_snapshot_reads_a_copy_and_never_consumes_the_span() -> None:
    ring, _clock = a_pinned_ring()
    first = ring.snapshot()
    second = ring.snapshot()
    assert first == second and first
    assert [frame.seq for frame in ring.pinned()][:3] == [10, 11, 12]
    assert ring.promote_pinned() and ring.snapshot() == b"", "the next read sees only audio not yet promoted"


def test_a_rejected_candidate_is_released_after_its_evidence_window() -> None:
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], evidence_seconds=2.0)
    for _ in range(10):
        clock[0] += 0.02
        ring.add(b"\x00" * FRAME_BYTES, float(USER_LEVEL))
    ring.pin(seq=5)
    ring.note_rejected(now=1.0)
    assert ring.pin_from is not None, "rejected audio is kept for the evidence window"
    clock[0] = 3.01
    ring.add(b"\x00" * FRAME_BYTES, float(USER_LEVEL))
    assert ring.pin_from is None, "the pin is released once the window has passed"
    assert ring.snapshot(), "and releasing the pin does not delete the frames"


# ------------------------------------------------------------------ the controller's contract


def test_candidate_check_does_not_consume_and_promotion_queues_the_span_once() -> None:
    controller, asr = make_controller()
    chunk = pcm(0.02, USER_LEVEL)
    controller.mute()
    for _ in range(10):
        controller.feed(chunk, float(USER_LEVEL))
    assert controller.queue.qsize() == 0, "her own playback never becomes a queued utterance"
    controller.note_onset(seq=5, level_floor=None)
    before = controller.capture.snapshot()
    heard = controller.candidate_text()
    assert heard.count("userword") == 10, heard
    assert controller.capture.snapshot() == before, "the check consumed the span it judged"
    assert asr.calls == 1
    assert controller.candidate_text() == heard and asr.calls == 1, "the same span is not re-transcribed"

    promoted = controller.promote_pinned()
    assert [frame.seq for frame in promoted] == list(range(10))
    queued = []
    while not controller.queue.empty():
        queued.append(controller.queue.get_nowait()[0])
    assert queued == [chunk] * 10
    # The microphone opens again: the same frame id arriving must not be appended twice, and
    # the frame that arrives next must be appended exactly once.
    controller.unmute()
    controller.feed(chunk, float(USER_LEVEL), seq=3)
    assert controller.queue.qsize() == 0, "a promoted frame is never queued a second time"
    controller.feed(pcm(0.02, ECHO_LEVEL), float(ECHO_LEVEL), seq=10)
    assert controller.queue.qsize() == 1, "later audio is appended exactly once"


def test_mute_and_unmute_never_drop_a_pinned_span() -> None:
    controller, _asr = make_controller()
    chunk = pcm(0.02, USER_LEVEL)
    controller.mute()
    for _ in range(8):
        controller.feed(chunk, float(USER_LEVEL))
    controller.note_onset(seq=4, level_floor=None)
    pinned = [frame.seq for frame in controller.capture.pinned()]
    controller.mute()
    controller.unmute()
    controller.mute()
    assert [frame.seq for frame in controller.capture.pinned()] == pinned == list(range(8))
    assert controller.capture.snapshot(), "unmute() used to clear the buffer; the audio is owed to the turn"


def test_the_gate_reports_the_onset_the_controller_pins_from() -> None:
    from voxa.bargectl import BARGE_IN_LISTEN_SECONDS, BargeInController

    clock = [10.0]
    control = BargeInController(clock=lambda: clock[0])
    control.reply_started("The weather is clear today.")
    outcomes = []
    for _ in range(80):  # 1.6 s of 20 ms samples, her echo then the user over it
        clock[0] += 0.02
        level = 0.2 if clock[0] < 10.8 else 0.75
        outcomes.append(control.observe(level, speaking=True, muted=True, wake_word="voxa"))
    assert any(outcome.onset for outcome in outcomes), "the onset tick is what pins the span"
    onset = next(index for index, outcome in enumerate(outcomes) if outcome.onset)
    assert outcomes[onset].reason == "listening"
    assert control.onset_level_floor() > 0.0


# ------------------------------------------------------------------ end to end through the bench runner


def run_clip(clip_dir):
    loaded = load_clip(clip_dir)
    runner = ClipRunner(asr=AudioReadingASR(), echo_mode="off", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
    return runner, summary, loaded


def test_a_short_request_spoken_during_a_reply_reaches_the_request_intact(tmp_path) -> None:
    runner, summary, _loaded = run_clip(short_over_reply_clip(tmp_path / "data"))

    confirms = events_of(runner, "barge-in-confirm")
    assert len(confirms) == 1, summary["counts"]
    promoted = events_of(runner, "capture-promote")
    assert len(promoted) == 1
    ids = promoted[0]["frames"]
    assert ids == sorted(set(ids)), "a frame of the accepted span is missing or counted twice"
    assert summary["requests"], "the interruption's words must become the request"
    request = summary["requests"][0]
    user_frames = frames_of(0.6)
    assert request.count("userword") == user_frames, (request, user_frames)
    # The words that proved the interruption reach the request exactly once: not twice, not not at all.
    heard = confirms[0]["heard"]
    reached = sum(1 for text in summary["requests"] if heard in text)
    assert reached == 1, (heard, summary["requests"])


def test_a_twelve_second_request_over_a_reply_survives_the_ring(tmp_path) -> None:
    clip_dir = long_over_reply_clip(tmp_path / "data")
    runner, summary, loaded = run_clip(clip_dir)

    assert summary["counts"]["barge_in"] == 1
    promoted = events_of(runner, "capture-promote")
    assert len(promoted) == 1
    ids = promoted[0]["frames"]
    assert ids == sorted(set(ids)) and len(ids) > frames_of(1.0)

    # The frames the pipeline handed to the speech-pause rules, by id: the pinned span plus the
    # audio that arrived after it. The stub recogniser reports one word per frame that carries
    # speech, so the request's word counts are a direct measure of loss and duplication.
    accepted = runner._segmented
    peaks = frame_peaks(loaded["mic"])
    first_user = frames_of(1.5)
    last_user = frames_of(13.5) - 1
    assert first_user in accepted, "the beginning of the request was lost"
    assert last_user in accepted, "the end of the request was lost"
    assert set(ids) <= accepted

    expected = [word_for(peaks[seq]) for seq in accepted]
    request = summary["requests"][0]
    for word in ("userword", "herword"):
        assert request.count(word) == expected.count(word), (word, request[:120], len(accepted))
    assert expected.count("userword") >= frames_of(11.0), "the 12 s request survived the ring"

    capture = runner.conversation.capture
    assert capture.pinned()[0].seq <= ids[0], "the pinned span still starts at the request's start"
    assert capture.dropped_frames < first_user, "nothing the user had already spoken is thrown away"


# ------------------------------------------------------------------ f176 pair-candidate completion
# A pinned candidate span the 900 ms window never promoted (an echo-measured onset that blocked the
# yield, a HOLD, or a candidate that never collected 900 ms before the clip ended) is transcribed
# exactly once into a segment transcript and stays HOLD - never a silent drop, never a command from
# weak audio (handoff 6.4/6.5). These drive the runner's completion path with the stub recogniser.


def _voxa_only_runner(tmp_path):
    """A runner over a clip where Voxa reads a reply and the user never interrupts: no user audio
    ever reaches a segment transcript, so the pinned-span completion path is unexercised."""
    mic = signal(3.0, ECHO_LEVEL)
    ref = signal(3.0, ECHO_LEVEL)
    clip = write_clip(Path(tmp_path) / "data", "tiny", "tiny-0002-voxa_only-00", mic, ref, {
        "id": "tiny-0002-voxa_only-00", "scenario": "voxa", "expect": "none",
        "playback": {"text": "The weather is clear today.", "onset_s": 0.0, "offset_s": 3.0},
        "user": None,
    })
    runner, _summary, _loaded = run_clip(clip)
    assert not events_of(runner, "transcript"), "no user speech reached a segment transcript"
    return runner


def test_a_pinned_candidate_that_never_completes_is_transcribed_not_dropped(tmp_path) -> None:
    runner = _voxa_only_runner(tmp_path)
    controller = runner.conversation
    controller.mute()
    start = controller.capture._last_seq + 1
    for offset in range(30):  # 0.6 s of the user over the reply, muted: never segmented
        controller.feed(pcm(0.02, USER_LEVEL), float(USER_LEVEL), seq=start + offset)
    controller.note_onset(seq=start + 10, level_floor=None)
    assert controller.capture.pin_from is not None, "the onset pins the candidate span"
    before_requests = len(events_of(runner, "request"))
    runner._finish_unpromoted_pinned_span()
    transcripts = events_of(runner, "transcript")
    assert transcripts, "the never-completed span reaches a segment transcript (never a silent drop)"
    holds = [e for e in events_of(runner, "endpoint") if e.get("dropped") == "hold"]
    assert holds, "and the transcript is marked HOLD, never a completed utterance"
    assert len(events_of(runner, "request")) == before_requests, "no command runs from audio this weak"


def test_the_completion_promotes_the_pinned_span_exactly_once(tmp_path) -> None:
    runner = _voxa_only_runner(tmp_path)
    controller = runner.conversation
    controller.mute()
    start = controller.capture._last_seq + 1
    for offset in range(20):
        controller.feed(pcm(0.02, USER_LEVEL), float(USER_LEVEL), seq=start + offset)
    controller.note_onset(seq=start + 5, level_floor=None)
    runner._finish_unpromoted_pinned_span()
    first = len(events_of(runner, "transcript"))
    assert first == 1
    runner._finish_unpromoted_pinned_span()
    assert len(events_of(runner, "transcript")) == first, "the span is transcribed exactly once"


def test_a_completed_clip_is_never_re_transcribed_by_the_completion(tmp_path) -> None:
    # short_over_reply_clip interrupts and already transcribed its user audio: the completion must
    # not transcribe the accepted words a second time (that would skew the WER/endpoint metrics).
    runner, _summary, _loaded = run_clip(short_over_reply_clip(tmp_path / "data"))
    before = len(events_of(runner, "transcript"))
    assert before >= 1
    runner._finish_unpromoted_pinned_span()
    assert len(events_of(runner, "transcript")) == before


def test_promote_false_finish_emits_a_transcript_but_never_a_request(tmp_path) -> None:
    runner, _summary, _loaded = run_clip(short_over_reply_clip(tmp_path / "data"))
    before_requests = len(events_of(runner, "request"))
    runner._finish_utterance(pcm(0.6, USER_LEVEL), 0, promote=False)
    assert len(events_of(runner, "transcript")) >= 1
    assert len(events_of(runner, "request")) == before_requests, "HOLD never becomes a command"
    assert any(e.get("dropped") == "hold" for e in events_of(runner, "endpoint"))


def test_a_recovered_clip_protected_word_is_counted_kept(tmp_path) -> None:
    """The segment transcript the completion adds carries the user's words into the record, so the
    metric that reads the segment transcripts (``protected_word_samples``) counts them kept. The
    stub recogniser maps the pinned user audio to ``userword``; a real one would put the command."""
    from voxa.bench import metrics

    runner = _voxa_only_runner(tmp_path)
    controller = runner.conversation
    controller.mute()
    start = controller.capture._last_seq + 1
    for offset in range(20):
        controller.feed(pcm(0.02, USER_LEVEL), float(USER_LEVEL), seq=start + offset)
    controller.note_onset(seq=start + 5, level_floor=None)
    runner._finish_unpromoted_pinned_span()
    # Build a metric fact whose user truth is the short answer the completion transcribed.
    truth = {"id": "tiny-0002-voxa_only-00", "scenario": "short", "expect": "keep",
             "playback": {"text": "x", "onset_s": 0.0, "offset_s": 3.0},
             "user": {"onset_s": 0.0, "offset_s": 0.4,
                      "words": [{"word": "userword", "onset_s": 0.0, "offset_s": 0.02}]}}
    fact = metrics.clip_facts(truth, runner.events, None)
    heard = metrics._heard_words(fact)
    assert "userword" in heard, "the completion put the user's word on the segment transcript"
    values = metrics.protected_word_samples([fact])["values"]
    assert values and sum(values) >= 1, "the protected word is counted kept"
