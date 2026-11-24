"""Roadmap task 10: the epoch itself - the live capture restarts together, the bench obeys it.

No GStreamer, no sound card: ``AudioCapture`` is exercised through the pure parts an epoch touches
(the reference ring, the canceller, the latency question, the recovery window), and the bench
runner through its own epoch machinery with a stubbed ASR.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from voxa.audio import AudioCapture, ReferenceRing
from voxa.echo import OUTPUT_BT_A2DP, OUTPUT_BT_HFP, OUTPUT_HEADPHONES, OUTPUT_SPEAKERS
from voxa.onset import OnsetDetector


class StubASR:
    """A fake faster-whisper that answers with one word, so the delivery path runs."""

    model_name = "stub"

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        seconds = len(pcm) / 32000.0
        return {"text": "open the garage", "words": [{"word": "open", "onset_s": 0.0,
                                                      "offset_s": round(seconds, 3)}],
                "no_speech_prob": 0.02, "audio_seconds": round(seconds, 3), "asr_seconds": 0.0,
                "segments": 1}

    def transcribe(self, pcm_s16le: bytes, language: str = "en", hint: str = "") -> str:
        return "open the garage"


# ------------------------------------------------------------------ live capture ---------------

def test_a_device_change_flushes_the_reference_and_opens_a_recovery_window() -> None:
    capture = AudioCapture()
    capture._ref_ring = ReferenceRing(8000)
    capture._ref_ring.append(np.zeros(1000, dtype=np.float32))
    capture._ref_anchor = 7
    capture.timeline = None
    capture._echo_mode = "voxa"
    capture.canceller = object()  # any object: the epoch must let go of it, not use it

    epoch = capture.begin_epoch(OUTPUT_BT_A2DP, reason="bluetooth-profile")
    assert epoch == 1
    assert capture.epoch == 1
    assert capture.output_class == OUTPUT_BT_A2DP
    assert capture.tuning.delay_search_ms == 400.0
    assert capture.tuning.aec3_seed_ms == 200.0
    # the reference history of the OLD device's clock says nothing about this one
    assert capture._ref_anchor is None
    assert capture._ref_ring is not None and capture._ref_ring.total == 0
    # a different acoustic path means a different canceller and no learned delay
    assert capture.canceller is not None
    assert capture._delay_tracker is None
    assert capture._delay_done is False
    assert capture.recovering is True


def test_the_recovery_window_closes_and_the_class_travels() -> None:
    capture = AudioCapture()
    capture._echo_mode = "off"
    capture.begin_epoch(OUTPUT_BT_HFP)
    assert capture.output_class == OUTPUT_BT_HFP
    capture._recovery_until = time.monotonic() - 1.0
    assert capture.recovering is False
    # an epoch that keeps the class does not pretend it changed
    assert capture.begin_epoch() == 2
    assert capture.output_class == OUTPUT_BT_HFP


def test_an_epoch_re_asks_the_live_latency_question() -> None:
    """A Bluetooth profile changes what the driver answers; the alignment must follow."""
    capture = AudioCapture()
    asked: list[int] = []

    def driver_latency() -> float:
        asked.append(len(asked))
        return 0.040 if len(asked) < 2 else 0.180

    capture.set_capture_latency(driver_latency)
    assert capture.capture_clock.latency() == pytest.approx(0.040)
    capture._echo_mode = "off"
    capture.begin_epoch(OUTPUT_BT_A2DP)
    assert len(asked) >= 2  # the epoch re-asked the live driver
    assert capture.capture_clock.latency() == pytest.approx(0.180)


def test_the_first_output_is_read_without_an_epoch() -> None:
    capture = AudioCapture()
    assert capture.note_output(OUTPUT_BT_A2DP) == OUTPUT_BT_A2DP
    assert capture.epoch == 0
    assert capture.recovering is False  # her first request must not be held back
    assert capture.tuning.delay_search_ms == 400.0
    assert capture.note_output("") == OUTPUT_BT_A2DP  # an unread device keeps what we know


def test_unqualified_audio_is_only_about_the_delay_and_the_model() -> None:
    capture = AudioCapture()
    assert capture.echo_path_qualified() is False  # no canceller at all

    class NLMS:
        delay_samples = 0
        delay_managed_internally = False

    capture.canceller = NLMS()
    capture._canceller_enabled = True
    assert capture.echo_path_qualified() is False   # never aligned: unqualified
    NLMS.delay_samples = 640
    assert capture.echo_path_qualified() is True

    class AEC3:
        delay_samples = 0
        delay_managed_internally = True

    capture.canceller = AEC3()
    assert capture.echo_path_qualified() is True    # AEC3 owns the alignment itself


def test_start_device_watch_reads_the_output_once_and_never_changes_it() -> None:
    # a bluez A2DP sink, exactly as pactl shows it
    sinks = (
        "Sink #1\n"
        "\tName: bluez_output.11_22_33_44_55_66.1\n"
        "\tDriver: module-bluez5-device.c\n"
        "\tProperties:\n"
        '\t\tbluetooth.protocol = "a2dp-sink"\n'
        '\t\tdevice.api = "bluez5"\n'
        '\t\tdevice.bus = "bluetooth"\n'
        "\tActive Port: output:a2dp-sink\n"
    )
    seen: list[list[str]] = []

    def runner(argv, **_kwargs):
        seen.append(list(argv))
        out = "bluez_output.11_22_33_44_55_66.1" if "get-default-sink" in argv else (
            sinks if "sinks" in argv else "")
        return _Proc(out)

    capture = AudioCapture()
    try:
        capture.start_device_watch(poll_seconds=50.0, runner=runner)
        assert capture.output_class == OUTPUT_BT_A2DP
        assert capture.epoch == 0  # the first read is not an epoch
        for argv in seen:
            assert argv[1] in ("list", "get-default-sink", "get-default-source", "info")
    finally:
        capture.stop_device_watch()


class _Proc:
    def __init__(self, out: str) -> None:
        self.returncode = 0
        self.stdout = out
        self.stderr = ""


# ------------------------------------------------------------------ bench runner ------------

@pytest.fixture(autouse=True)
def _stub_vad(monkeypatch):
    class StubDetector:
        threshold = 0.5
        last_prob = 0.9

        def reset(self) -> None:
            self.last_prob = 0.9

        def feed(self, pcm: bytes) -> bool:
            return True

    monkeypatch.setattr("voxa.voicedetect.VoiceDetector", StubDetector)


def _runner(echo_mode: str = "voxa"):
    from voxa.bench.runner import ClipRunner
    return ClipRunner(asr=StubASR(), echo_mode=echo_mode)


def test_a_switch_fires_one_epoch_at_its_moment_and_retools_the_pipeline() -> None:
    runner = _runner()
    assert runner.output_class == OUTPUT_SPEAKERS
    assert runner.tuning.delay_search_ms == 300.0

    runner._note_output({"output_class": OUTPUT_SPEAKERS,
                         "device_switch": {"at_s": 1.0, "output_class": OUTPUT_BT_A2DP}})
    runner._apply_output_tuning()
    runner._now = 0.5
    assert runner._switch_due(25) is None           # too early: the switch has not happened yet
    runner._now = 1.2
    wanted = runner._switch_due(60)                 # 1.2 s of audio in
    assert wanted == OUTPUT_BT_A2DP
    runner._begin_epoch(wanted)
    assert runner.epoch == 1
    assert runner.output_class == OUTPUT_BT_A2DP
    assert runner.tuning.delay_search_ms == 400.0
    assert runner.tuning.residual_margin_db > 6.0
    assert runner._recovering() is True
    assert runner._switch_due(70) is None           # exactly one epoch per switch
    epochs = [event for event in runner.events if event["event"] == "device-epoch"]
    assert epochs and epochs[0]["output_class"] == OUTPUT_BT_A2DP
    assert runner.barge.onset._margin_db == 15.0
    assert runner.dt_gate.margin_db == 9.0


def test_an_epoch_flushes_the_reference_and_the_learned_delay() -> None:
    runner = _runner()
    runner.ref_ring = ReferenceRing(1000)
    runner.ref_ring.append(np.zeros(200, dtype=np.float32))
    runner._delay_samples = 3200
    runner._delay_done = True
    runner._begin_epoch(OUTPUT_HEADPHONES)
    assert runner.ref_ring.total == 0
    assert runner._delay_samples == 0
    assert runner._delay_done is False
    assert runner.tuning.echo_path_present is False
    # the onset rule restarts with THIS output's margin: a jack with no echo path hears more
    assert isinstance(runner.barge.onset, OnsetDetector)
    assert runner.barge.onset._margin_db < 12.0


def test_during_recovery_no_command_rises_from_unqualified_audio() -> None:
    runner = _runner()
    runner._recovery_until_s = runner._now + 5.0
    runner._delay_samples = 0  # the canceller cannot say what this microphone is
    before = len(runner.requests)
    runner._deliver("open the garage", 10, kind="request")
    assert len(runner.requests) == before
    assert runner.held_commands == 1
    held = [event for event in runner.events if event["event"] == "command-held"]
    assert held and held[0]["reason"] == "recovery-window-unqualified-audio"

    # the audio is NOT lost: holding is a decision about the command, never a deletion. The
    # reference and the cleaned stream the frame loop keeps are exactly as they were.
    runner.ref_ring = ReferenceRing(1000)
    runner.ref_ring.append(np.zeros(200, dtype=np.float32))
    kept = runner.ref_ring.total
    runner._deliver("open the garage", 10, kind="request")
    assert runner.ref_ring.total == kept
    assert runner._clean_stream == []  # nothing was ever removed from it
    # once the path is aligned again the very same utterance is allowed through
    runner._delay_samples = 640
    runner._deliver("open the garage", 11, kind="request")
    assert len(runner.requests) == before + 1


def test_an_aligned_path_keeps_its_commands_during_recovery() -> None:
    runner = _runner()
    runner._recovery_until_s = runner._now + 5.0
    runner._delay_samples = 480
    runner._deliver("open the garage", 4, kind="request")
    assert runner.requests == ["open the garage"]
    assert runner.held_commands == 0


# ------------------------------------------------ f184: self-command held at her OWN playback ----

def _echo_evidence() -> "object":
    """A span the attribution calls her OWN echo: a live reference, the delayed reference explains
    the microphone, the joint gate's mirror verdict held, and nothing supports a second source."""
    from voxa.attribution import SourceEvidence
    return SourceEvidence(
        reference_active=True, reference_power=0.05,
        echo_explained_db=0.0,            # the reference explains the whole microphone
        residual_over_prediction_db=-3.0,  # residual at/under the echo path's baseline
        joint_echo=True,                   # the MIRROR verdict: this was the delayed reference
        alignment_healthy=True, mic_level=0.02, own_level=0.02, vad_prob=0.0,
    )


def _user_evidence() -> "object":
    """A span the attribution cleanly supports as a SECOND source (the user over her)."""
    from voxa.attribution import SourceEvidence
    return SourceEvidence(
        reference_active=True, reference_power=0.05,
        echo_explained_db=-8.0,            # the microphone holds far beyond the reference
        residual_over_prediction_db=9.0,   # residual well over the echo path's baseline
        joint_suspect=True,                # speech, margin and echo-inconsistency held together
        alignment_healthy=True, mic_level=0.3, own_level=0.02, vad_prob=0.9,
    )


def test_a_self_echo_candidate_at_playback_is_held_not_a_command() -> None:
    """Roadmap task 10 extended (f184): an unqualified candidate attributed to her OWN echo while
    she is playing never opens a turn/request/command - the SAME shape as the recovery-window hold."""
    runner = _runner()
    runner.own_speech.started("turn on the lights")   # her voice is a live source (she is playing)
    runner.last_evidence = _echo_evidence()
    before = len(runner.requests)
    runner._deliver("turn on the lights", 30, kind="request")
    assert len(runner.requests) == before            # never a command from her own echo
    assert runner.held_commands == 1
    held = [e for e in runner.events if e["event"] == "self-command-held"]
    assert held and held[0]["reason"] == "unqualified-echo-at-playback"
    # the audio is KEPT: holding is a decision about the command, never a deletion. The decision is
    # deferred by arming the prompt, exactly like the recovery-window hold.
    assert runner._clean_stream == []
    if runner.conversation is not None:  # the prompt is armed (the decision is only deferred)
        runner.conversation.arm_prompt()
        assert runner.conversation.waiting_for_prompt is True


def test_a_user_command_at_playback_still_routes_and_degraded_is_unchanged() -> None:
    """A genuinely user-spoken candidate during her playback still runs: the attribution cleanly
    supports a second source, so the f184 hold must NOT touch it. The degraded joint-None path (no
    span evidence) is left exactly as handoff 6.5 keeps it."""
    runner = _runner()
    runner.own_speech.started("turn on the lights")
    runner.last_evidence = _user_evidence()
    runner._deliver("open the pod bay", 31, kind="request")
    assert runner.requests == ["open the pod bay"]   # the user's turn over her, never held
    assert runner.held_commands == 0

    # the degraded joint-None path is unchanged: no span evidence, so nothing is held
    runner2 = _runner()
    runner2.own_speech.started("turn on the lights")
    runner2.last_evidence = None
    runner2._deliver("open the pod bay", 32, kind="request")
    assert runner2.requests == ["open the pod bay"]
    assert runner2.held_commands == 0
