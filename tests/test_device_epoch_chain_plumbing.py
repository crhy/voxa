"""f179: the f137 device/epoch work merged onto the shipped f178 chain.

The merge had to keep BOTH halves of the plumbing: the chain's joint-gate sweep
(``--gate-*`` -> dt_params, the mask sweep, the word window, the data root) and f137's
per-output table, device switch and recovery window. These tests pin that they coexist:
the sweep survives the tuning, a non-speakers output overrides it from the table, an
epoch re-anchors the CHAIN's exact lock (and starts the residual stage ungifted), and
the recovery window holds automatic requests only - never turns, never audio.

No GStreamer, no sound card, stub ASR and VAD (as tests/test_device_epoch.py does).
"""

from __future__ import annotations

import numpy as np
import pytest

from voxa.audio import AudioCapture
from voxa.echo import OUTPUT_BT_A2DP, OUTPUT_SPEAKERS
from voxa.refmask import LockedResidualMask


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


def _runner(**dt_params) -> "object":
    from voxa.bench.runner import ClipRunner
    runner = ClipRunner(asr=StubASR(), echo_mode="voxa", dt_params=dt_params or None)
    runner._prepare_echo_stages()
    return runner


# ------------------------------------------------------------------ sweep AND tuning --------

def test_the_runner_keeps_the_joint_gate_sweep_through_the_output_tuning() -> None:
    """f178's joint-gate sweep and f137's _note_output/_apply_output_tuning coexist: a
    speakers clip (every frozen clip) keeps the swept values, not the table's."""
    runner = _runner(joint_ms=100.0, speech_threshold=0.5, margin_db=4.0)
    runner._note_output({})  # frozen clips: speakers, no switch
    runner._apply_output_tuning()
    assert runner.output_class == OUTPUT_SPEAKERS
    assert runner.dt_gate.joint_ms == 100.0          # the sweep survived
    assert runner.dt_gate.speech_threshold == 0.5
    assert runner.dt_gate.margin_db == 4.0


def test_a_non_speakers_output_overrides_the_sweep_from_the_table() -> None:
    """The table still rules the device strata: A2DP widens and raises over the sweep."""
    runner = _runner(joint_ms=100.0, margin_db=4.0)
    runner._note_output({"output_class": OUTPUT_BT_A2DP})
    runner._apply_output_tuning()
    assert runner.dt_gate.margin_db == 9.0           # A2DP residual margin
    assert runner.dt_gate.joint_ms == 200.0          # A2DP joint run
    assert runner.barge.onset._margin_db == 15.0     # A2DP onset margin


def test_run_clips_passes_the_whole_sweep_and_the_data_root() -> None:
    """The merged ClipRunner(...) call site still passes mask_params, dt_params,
    word_window_ms AND data_root - the f178 lesson, kept for f179."""
    import inspect

    from voxa.bench import runner as runner_mod
    src = inspect.getsource(runner_mod.run_clips)
    for needle in ("mask_params", "dt_params", "word_window_ms", "data_root"):
        assert needle in src


# ------------------------------------------------------------------ epoch + the lock --------

def test_an_epoch_re_anchors_the_exact_lock_and_starts_the_mask_ungifted() -> None:
    """The chain's delay machinery restarts with the path: a fresh ExactDelayLock (its
    aligner IS the runner's), and a residual mask with no delay of the old path gifted."""
    runner = _runner()
    old_lock, old_aligner = runner._delay_lock, runner._aligner
    old_mask = runner.mask
    assert old_lock is not None and old_mask is not None
    runner._begin_epoch(OUTPUT_BT_A2DP)
    assert runner._delay_lock is not old_lock
    assert runner._aligner is runner._delay_lock.aligner
    assert runner._aligner is not old_aligner
    assert isinstance(runner.mask, LockedResidualMask) and runner.mask is not old_mask
    assert runner._lock_proven is False
    assert runner._locked_ref_offset == 0
    assert runner._delay_frac == 0.0


def test_the_live_epoch_clears_the_chain_delay_lock_and_residual_gift() -> None:
    """AudioCapture.begin_epoch drops the ExactDelayLock (which holds the fallback
    estimate too) and rebuilds the residual stage, exactly like a mode change does."""
    capture = AudioCapture()
    capture._echo_mode = "voxa"
    capture.canceller = object()          # the epoch lets go of it, never uses it
    capture._delay_lock = object()
    capture._residual_mask = object()
    capture.begin_epoch(OUTPUT_BT_A2DP)
    assert capture._delay_lock is None    # the learned delay of the old path is gone
    assert isinstance(capture._residual_mask, LockedResidualMask)


def test_an_epoch_keeps_the_joint_gate_sweep_for_speakers() -> None:
    """An epoch that lands while a sweep is in force keeps the swept values until the
    output is not speakers: the table never silently replaces a running sweep."""
    runner = _runner(joint_ms=100.0)
    runner._begin_epoch(OUTPUT_SPEAKERS)  # e.g. back to speakers: nothing overrides
    assert runner.dt_gate.joint_ms == 100.0


# ------------------------------------------------------------------ recovery rule ---------

def test_the_recovery_window_holds_requests_never_turns_never_audio() -> None:
    """Task 10's rule at the delivery funnel: inside the window an automatic COMMAND from
    unqualified audio is held; the user's own turn (kind="turn") and the clip's audio
    are untouched - holding is a decision, never a deletion."""
    runner = _runner()
    runner._recovery_until_s = runner._now + 5.0
    runner._delay_samples = 0            # unqualified: no delay, no echo model yet
    runner._deliver("open the garage", 10, kind="request")
    assert runner.requests == []
    assert runner.held_commands == 1
    kept = runner.ref_ring.total
    runner._deliver("happy birthday to you", 11, kind="turn")
    assert runner.requests == ["happy birthday to you"]   # the turn reaches the record
    assert runner.ref_ring.total == kept                  # and no audio was ever removed
