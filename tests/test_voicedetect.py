from __future__ import annotations

import numpy as np
import pytest

from voxa.voicedetect import DoubleTalkGate, VoiceDetector


def _load_detector():
    try:
        return VoiceDetector(threshold=0.5)
    except Exception:  # the onnx model or onnxruntime is unavailable
        return None


def test_silence_is_not_voice() -> None:
    det = _load_detector()
    if det is None:
        pytest.skip("silero model unavailable")
    chunk = bytes(320 * 2)  # 20 ms of zeros
    for _ in range(20):
        assert det.feed(chunk) is False


def test_streaming_matches_one_shot() -> None:
    det = _load_detector()
    if det is None:
        pytest.skip("silero model unavailable")
    rng = np.random.default_rng(7)
    n = 512 * 5
    pcm = (rng.uniform(-0.4, 0.4, n) * 32768).astype("<i2")

    for i in range(0, n, 320):
        det.feed(pcm[i : i + 320].tobytes())
    streamed = np.asarray(det.probs, dtype=np.float64)

    det.reset()
    det.feed(pcm.tobytes())  # one-shot: the whole audio in a single chunk
    one_shot = np.asarray(det.probs, dtype=np.float64)

    assert len(streamed) == len(one_shot)
    assert np.max(np.abs(streamed - one_shot)) < 1e-3


def _frames(seq, size: int = 320):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def test_double_talk_gate_echo_only_clip_is_echo_consistent() -> None:
    """No near-end source: the microphone IS the delayed reference. The mirror verdict must
    fire and the suspect must not."""
    rng = np.random.default_rng(3)
    delay, n = 100, 320 * 10
    ref_full = rng.uniform(-0.5, 0.5, n + delay)
    mic_full = np.concatenate((np.zeros(delay), ref_full[:n]))  # mic[t] = ref[t - delay]
    gate = DoubleTalkGate()
    out = False
    for ref_f, mic_f in zip(_frames(ref_full), _frames(mic_full)):
        out = gate.observe(dt_ms=20.0, prob=0.1, mic=mic_f, ref=ref_f, delay_samples=delay,
                           residual_over_prediction_db=0.0, erle_db=12.0,
                           reference_active=True, alignment_healthy=True)
    assert out is False
    assert gate.coherence is not None and gate.coherence > 0.9
    assert gate.suspect is False
    assert gate.echo_consistent is True
    assert gate.erle_baseline_db is not None


def test_double_talk_gate_nearend_over_echo_raises_the_suspect() -> None:
    """A second source uncorrelated with the reference: speech + margin + inconsistency
    together raise the suspect after joint_ms of them."""
    rng = np.random.default_rng(4)
    gate = DoubleTalkGate()
    out = False
    for _ in range(10):
        out = gate.observe(dt_ms=20.0, prob=0.9,
                           mic=rng.uniform(-0.5, 0.5, 320), ref=rng.uniform(-0.5, 0.5, 320),
                           delay_samples=100, residual_over_prediction_db=10.0, erle_db=None,
                           reference_active=True, alignment_healthy=True)
    assert out is True
    assert gate.suspect is True
    assert gate.joint_run_ms >= 150.0
    assert gate.echo_consistent is False


def test_double_talk_gate_suspect_holds_through_short_gaps() -> None:
    """The suspect persists while joint evidence is only briefly missing (< hold_ms) and
    dies after a longer gap."""
    rng = np.random.default_rng(5)
    gate = DoubleTalkGate()
    for _ in range(10):
        gate.observe(dt_ms=20.0, prob=0.9,
                     mic=rng.uniform(-0.5, 0.5, 320), ref=rng.uniform(-0.5, 0.5, 320),
                     delay_samples=100, residual_over_prediction_db=10.0, erle_db=None,
                     reference_active=True, alignment_healthy=True)
    assert gate.suspect is True
    for _ in range(8):  # 160 ms gap < hold_ms 300: still suspect
        gate.observe(dt_ms=20.0, prob=0.1,
                     mic=rng.uniform(-0.5, 0.5, 320), ref=rng.uniform(-0.5, 0.5, 320),
                     delay_samples=100, residual_over_prediction_db=10.0, erle_db=None,
                     reference_active=True, alignment_healthy=True)
    assert gate.suspect is True
    for _ in range(10):  # past hold_ms: released
        gate.observe(dt_ms=20.0, prob=0.1,
                     mic=rng.uniform(-0.5, 0.5, 320), ref=rng.uniform(-0.5, 0.5, 320),
                     delay_samples=100, residual_over_prediction_db=10.0, erle_db=None,
                     reference_active=True, alignment_healthy=True)
    assert gate.suspect is False


def test_double_talk_gate_is_honest_when_it_cannot_measure() -> None:
    """No probability, no margin, no alignment: neither verdict fires - 'not suspect' is
    INDETERMINATE, never evidence of echo."""
    gate = DoubleTalkGate()
    for _ in range(10):
        out = gate.observe(dt_ms=20.0, prob=None, mic=np.zeros(320), ref=np.zeros(320),
                           delay_samples=None, residual_over_prediction_db=None, erle_db=None,
                           reference_active=False, alignment_healthy=False)
    assert out is False
    assert gate.suspect is False
    assert gate.echo_consistent is False
    assert gate.conditions["measurable"] is False
