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


# --- f172 (roadmap task 5): the gate on the LOCKED-delay chain ---------------------------------
# The f132 blocker was that this gate read a delay that drifted 1.5-2.5 ms; f168 shipped the
# exact render-timeline lock (delay error 0.07 ms) and f171 ships LockedResidualMask on that
# output. These fixtures replay the shipping situation: the echo arrives at the delay the lock
# proved, the room tail rings after it, and the near end is the only thing the streams miss.

def _echo_train(ref: np.ndarray, delay: int, taps: tuple[float, ...]) -> np.ndarray:
    """The reference through a decaying room tail (the RT60 0.2-0.8 s of the frozen clips):
    the first tap at exactly the delay the lock proved, the rest ringing behind it."""
    mic = np.zeros(ref.size + delay + 40 * len(taps), dtype=np.float64)
    for b, gain in enumerate(taps):
        shift = delay + 40 * b
        mic[shift:shift + ref.size] += gain * ref
    return mic


def _locked_delay_fixture(n_frames: int, delay: int, rng, *, user: bool = False,
                          user_from: int = 5, user_to: int | None = None):
    """One clip of the shipping path: mic frames = echo train, plus (for the user case) an
    uncorrelated speechful second source the delayed reference cannot be."""
    user_to = user_to if user_to is not None else n_frames
    ref_frames, mic_frames = [], []
    ref_full = rng.uniform(-0.4, 0.4, n_frames * 320 + delay + 512)
    mic_full = _echo_train(ref_full, delay, (1.0, 0.45, 0.25, 0.12))
    if user:
        u = rng.uniform(-1.2, 1.2, mic_full.size)
        lo = user_from * 320
        hi = user_to * 320
        u[:lo] = 0.0
        u[hi:] = 0.0
        mic_full = mic_full + u
    for f in range(n_frames):
        ref_frames.append(ref_full[f * 320:(f + 1) * 320])
        mic_frames.append(mic_full[f * 320:(f + 1) * 320])
    return ref_frames, mic_frames


def test_double_talk_gate_locked_delay_with_room_tail_is_echo_consistent() -> None:
    """Echo at the locked delay with a ringing tail: the mirror verdict fires, the suspect
    does not - the coherence of the microphone against the reference AT THE LOCKED DELAY is
    what f132 could not measure (its delay drifted), and now it measures high."""
    rng = np.random.default_rng(11)
    delay = 2000
    gate = DoubleTalkGate()
    ref_frames, mic_frames = _locked_delay_fixture(20, delay, rng)
    out = False
    for ref_f, mic_f in zip(ref_frames, mic_frames):
        out = gate.observe(dt_ms=20.0, prob=0.1, mic=mic_f, ref=ref_f, delay_samples=delay,
                           residual_over_prediction_db=0.0, erle_db=15.0,
                           reference_active=True, alignment_healthy=True)
    assert out is False
    assert gate.suspect is False
    assert gate.coherence is not None and gate.coherence >= 0.5
    assert gate.echo_consistent is True


def test_double_talk_gate_locked_delay_user_raises_suspect_never_echo() -> None:
    """The user starts mid-clip over her echo, at the same locked delay: the three streams
    hold together and the mirror verdict the echo-only frames raised is RELEASED past its
    hold_ms - the gate separates near-exact echo from a second source on the SAME delay,
    which is the re-plan f132 was parked for."""
    rng = np.random.default_rng(12)
    delay = 2000
    gate = DoubleTalkGate()
    ref_frames, mic_frames = _locked_delay_fixture(32, delay, rng, user=True, user_from=16,
                                                   user_to=32)
    suspect_at = echo_at = None
    for f, (ref_f, mic_f) in enumerate(zip(ref_frames, mic_frames)):
        prob = 0.9 if 16 <= f < 32 else 0.1
        margin = 10.0 if 16 <= f < 32 else 0.0
        gate.observe(dt_ms=20.0, prob=prob, mic=mic_f, ref=ref_f, delay_samples=delay,
                     residual_over_prediction_db=margin, erle_db=15.0,
                     reference_active=True, alignment_healthy=True)
        if gate.suspect and suspect_at is None:
            suspect_at = f
        if gate.echo_consistent and echo_at is None:
            echo_at = f
    assert suspect_at is not None and suspect_at >= 16  # ~joint_ms of user frames
    assert echo_at is not None and echo_at < 16        # only the echo-only frames claimed echo
    assert gate.echo_consistent is False               # released past hold_ms of user span


def test_double_talk_gate_mirror_holds_through_a_syllable_dip() -> None:
    """f172: the MIRROR verdict persists through a coherence dip shorter than hold_ms (a word
    boundary in her own speech is not evidence of a second source), and dies past it - the
    gap rule the suspect already used, now mirrored."""
    rng = np.random.default_rng(13)
    delay = 2000
    gate = DoubleTalkGate()
    ref_frames, mic_frames = _locked_delay_fixture(20, delay, rng)
    for ref_f, mic_f in zip(ref_frames, mic_frames):
        gate.observe(dt_ms=20.0, prob=0.1, mic=mic_f, ref=ref_f, delay_samples=delay,
                     residual_over_prediction_db=0.0, erle_db=15.0,
                     reference_active=True, alignment_healthy=True)
    assert gate.echo_consistent is True
    # 160 ms of mic the reference cannot explain: shorter than hold_ms -> the verdict holds.
    for _ in range(8):
        gate.observe(dt_ms=20.0, prob=0.1, mic=rng.uniform(-0.4, 0.4, 320),
                     ref=rng.uniform(-0.4, 0.4, 320), delay_samples=delay,
                     residual_over_prediction_db=0.0, erle_db=15.0,
                     reference_active=True, alignment_healthy=True)
    assert gate.echo_consistent is True
    # Past hold_ms of the same: the mirror is released, and "not echo" stays INDETERMINATE.
    for _ in range(16):
        gate.observe(dt_ms=20.0, prob=0.1, mic=rng.uniform(-0.4, 0.4, 320),
                     ref=rng.uniform(-0.4, 0.4, 320), delay_samples=delay,
                     residual_over_prediction_db=0.0, erle_db=15.0,
                     reference_active=True, alignment_healthy=True)
    assert gate.echo_consistent is False
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
