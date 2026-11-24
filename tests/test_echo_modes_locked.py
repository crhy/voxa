"""Hearing 2 (task f164): every linear arm given Voxa's LOCKED render-timeline delay.

The point of the task is that Voxa knows the exact samples she plays and when (the render
timeline, ``voxa.align.ExactAligner``), so the bulk echo delay can be LOCKED, not searched.
Each locked arm is run against its UNLOCKED twin on a small synthetic room with a KNOWN bulk
delay and drift, in the two ways the task names:

  * an arm whose library cannot be handed the delay at all (SpeexDSP MDF, the AEC3 wheel) is
    given it by re-timing the reference OUTSIDE it - a ring offset, exactly what
    ``voxa.audio.mix_chunk`` does in the runner;
  * an arm that can (Voxa's NLMS) is given it fractionally, via ``set_offset``.

Fast, offline, no sound: synthetic signals only. The locked arm must converge faster OR cancel
deeper than its twin, and never crash on the aligner's confidence fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxa.align import ExactAligner
from voxa.anc import Aec3Canceller, EchoCanceller
from voxa.bench import modes

SR = 16000
BLOCK = 160  # every canceller here processes 10 ms frames


# --------------------------------------------------------------------------- synthetic room

def broadband(n: int, seed: int = 0) -> np.ndarray:
    """A broadband reference: a delay cannot be measured from a few pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float64)


def room(seed: int, tail_s: float = 0.02) -> np.ndarray:
    """Room impulse response: a direct spike then a decaying tail (the part the MDF/wheel search)."""
    rng = np.random.default_rng(seed)
    n = int(tail_s * SR)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.008 * SR))
    h = 0.3 * tail / np.sqrt(np.sum(tail ** 2))
    h[0] += 1.0
    return h.astype(np.float64)


def echo_at(ref: np.ndarray, delay_fn) -> np.ndarray:
    """ref delayed by ``delay_fn(t_seconds)`` samples (linear interp -> fractional lag)."""
    n = len(ref)
    t = np.arange(n) / SR
    lag = np.asarray(delay_fn(t), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64) - lag
    return np.interp(idx, np.arange(n, dtype=np.float64), ref, left=0.0, right=0.0)


def ref_window(ref: np.ndarray, start: int, length: int) -> np.ndarray:
    """The reference the canceller is fed when the bulk delay is re-timed at the ring: exactly the
    ``voxa.audio.ReferenceRing.window`` the runner's mix_chunk pulls, zero-padded at the edges."""
    out = np.zeros(length, dtype=np.float64)
    if length <= 0:
        return out
    start = int(round(start))
    lo = max(start, 0)
    hi = min(start + length, len(ref))
    if hi > lo:
        out[lo - start:hi - start] = ref[lo:hi]
    return out


def _process(canceller, mic, ref, delay_fn=None) -> np.ndarray:
    """Run a canceller over the clip in 160-sample frames. ``delay_fn`` re-times the reference by
    the known bulk delay (the ring offset the locked arms use); ``None`` leaves the arm searching."""
    out = np.zeros(len(mic), dtype=np.float64)
    for i in range(0, len(mic) - BLOCK + 1, BLOCK):
        start = i if delay_fn is None else i - int(round(float(delay_fn(i / SR))))
        cleaned = canceller.process(mic[i:i + BLOCK], ref_window(ref, start, BLOCK))
        if cleaned.size:
            take = min(len(cleaned), len(out) - i)
            out[i:i + take] = np.asarray(cleaned[:take], dtype=np.float64)
    return out


def _db(x) -> float:
    return 10.0 * float(np.log10(max(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)), 1e-20)))


def _erle(mic, cleaned, lo, hi) -> float:
    band = slice(int(lo * SR), int(hi * SR))
    return _db(mic[band]) - _db(cleaned[band])


def _make(seed: int, delay_fn, seconds: float, user_from: float) -> tuple[np.ndarray, np.ndarray]:
    ref = broadband(int(seconds * SR), seed)
    mic = np.convolve(echo_at(ref, delay_fn), room(seed + 1, 0.02))[:len(ref)]
    # The user speaks over the tail: the canceller must keep her while cancelling the echo.
    rng = np.random.default_rng(seed + 2)
    user = np.zeros(len(ref))
    start = int(user_from * SR)
    user[start:] = 0.15 * rng.standard_normal(len(ref) - start)
    return ref, (mic + user).astype(np.float64)


# --------------------------------------------------------------------------- SpeexDSP

def test_speex_locked_cancels_deeper_than_unlocked():
    """A bulk delay past the MDF span: the unlocked MDF cannot reach it; the locked (re-timed)
    reference brings the echo inside the span, so the MDF cancels it."""
    if not modes.speex_available()[0]:
        pytest.skip("libspeexdsp not importable: " + modes.speex_available()[1])
    delay = lambda t: 1600.0  # 100 ms (1600 samples): far past the 480-tap (30 ms) span
    ref, mic = _make(1, delay, 3.0, 2.0)
    far = (0.0, 1.5)

    def erle(delayed):
        canceller = modes.SpeexCanceller(SR, taps=480)
        try:
            cleaned = _process(canceller, mic, ref, delay if delayed else None)
        finally:
            canceller.close()
        return _erle(mic, cleaned, *far)

    unlocked, locked = erle(False), erle(True)
    assert locked - unlocked > 10.0, (unlocked, locked)
    assert locked > 15.0, locked


# --------------------------------------------------------------------------- AEC3 wheel

def test_aec3_locked_converges_faster_than_unlocked():
    """A path change (drift): the unlocked wheel must re-search its own delay across the step;
    the locked arm already carries the new delay, so it is cancelling through the step."""
    if not modes.aec3_locked_available()[0]:
        pytest.skip("aec3 wheel unavailable: " + modes.aec3_locked_available()[1])
    step_t = 2.0
    delay = lambda t: np.where(t < step_t, 1600.0, 2600.0)  # +1000-sample step at step_t
    ref, mic = _make(2, delay, 4.0, 3.4)
    post = (step_t + 0.2, step_t + 0.7)  # the re-search window the unlocked wheel is still in

    def erle(delayed):
        canceller = Aec3Canceller(double_talk_guard=False)
        try:
            cleaned = _process(canceller, mic, ref, delay if delayed else None)
        finally:
            canceller.close()
        return _erle(mic, cleaned, *post)

    unlocked, locked = erle(False), erle(True)
    assert locked - unlocked > 10.0, (unlocked, locked)
    assert locked > 20.0, locked


# --------------------------------------------------------------------------- Voxa NLMS

def test_voxa_locked_converges_faster_than_unlocked():
    """The NLMS span (4096 taps) already reaches the bulk delay, so the win is convergence:
    reading the reference at the locked offset starts the filter aligned instead of searching."""
    delay = lambda t: 1600.0  # 100 ms, still inside the 4096-tap (256 ms) NLMS span
    ref, mic = _make(3, delay, 3.0, 2.0)
    conv = (0.4, 1.2)  # just past the echo onset, before an unlocked filter has converged

    def erle(locked):
        canceller = EchoCanceller(sample_rate=SR, taps=4096, mu=0.7)
        if locked:
            canceller.set_offset(1600, 0.0, 1600.0, 0.0, 1600.0)
        cleaned = _process(canceller, mic, ref)  # set_offset already carries the bulk delay
        return _erle(mic, cleaned, *conv)

    unlocked, locked = erle(False), erle(True)
    assert locked - unlocked > 8.0, (unlocked, locked)
    assert locked > 15.0, locked


# --------------------------------------------------------------------------- confidence fields

@pytest.mark.parametrize("arm", ("voxa-locked", "aec3-locked", "speex-locked"))
def test_a_locked_arm_never_crashes_on_the_aligners_confidence_fields(arm):
    """The runner hands each arm the aligner's locked delay + confidence per frame (``set_offset``
    fractional for the NLMS, a ring offset for the MDF/wheel). Before the first lock the aligner
    reports ``None``: the arm must be fed nothing and simply pass the microphone through."""
    if arm in ("aec3-locked",) and not modes.aec3_locked_available()[0]:
        pytest.skip("aec3 wheel unavailable")
    if arm == "speex-locked" and not modes.speex_available()[0]:
        pytest.skip("libspeexdsp not importable")
    ref, mic = _make(4, lambda t: 1600.0, 3.0, 2.0)
    aligner = ExactAligner()
    ref_offset_lock = arm in ("aec3-locked", "speex-locked")
    canceller = (EchoCanceller(sample_rate=SR, taps=4096, mu=0.7) if arm == "voxa-locked"
                 else Aec3Canceller(double_talk_guard=False) if arm == "aec3-locked"
                 else modes.SpeexCanceller(SR, taps=1024))
    try:
        for lo in range(0, len(ref), BLOCK):
            hi = lo + BLOCK
            aligner.observe(hi / SR, mic[lo:hi], ref[lo:hi])
            locked = aligner.locked_delay
            # the confidence fields the report reads must all be finite numbers / bools
            assert np.isfinite(aligner.confidence) and isinstance(aligner.refined, bool)
            if locked is None:
                continue  # the runner applies no delay before the first lock
            if ref_offset_lock:
                # the arm is given the bulk delay as a reference-ring offset (its own search then
                # only refines the room tail); set_offset is not how these libraries take a delay.
                rc = ref_window(ref, lo - int(round(locked)), BLOCK)
                out = canceller.process(mic[lo:hi], rc)
            else:
                canceller.set_offset(int(round(locked)), 0.0, locked, 0.0, locked)
                out = canceller.process(mic[lo:hi], ref[lo:hi])
            assert np.all(np.isfinite(np.asarray(out, dtype=np.float64)))
    finally:
        getattr(canceller, "close", lambda: None)()
