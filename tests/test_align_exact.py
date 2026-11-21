"""Unit tests for the exact echo aligner (roadmap Hearing 2, issue #72, task h02).

Synthetic broadband signals with a KNOWN delay / drift / step, fed to
:class:`voxa.align.ExactAligner` in 20 ms chunks the way the bench runner does, so the
three pieces the roadmap names - the predictor, the sub-sample refinement and the
fractional-delay drift resampler - are each checked against ground truth.
"""

from __future__ import annotations

import numpy as np

from voxa.align import ExactAligner

SR = 16000
CHUNK = SR * 20 // 1000  # the 20 ms frame the runner observes


def broadband(n: int, seed: int = 0) -> np.ndarray:
    """A broadband reference (music-like): a delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float64)


def room(seed: int, tail_s: float = 0.05) -> np.ndarray:
    """Room impulse response: a direct spike followed by a decaying reverberant tail."""
    rng = np.random.default_rng(seed)
    n = int(tail_s * SR)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.012 * SR))
    h = 0.3 * tail / np.sqrt(np.sum(tail**2))
    h[0] += 1.0
    return h.astype(np.float64)


def echo_at(ref: np.ndarray, delay_fn) -> np.ndarray:
    """ref delayed by ``delay_fn(t_seconds)`` samples (linear interp -> fractional lag)."""
    n = len(ref)
    t = np.arange(n) / SR
    idx = np.arange(n, dtype=np.float64) - delay_fn(t)
    return np.interp(idx, np.arange(n, dtype=np.float64), ref, left=0.0, right=0.0)


def observe(aligner: ExactAligner, ref: np.ndarray, mic: np.ndarray):
    """Feed mic/ref in 20 ms chunks, return [(t_s, locked_float, refined)] for playback frames."""
    out = []
    for lo in range(0, len(ref), CHUNK):
        hi = lo + CHUNK
        t = hi / SR
        aligner.observe(t, mic[lo:hi], ref[lo:hi])
        if float(np.mean(ref[lo:hi] ** 2)) <= 1e-9:
            continue
        locked = aligner.locked_delay
        if locked is not None:
            out.append((t, locked, aligner.refined))
    return out


# ---------------------------------------------------------------------------
# 1. The predictor: a render-timeline seed lands the lock on the first playback frame.
# ---------------------------------------------------------------------------


def test_predictor_seeds_lock_without_a_search():
    ref = broadband(2 * SR)
    mic = echo_at(ref, lambda t: 400.0)
    mic = np.convolve(mic, room(3))[: len(ref)]
    aligner = ExactAligner()
    aligner.predict(400.0)  # the render timeline says the delay is ~400 samples
    frames = observe(aligner, ref, mic)
    # no 0.5 s blind search: the lock is already on the seeded delay within the first window
    assert frames[0][1] is not None
    assert abs(frames[0][1] - 400.0) <= 12.0


# ---------------------------------------------------------------------------
# 2. The drift resampler: a known clock drift is tracked, not left as residual error.
# ---------------------------------------------------------------------------


def test_drift_resampler_tracks_clock_drift():
    for ppm in (100.0, 300.0, -250.0):
        ref = broadband(4 * SR)
        rate = ppm / 1e6 * SR  # samples of extra delay per second
        mic = echo_at(ref, lambda t, r=rate: 500.0 + r * t)
        mic = np.convolve(mic, room(5))[: len(ref)]
        frames = observe(ExactAligner(), ref, mic)
        # judge the second half, once the lock has converged
        errs = [abs(locked - (500.0 + rate * t)) for t, locked, refined in frames if refined and t > 1.0]
        assert errs, (ppm, frames)
        assert max(errs) <= 3.0, (ppm, max(errs))


def test_drift_ppm_is_reported():
    ref = broadband(4 * SR)
    rate = 300.0 / 1e6 * SR
    mic = echo_at(ref, lambda t: 500.0 + rate * t)
    mic = np.convolve(mic, room(6))[: len(ref)]
    aligner = ExactAligner()
    observe(aligner, ref, mic)
    assert abs(aligner.drift_ppm - 300.0) <= 120.0


# ---------------------------------------------------------------------------
# 3. The refinement never jumps: a room reflection far from the lock cannot pull it off.
# ---------------------------------------------------------------------------


def test_refinement_holds_through_a_strong_late_reflection():
    ref = broadband(3 * SR)
    direct = echo_at(ref, lambda t: 300.0)
    late = echo_at(ref, lambda t: 1400.0)  # a late, strong path far from the direct one
    mic = (direct + 0.4 * late)[: len(ref)]
    aligner = ExactAligner()
    aligner.predict(300.0)  # the render timeline anchors the direct path
    frames = [f for f in observe(aligner, ref, mic) if f[2]]  # confident frames
    assert frames
    assert all(abs(locked - 300.0) <= 15.0 for _, locked, _ in frames)


# ---------------------------------------------------------------------------
# 4. Reconvergence: an abrupt step (a glitch) re-anchors as an epoch, then is exact again.
# ---------------------------------------------------------------------------


def test_step_reanchors_and_tracks_new_delay():
    ref = broadband(6 * SR)
    step_t = 2.0

    def delay_fn(t):
        return np.where(t < step_t, 600.0, 600.0 + 900.0)  # +900 samples at step_t

    mic = echo_at(ref, delay_fn)
    mic = np.convolve(mic, room(7))[: len(ref)]
    aligner = ExactAligner()
    frames = observe(aligner, ref, mic)
    assert aligner.epochs >= 1
    # after the echo of the new path has arrived and the buffer has refilled past the step
    # (about one search window), the lock must be exact again on the new path
    late = [(t, locked) for t, locked, refined in frames if refined and t > step_t + 0.8]
    assert late
    assert all(abs(locked - 1500.0) <= 6.0 for _, locked in late)


# ---------------------------------------------------------------------------
# 5. A first lock on a plain known delay is sub-sample accurate once refined.
# ---------------------------------------------------------------------------


def test_first_lock_is_subsample_accurate():
    for delay in (200.0, 592.5, 1200.0):
        ref = broadband(3 * SR)
        mic = echo_at(ref, lambda t, d=delay: d)
        mic = np.convolve(mic, room(int(delay)))[: len(ref)]
        frames = [locked for _, locked, refined in observe(ExactAligner(), ref, mic) if refined]
        assert frames, delay
        assert max(abs(locked - delay) for locked in frames) <= 1.0


# ---------------------------------------------------------------------------
# 6. Silence holds the lock: with no reference the delay is never reset or jumped.
# ---------------------------------------------------------------------------


def test_silence_holds_the_lock():
    ref = np.zeros(2 * SR)
    mic = np.zeros(2 * SR)
    aligner = ExactAligner()
    aligner.predict(500.0)
    for lo in range(0, len(ref), CHUNK):
        aligner.observe((lo + CHUNK) / SR, mic[lo:lo + CHUNK], ref[lo:lo + CHUNK])
    assert aligner.locked_delay == 500.0
