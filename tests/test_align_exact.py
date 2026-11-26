"""Unit tests for the exact echo aligner (roadmap Hearing 2, issue #72, task h02).

Synthetic broadband signals with a KNOWN delay / drift / step, fed to
:class:`voxa.align.ExactAligner` in 20 ms chunks the way the bench runner does, so the
three pieces the roadmap names - the predictor, the sub-sample refinement and the
fractional-delay drift resampler - are each checked against ground truth.
"""

from __future__ import annotations

import numpy as np

from voxa.align import ExactAligner, ExactDelayLock
from voxa.anc import EchoCanceller

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


def test_negative_travel_locks_signed_issue_72_v11d():
    """Issue #72, task v11d: the render anchor parks the echo at a NEGATIVE travel - the live
    chirp probe measured the microphone carrying the audible map about 34 ms BEFORE the anchor
    said it was audible (-33.6..-34.6 ms over five chirps). A positive-lag search window never
    saw it, so the exact lock never confirmed live, no "echo delay:" line ever appeared, and
    DTLN stayed handed the reference at travel 0 with a +13..+32 dB residual. The search is
    signed now: the lock lands on -544 samples and proves itself there."""
    ref = broadband(2 * SR)
    delay = -0.034 * SR  # -544 samples: the microphone LEADS the map
    mic = echo_at(ref, lambda _t: delay)
    mic = np.convolve(mic, room(31))[: len(ref)]
    frames = observe(ExactAligner(), ref, mic)
    assert frames, "the negative travel must lock, not stay unsearchable"
    locked = [v for _t, v, _r in frames]
    assert abs(locked[-1] - delay) <= 12.0, (locked[-1], delay)
    assert any(r for _t, _v, r in frames), "the signed lock must be able to prove itself"


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


# ---------------------------------------------------------------------------
# 7. f163: a step is never reported CONFIDENTLY on the pre-step delay, and it snaps to
#    the post-step delay instead of sliding halfway across it (the smoke-suite tail
#    h02 measured: confident frames ~40-70 ms wrong in the reconvergence window).
# ---------------------------------------------------------------------------


def _step_run(step_samples: float, step_t: float, base: float, seconds: float, room_seed: int):
    ref = broadband(int(seconds * SR))
    mic = echo_at(ref, lambda t: np.where(t < step_t, base, base + step_samples))
    mic = np.convolve(mic, room(room_seed))[: len(ref)]
    aligner = ExactAligner()
    return aligner, observe(aligner, ref, mic)


def test_step_never_confident_on_the_pre_step_delay():
    # Once the recent window is PAST the step, the pre-step delay may never be reported
    # confidently again; the reconvergence window itself is honest (the two-path ambiguity
    # holds the lock and drops confidence).
    aligner, frames = _step_run(700.0, 2.0, 600.0, 6.0, 11)
    assert aligner.epochs >= 1
    wrong = [locked for t, locked, refined in frames
             if refined and t > 2.15 and abs(locked - 600.0) < 300.0]
    assert not wrong, wrong  # every past-the-step frame is on the NEW path
    late = [locked for t, locked, refined in frames if refined and t > 2.6]
    assert late and all(abs(locked - 1300.0) <= 6.0 for locked in late)


def test_small_step_snaps_instead_of_sliding():
    # +400 samples fits INSIDE the 60 ms window: h02 refined across it (alpha=0.5), which left
    # frames confident but 10 ms off; the two-hypothesis logic must not slide a step at all.
    aligner, frames = _step_run(400.0, 2.0, 600.0, 6.0, 12)
    sliding = [(t, locked) for t, locked, refined in frames
               if refined and 2.15 < t < 2.5 and 610.0 < locked < 990.0]
    assert not sliding, sliding
    late = [(t, locked) for t, locked, refined in frames if refined and t > 2.6]
    assert late and all(abs(locked - 1000.0) <= 6.0 for _, locked in late)


def test_step_on_a_quiet_signal_reanchors_without_lying():
    # quiet clip (SER -10 dB / -20 dB): the USER is quiet under Voxa's echo, so the mic carries
    # a near-end copy the aligner must not lock onto; it must still snap the step honestly.
    for ser_db in (-10.0, -20.0):
        aligner = ExactAligner()
        ref = broadband(6 * SR)
        mic = echo_at(ref, lambda t: np.where(t < 2.0, 350.0, 850.0))
        mic = np.convolve(mic, room(13))[: len(ref)]
        rng = np.random.default_rng(9)
        user = rng.standard_normal(len(ref))
        user *= (float(np.sqrt(np.mean(mic * mic))) / float(np.sqrt(np.mean(user * user)))
                 * 10.0 ** (ser_db / 20.0))
        frames = observe(aligner, ref, mic + user)
        # the recent (400 ms) window is fully past the step only at t > 2.4: past that, no
        # confident frame may still sit on the pre-step delay
        wrong = [locked for t, locked, refined in frames
                 if refined and t > 2.4 and abs(locked - 350.0) < 250.0]
        assert not wrong, (ser_db, wrong)
        late = [locked for t, locked, refined in frames if refined and t > 2.8]
        assert late and all(abs(locked - 850.0) <= 12.0 for locked in late), ser_db


# ---------------------------------------------------------------------------
# 8. f168: the SHIPPING lock (:class:`voxa.align.ExactDelayLock`) - the lock drives the
#    canceller while proven, the estimate keeps it pointed while NOT (visible degraded),
#    and a route/rate change is an explicit epoch.
# ---------------------------------------------------------------------------


def _feed(lock: ExactDelayLock, ref: np.ndarray, mic: np.ndarray) -> list[str]:
    """Feed mic/ref in 20 ms chunks through the lock, return the applied states."""
    states = []
    for lo in range(0, len(ref), CHUNK):
        hi = lo + CHUNK
        states.append(lock.observe(hi / SR, mic[lo:hi], ref[lo:hi]))
    return states


def test_shipping_lock_points_the_canceller_while_proven():
    ref = broadband(2 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(21))[: len(ref)]
    canceller = EchoCanceller(sample_rate=SR, taps=2048, mu=0.7)
    lock = ExactDelayLock(sample_rate=SR, canceller=canceller)
    states = _feed(lock, ref, mic)
    assert "lock" in states
    assert not lock.degraded                      # the LAST state was the lock: not degraded
    assert lock.delay_confident and lock.delay_samples > 0
    assert abs(lock.delay_samples - 500.0) <= 6.0  # the canceller got the LOCK's offset
    assert canceller.delay_samples == lock.delay_samples


def test_route_change_epoch_reanchors_the_lock_and_estimate_together():
    ref = broadband(2 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(22))[: len(ref)]
    lock = ExactDelayLock(sample_rate=SR)
    _feed(lock, ref, mic)
    assert lock.delay_confident
    lock.epoch()  # an explicit route/rate change, like voxa.audio mode switches do
    assert not lock.delay_confident               # nothing is proven since the epoch
    assert lock.delay_frac is None and lock.delay_samples == 0
    # a fresh epoch re-anchors (bounded search then refine) - it never lies about being proven
    states = _feed(lock, ref, mic)
    assert states[0] == "hold"                    # buffer too short to lock on the first frame
    assert "lock" in states
    assert abs(lock.delay_samples - 500.0) <= 6.0


def test_lock_survives_a_mid_clip_step_as_an_epoch():
    ref = broadband(6 * SR)
    mic = np.convolve(echo_at(ref, lambda t: np.where(t < 2.0, 600.0, 1500.0)),
                      room(23))[: len(ref)]
    lock = ExactDelayLock(sample_rate=SR)
    _feed(lock, ref, mic)
    assert lock.aligner.epochs >= 1               # the step re-anchored the lock
    assert lock.delay_confident                   # and re-proved itself after the step
    assert abs(lock.delay_samples - 1500.0) <= 6.0
    assert not lock.degraded                      # the lock (not the estimate) is driving


def test_fallback_estimate_is_visible_while_the_lock_is_unproven():
    # The aligner is made unable to prove the path (coherence gate shut); the canceller must
    # STILL be pointed by the shipping estimate, visibly degraded - never left blind, never
    # raw - and the lock's confidence stays False honestly.
    ref = broadband(2 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(24))[: len(ref)]
    canceller = EchoCanceller(sample_rate=SR, taps=2048, mu=0.7)
    lock = ExactDelayLock(sample_rate=SR, canceller=canceller)
    lock.aligner = ExactAligner(coherence_min=0.999)  # the lock can never prove itself
    states = _feed(lock, ref, mic)
    assert "estimate" in states and "lock" not in states
    assert lock.degraded and not lock.delay_confident
    assert abs(canceller.delay_samples - 500.0) <= 8.0  # the estimate kept it pointed


def test_confidence_fields_never_crash_on_silence_or_near_end_only():
    lock = ExactDelayLock(sample_rate=SR)
    states = _feed(lock, np.zeros(SR), np.zeros(SR))     # pure silence
    assert states and set(states) == {"hold"}
    assert lock.delay_frac is None and not lock.delay_confident
    assert lock.delay_samples == 0 and not lock.degraded
    rng = np.random.default_rng(25)
    user = rng.standard_normal(SR) * 0.1                 # the user speaks, no echo anywhere
    states = _feed(lock, user, user)
    assert set(states) <= {"hold", "estimate", "lock"}
    assert np.isfinite(lock.drift_ppm)
