"""Roadmap task 3 (issue #72): GCC-PHAT delay + clock-drift tracking, and the canceller
filter is NEVER reset between TTS sentences.

Every test here drives the delay/drift machinery with a synthetic path whose delay (and its
slope) the test sets, so the measured offset, its hysteresis, the ppm the slope implies and an
abrupt-jump epoch can be checked against the truth rather than a learned model.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals  # noqa: E402

from voxa.anc import DelayTracker, EchoCanceller, erle_db, gccphat_delay

SR = 16000


def broadband(n: int, seed: int = 0) -> np.ndarray:
    """A broadband reference (music-like): delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float32)


def room(seed: int, tail_s: float = 0.05) -> np.ndarray:
    """A room impulse response: a direct path followed by a decaying reverberant tail.

    The direct spike is what a real near/mid-field path has and what GCC-PHAT locks onto;
    a tail-only response would smear its energy and the peak would sit on the tail centroid."""
    rng = np.random.default_rng(seed)
    n = int(tail_s * SR)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.012 * SR))
    h = 0.3 * tail / np.sqrt(np.sum(tail**2))
    h[0] += 1.0  # the direct path
    return h.astype(np.float64)


def echo_at(ref: np.ndarray, delay_fn) -> np.ndarray:
    """ref delayed by ``delay_fn(t_seconds)`` samples (linear interp -> fractional lag)."""
    n = len(ref)
    t = np.arange(n) / SR
    idx = np.arange(n, dtype=np.float64) - delay_fn(t)
    return np.interp(idx, np.arange(n, dtype=np.float64), ref, left=0.0, right=0.0)


def path(seconds: int, delay0: float, drift_ppm: float = 0.0, seed: int = 1):
    """Return (ref, mic) where mic carries ref through a room at delay0 + drift*t samples."""
    n = seconds * SR
    ref = broadband(n, seed=seed)
    rate = drift_ppm / 1e6 * SR  # samples of extra delay per second
    mic = echo_at(ref, lambda t, d0=delay0, r=rate: d0 + r * t)
    mic = np.convolve(mic, room(seed + 1))[:n]
    return ref.astype(np.float32), mic.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. GCC-PHAT measures a known delay (and follows a known drift within a window).
# ---------------------------------------------------------------------------


def test_gccphat_known_delay():
    for delay in (200, 592, 1200):
        ref, mic = path(3, float(delay))
        got = gccphat_delay(mic, ref, sample_rate=SR)
        assert got is not None
        assert abs(got - delay) <= 2.0, (delay, got)


def test_gccphat_subsample_accuracy():
    # A non-integer lag: the parabolic peak refinement must land within ~0.5 sample.
    ref, mic = path(3, 592.5)
    got = gccphat_delay(mic, ref, sample_rate=SR)
    assert abs(got - 592.5) <= 0.75


def test_gccphat_silence_is_none():
    ref, mic = path(1, 400.0)
    assert gccphat_delay(np.zeros_like(mic), ref, sample_rate=SR) is None
    assert gccphat_delay(mic, np.zeros_like(ref), sample_rate=SR) is None


def test_coherence_guard_rejects_unrelated():
    # The guard behind the tracker's rejection: a microphone window that is not a copy of
    # the reference at any lag must fail the plain overlap-correlation test.
    ref, _mic = path(2, 592.0)
    rng = np.random.default_rng(9)
    noise = rng.standard_normal(len(ref)).astype(np.float32)
    assert DelayTracker._coherent(noise, ref, 592) is False
    _ref2, mic = path(2, 592.0)
    assert DelayTracker._coherent(mic, ref, 592) is True


def test_gccphat_short_window_is_none():
    ref, mic = path(1, 400.0)
    assert gccphat_delay(mic[:100], ref[:100], sample_rate=SR) is None


# ---------------------------------------------------------------------------
# 2. DelayTracker: utterance onset -> coarse offset; a drifting path -> ppm; a jump -> epoch.
# ---------------------------------------------------------------------------


def test_tracker_recovers_known_delay():
    ref, mic = path(2, 592.0)
    tr = DelayTracker(sample_rate=SR)
    first = None
    for i in range(0, len(mic), 320):
        upd = tr.observe(mic[i:i + 320], ref[i:i + 320])
        if upd is not None:
            first = upd
            break
    assert first is not None and first["kind"] == "utterance"
    assert abs(first["delay_samples"] - 592) <= 2.0, first


def test_tracker_drift_latches_ppm():
    # 200 ppm over 12 s: the estimate slope must read close to 200 ppm and latch the flag.
    ref, mic = path(12, 800.0, drift_ppm=200.0)
    tr = DelayTracker(sample_rate=SR)
    last = None
    for i in range(0, len(mic), 320):
        upd = tr.observe(mic[i:i + 320], ref[i:i + 320])
        if upd is not None:
            last = upd
    assert last is not None
    assert tr.has_clock_drift, tr.drift_ppm
    assert abs(tr.drift_ppm - 200.0) <= 40.0, tr.drift_ppm


def test_tracker_coarse_hysteresis_ignores_small_shift():
    # A path that jitters by well under the 1 ms hysteresis must not move the coarse offset.
    ref, mic = path(2, 592.0)
    tr = DelayTracker(sample_rate=SR)
    upd0 = tr.observe(mic, ref)
    first = upd0["delay_samples"]
    # feed a second window whose delay is only ~0.5 sample different: no "shift" kind.
    ref2, mic2 = path(2, first + 0.5)
    upd1 = tr.observe(mic2, ref2)
    assert upd1 is not None
    assert upd1["kind"] != "shift"
    assert tr.coarse == first


def test_tracker_epoch_on_abrupt_jump():
    # A 20 ms path change mid-stream is an epoch, confirmed by the next window, not a drift.
    ref1, mic1 = path(3, 592.0)
    ref2, mic2 = path(3, 592.0 + 0.020 * SR)
    tr = DelayTracker(sample_rate=SR)
    kinds = []
    for i in range(0, len(mic1), 320):
        upd = tr.observe(mic1[i:i + 320], ref1[i:i + 320])
        if upd is not None:
            kinds.append(upd["kind"])
    for i in range(0, len(mic2), 320):
        upd = tr.observe(mic2[i:i + 320], ref2[i:i + 320])
        if upd is not None:
            kinds.append(upd["kind"])
    assert "epoch" in kinds, kinds
    assert tr.epochs >= 1


def test_tracker_coherence_guard_rejects_near_end_dominant():
    # Mic is the user's voice, not an echo: the peak is an accident, so no estimate is given.
    ref, _mic = path(2, 592.0)
    speech, _e, _sp, _d = anc_signals.make()
    tr = DelayTracker(sample_rate=SR)
    upd = tr.observe(speech[:len(ref)], ref)
    assert upd is None


# ---------------------------------------------------------------------------
# 3. The canceller filter is KEPT between utterances: set_offset re-points the reference
#    offset without dropping weights/convergence; the silent gap never rebuilds it.
# ---------------------------------------------------------------------------


def test_set_offset_keeps_filter_state():
    ref, mic = path(2, 592.0)
    c = EchoCanceller(taps=1024, mu=0.5, delay=592)
    outs = []
    for i in range(0, len(mic), 160):
        outs.append(c.process(mic[i:i + 160], ref[i:i + 160]))
    learned = c.weights
    assert learned is not None and float(np.max(np.abs(learned))) > 0.0
    w_before = learned.copy()
    c.set_offset(592, 0.0, 0.0, 0.5, 0.0)  # re-point only
    assert c.weights is not None
    assert np.allclose(c.weights, w_before)  # filter state untouched
    assert c._offset_anchors is not None


def test_filter_not_reset_across_two_utterances():
    # One canceller instance across two TTS replies with a silent gap between: the second
    # reply must be cancelled at once from the learned filter (no per-sentence rebuild).
    ref1, mic1 = path(2, 592.0)
    ref2, mic2 = path(2, 592.0)
    c = EchoCanceller(taps=1024, mu=0.5, delay=592)

    def feed(mic, ref):
        return np.concatenate([c.process(mic[i:i + 160], ref[i:i + 160])
                               for i in range(0, len(mic), 160)])

    feed(mic1, ref1)
    w_after_first = c.weights.copy()
    # Silence: speakers off, only room noise in the mic: it must pass through untouched
    # and the filter must survive.
    rng = np.random.default_rng(4)
    quiet = (0.001 * rng.standard_normal(len(mic1))).astype(np.float32)
    silent = np.zeros(len(mic1), dtype=np.float32)
    gap = feed(quiet, silent)
    assert np.array_equal(gap, quiet)
    assert np.allclose(c.weights, w_after_first)
    # Reply two: ERLE is high immediately on the first 300 ms (learned filter carried over).
    out2 = feed(mic2, ref2)
    assert erle_db(mic2[:3000], out2[:3000]) >= 20.0
