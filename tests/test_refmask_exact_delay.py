"""f169 (roadmap task 4 on exact-delay audio): the residual stage on the LOCKED
canceller's output - voxa/refmask.LockedResidualMask + the bench's voxa-residual arm.

Small synthetic signals only: a known room tail, a double-talk onset, and the bench
ClipRunner on a known-delay clip. Fast, offline, no seed drift (fixed rng seeds).
"""

from __future__ import annotations

import numpy as np

from voxa.refmask import LockedResidualMask

SR = 16000


def _ref_words(n: int, seed: int) -> np.ndarray:
    """Broadband reference with word bursts (echo leaks are WORDS, so the reference has
    onsets and gaps like speech)."""
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal(n)
    spec = np.fft.rfft(ref)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    ref = ref / np.max(np.abs(ref))
    words = np.zeros(n)
    for w in range(5):
        s = int((0.2 + 0.5 * w) * SR)
        words[s:s + int(0.35 * SR)] = ref[s:s + int(0.35 * SR)]
    return (0.4 * words).astype(np.float64)


def _room_ir(seed: int, rt_s: float = 0.35) -> np.ndarray:
    """Room impulse response: direct spike then a RT60-like tail (the bench rooms ring
    0.2-0.8 s; the mask holds its reference model at the room's rate to predict it)."""
    n = int(rt_s * SR)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-np.log(10.0) * 6.0 * t / rt_s)
    h[0] += 1.0
    return h / np.sqrt(np.sum(h ** 2))


def _lock_inputs(delay: int = 1600, n: int = 4 * SR, seed: int = 7,
                 residue: float = 0.3, user_at: float = 1.2, user_len: float = 0.6,
                 user_level: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """One clip's (locked-canceller output, reference): the linear stage leaves ``residue``
    of the echoed reference (direct path at ``delay`` samples + a ringing room), and the
    user's word onset arrives at ``user_at`` over it (double talk)."""
    ref = _ref_words(n, seed)
    h = _room_ir(seed + 1)
    echo = np.convolve(ref, h)[: n + delay - 1]
    echo = np.concatenate([np.zeros(delay), echo])[:n]
    user = np.zeros(n)
    s = int(user_at * SR)
    rng = np.random.default_rng(seed + 2)
    burst = rng.standard_normal(int(user_len * SR))
    env = 0.5 * np.hanning(len(burst)) + 0.5
    env[: int(0.02 * SR)] = 1.0  # a hard attack: a WORD onset
    user[s:s + len(burst)] = user_level * burst * env / np.max(np.abs(burst * env))
    out = residue * echo + user
    return out.astype(np.float64), ref


def _energy(x: np.ndarray, a: float, b: float) -> float:
    lo, hi = int(a * SR), int(b * SR)
    return float(np.mean(np.asarray(x[lo:hi], dtype=np.float64) ** 2))


def test_the_rule_is_six_db_on_top_of_the_predicted_residual():
    mask = LockedResidualMask()
    assert abs(mask.protect - 10.0 ** 0.6) < 1e-9  # protect >= predicted residual + 6 dB
    assert mask.ratio_quantile_hi == 0.98  # f131: the prediction rides the residue jumps
    # the reference is read at the LOCKED delay: the mask's own lag estimate stays off
    mask.set_delay(1599.6)
    assert mask.lag == 1600 and mask._exact_lag == 1600


def test_known_room_tail_on_the_locked_output_has_its_leak_cut():
    """Roadmap task 4 on exact-delay audio: with the delay handed to the mask, the echo
    of a word that ENDED (the room still rings) is removed - the held reference decays
    at the room's own rate and those bands are suppressed, not protected."""
    out_in, ref = _lock_inputs()
    mask = LockedResidualMask()
    mask.set_delay(1600)
    out = mask.process(out_in, ref)  # one clip through the stage (the bench feeds chunks;
    # a single call keeps the emitted stream sample-aligned for the assertions below)
    window = (0.9, 1.15)  # the ring after a word: echo only, the user has not started
    cut = _energy(out_in, *window) / max(_energy(out, *window), 1e-20)
    assert cut >= 3.0, cut  # >= ~5 dB of the leak's energy removed


def test_double_talk_onset_is_held_within_budget_and_never_deleted():
    """The user's WORD ONSET (her burst attack over her own echo) must not be deleted:
    near-end attenuation across the double-talk span stays within a few dB."""
    out_in, ref = _lock_inputs()
    mask = LockedResidualMask()
    mask.set_delay(1600)
    out = mask.process(out_in, ref)
    span = (1.20, 1.50)  # her burst over the ring
    held = _energy(out, *span) / max(_energy(out_in, *span), 1e-20)
    assert held >= 0.5, held  # <= 3 dB of near-end attenuation on the onset

    onset = (1.20, 1.30)  # the first 100 ms of it: the onset itself
    held_on = _energy(out, *onset) / max(_energy(out_in, *onset), 1e-20)
    assert held_on >= 0.5, held_on


def test_quiet_and_short_user_audio_after_the_tail_is_untouched():
    """Never discard short/quiet user audio: once TTS + the tail window passed, a short
    quiet burst of hers goes through untouched (it is evidence, not deletion)."""
    out_in, ref = _lock_inputs(n=4 * SR)
    ref[3 * SR:] = 0.0
    out_in[3 * SR:] = 0.0
    rng = np.random.default_rng(11)
    burst = (0.02 * rng.standard_normal(int(0.15 * SR))).astype(np.float64)
    s = int(3.6 * SR)
    out_in[s:s + len(burst)] = burst
    mask = LockedResidualMask(tail_s=0.3)
    mask.set_delay(1600)
    out = mask.process(out_in, ref)
    tail_in = _energy(out_in, 3.7, 3.75)
    tail_out = _energy(out, 3.7, 3.75)
    assert tail_out >= 0.5 * tail_in  # not suppressed (the WOLA ripple can even raise it)


def test_no_crash_and_no_suppression_on_silence_and_confidence_fields():
    """The lock's confidence fields can stay False/None for a whole clip (near-end-only
    audio): the stage must not crash, must not suppress, and must not invent a delay."""
    rng = np.random.default_rng(5)
    user = (0.3 * rng.standard_normal(2 * SR)).astype(np.float64)
    silent = np.zeros(2 * SR)
    mask = LockedResidualMask()
    out = mask.process(user, silent)
    assert np.array_equal(out, user.astype(np.float32))  # passthrough, never deleted
    assert mask._exact_lag is None  # nobody handed the mask a delay: it never claims one
