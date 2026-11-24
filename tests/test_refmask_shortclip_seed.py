"""f190 (roadmap task 4 on the SHORT echo-dominated clips): give the LockedResidualMask a
finite per-band residual prediction on short clips.

The measured blocker (f185 notes): on a short clip the linear canceller cancels the echo so
well that the per-band output-vs-reference ratio the mask learns from its OWN output collapses
to ~0, so `coupling` stays 0, `echo_b` is 0 in every band, and the per-band output-over-
prediction margin the joint double-talk gate needs to tell her echo from a second source
(``band_margin_db``) is ``-inf`` at every barge-in-confirm - the guard can never agree it is
echo. f190 primes that prediction from the utterance onset (the window the ExactDelayLock
converged on, f168/f171), read at the delay the lock handed the mask.

Small synthetic signals only: a short broadband reference, a canceller that leaves real residue
during the onset and ~none after, and the user's voice arriving clearly over the prediction.
"""

from __future__ import annotations

import numpy as np

from voxa.refmask import LockedResidualMask

SR = 16000


def _ref(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal(n)
    spec = np.fft.rfft(ref)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    ref = ref / np.max(np.abs(ref))
    return (0.4 * ref).astype(np.float64)


def _short_echo_clip(n: int, delay: int = 100, resid_s: float = 0.5):
    """A short echo-dominated clip: a loud broadband reference, the canceller leaving real
    residue only over the first ``resid_s`` (as it converges), then near-perfect cancellation
    (echo bands ~0) - the condition that collapses the learned per-band ratio to 0."""
    ref = _ref(n)
    mic = np.zeros(n, dtype=np.float64)
    resid_end = int(resid_s * SR)
    shifted = np.zeros(n)
    shifted[delay:] = ref[: n - delay]
    mic[:resid_end] = 0.3 * shifted[:resid_end]
    return mic, ref


def test_short_clip_gets_a_finite_band_prediction_after_the_lock():
    """Without f190's seeding the mask learns ~0 ratios on this clip and ``band_margin_db``
    stays -inf (f185's measured blocker). Handed the locked delay, the onset primes a finite
    per-band prediction so ``band_margin_db`` is a real (finite) number."""
    n = int(1.2 * SR)
    mic, ref = _short_echo_clip(n)

    # control: no delay handed -> the mask never primes, prediction collapses, no claim.
    plain = LockedResidualMask()
    plain.process(mic, ref)
    assert plain._prime_coupling.max() == 0.0
    assert not np.isfinite(plain.band_margin_db)

    seeded = LockedResidualMask()
    seeded.set_delay(100)
    seeded.process(mic, ref)
    assert seeded._prime_coupling.max() > 0.0, "onset did not prime a prediction floor"
    assert np.isfinite(seeded.band_margin_db), "band_margin_db still -inf on the short clip"


def test_band_over_the_predicted_residual_by_6db_is_protected():
    """The >= 6 dB protection rule still holds (the seeding rides only the margin EVIDENCE,
    never the frame gain): a band whose output stands clearly above the predicted residual is
    the near-end user, not residual echo, and goes through untouched."""
    n = int(1.2 * SR)
    mic, ref = _short_echo_clip(n)
    # the user arrives well after the onset residue, over a band the model predicted quiet:
    # her voice stands far above the (primed) residual prediction, so it is protected.
    tone_start = int(0.95 * SR)
    tone = np.zeros(n)
    tt = np.arange(n - tone_start) / SR
    tone[tone_start:] = 0.5 * np.sin(2.0 * np.pi * 2000.0 * tt)
    mask = LockedResidualMask()
    mask.set_delay(100)
    out = mask.process((mic + tone).astype(np.float64), ref)
    lo, hi = tone_start + int(0.05 * SR), n  # the tail of the tone, away from its attack
    held = float(np.mean(out[lo:hi] ** 2)) / max(float(np.mean(tone[lo:hi] ** 2)), 1e-20)
    assert held >= 0.5, held  # the user's word is not deleted just because the mask now predicts echo


def test_near_end_only_never_primes_and_ne_suppresses():
    """A clip nobody locked (near-end only, silent reference) must not prime, must not claim a
    prediction, and must pass the user through untouched (evidence, not deletion)."""
    rng = np.random.default_rng(5)
    user = (0.3 * rng.standard_normal(2 * SR)).astype(np.float64)
    silent = np.zeros(2 * SR)
    mask = LockedResidualMask()
    out = mask.process(user, silent)
    assert np.array_equal(out, user.astype(np.float32))
    assert mask._prime_coupling.max() == 0.0
    assert not np.isfinite(mask.band_margin_db)
