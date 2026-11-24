"""f185 (crhy/voxa#72): an echo-hijacked barge-in must not deliver Voxa's own words as the
request.

The V3 ``request_word_retention`` regression (0.0353) is an echo-hijack: her playback arrives
mid-turn, the barge-in confirms on it, and milestone A's promotion turns that pinned span into
the request once, so the user's real segment is dropped. The narrow fix reads the three shipped
hearing signals together - ExactDelayLock (f168), LockedResidualMask (f169/f171) and the
DoubleTalkGate mirror verdict (f173) - and withholds the promote ONLY when all three agree the
pinned span is her echo. Any verdict missing or indeterminate keeps the shipped promote: an
honest park beats a fake suppress (voicedetect.py's mirror rule).

Everything is offline: the gate logic is exercised on a stub with the three signals set by hand,
so no model, no network, no sound device, no sleeping.
"""

from __future__ import annotations

import types

import numpy as np

from voxa.bench.runner import ClipRunner


def _stub_runner(*, lock: bool = True, band_margin: float = -3.0,
                 echo: bool = True, margin: float = 6.0) -> ClipRunner:
    """A ClipRunner with just the three signals the discriminator reads."""
    runner = ClipRunner.__new__(ClipRunner)
    runner._lock_proven = lock
    runner._dt_echo = echo
    runner.mask = types.SimpleNamespace(band_margin_db=band_margin)
    runner.dt_gate = types.SimpleNamespace(margin_db=margin)
    return runner


def test_all_three_signals_agree_suppresses_the_pinned_span() -> None:
    # Lock proven, residual learned a band the cleaned output does NOT exceed, joint gate
    # latched echo-consistent: the pinned span is her echo, so it is withheld from promotion.
    assert _stub_runner(lock=True, band_margin=-3.0, echo=True)._pinned_candidate_is_echo()


def test_joint_verdict_none_keeps_the_promote() -> None:
    # "The joint test did not fire" is not evidence of echo (the mirror rule): no latch, keep it.
    assert not _stub_runner(lock=True, band_margin=-3.0, echo=False)._pinned_candidate_is_echo()


def test_unlearned_residual_band_keeps_the_promote() -> None:
    # band_margin_db is -inf until the mask learns a per-band prediction: indeterminate, keep it.
    assert not _stub_runner(lock=True, band_margin=float("-inf"),
                            echo=True)._pinned_candidate_is_echo()


def test_output_above_prediction_keeps_the_promote() -> None:
    # Output over the predicted residual by >= the joint margin means a second source, not echo.
    assert not _stub_runner(lock=True, band_margin=8.0,
                            echo=True, margin=6.0)._pinned_candidate_is_echo()


def test_unproven_lock_keeps_the_promote() -> None:
    # No delay lock means the residual/coherence pair was never driven at a proven delay.
    assert not _stub_runner(lock=False, band_margin=-3.0,
                            echo=True)._pinned_candidate_is_echo()
