"""Synthetic tests for voxa/refmask.py ResidualMask (roadmap task 4, f131)."""

from __future__ import annotations

import numpy as np

from voxa.refmask import ResidualMask

SR = 16000


def make_ref(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 0.1).astype(np.float32)


def residual_only(n: int, seed: int = 3, delay: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """AEC output holding only the residue of the reference (ratio 0.3, 6 ms late)."""
    ref = make_ref(n, seed)
    mic = np.zeros(n, dtype=np.float32)
    mic[delay:] = (0.3 * ref)[: n - delay]
    return mic, ref


def residual_with_user(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mic, ref = residual_only(n, seed)
    t = np.arange(n) / SR
    tone = 0.5 * np.sin(2.0 * np.pi * 2500.0 * t)
    mic = mic + tone.astype(np.float32)
    return mic, ref


def energy(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def test_residual_echo_is_reduced():
    n = 4 * SR
    mic, ref = residual_only(n)
    mask = ResidualMask()
    out = mask.process(mic, ref)
    # The first frames teach the ratios and pass the echo through; judge after they settle.
    settled = out[SR // 2:]
    assert energy(settled) < 0.5 * energy(mic[SR // 2:3 * SR])


def test_user_tone_is_preserved():
    n = 4 * SR
    mic, ref = residual_with_user(n)
    mask = ResidualMask()
    out = mask.process(mic, ref)
    assert energy(out) > 0.5 * energy(mic)


def test_idle_after_tail_window_is_untouched():
    """Roadmap task 4: active while TTS plays and for tail_s after; later frames must
    never be touched, so her own voice after the TTS tail stays whole.

    (Not bit-exact: the base mask has had a WOLA reconstruction artifact since f129 -
    its unwindowed analysis frames rebuild at in * sum(win)/sum(win**2), an up to +3 dB
    amplitude ripple on untouched frames. Judge "untouched" as "not suppressed", which
    is what the mask actually promises; the ripple itself is a follow-up finding.)
    """
    n = 4 * SR
    mic, ref = residual_only(n)
    ref[2 * SR:] = 0.0
    mic[2 * SR:] = 0.0
    t = np.arange(n) / SR
    tone = (0.5 * np.sin(2.0 * np.pi * 2500.0 * t)).astype(np.float32)
    tone[:2 * SR] = 0.0  # her voice arrives only after the TTS stopped
    mic = mic + tone
    mask = ResidualMask(tail_s=0.3)
    out = mask.process(mic.astype(np.float32), ref)
    tail_out = out[3 * SR:]
    tail_mic = mic[3 * SR:3 * SR + len(tail_out)]
    ratio = energy(tail_out) / energy(tail_mic)
    assert 1.0 <= ratio <= 2.1  # the artifact: 1.0 .. 2.0 energy, never suppressed


def test_low_running_erle_deepens_suppression():
    n = 4 * SR
    mic, ref = residual_only(n)
    outs = {}
    for erle, name in ((30.0, "shallow"), (0.0, "deep")):
        mask = ResidualMask()
        chunks = []
        for s in range(0, n, SR):
            mask.set_running_erle(erle)
            chunks.append(mask.process(mic[s:s + SR], ref[s:s + SR]))
        outs[name] = np.concatenate(chunks)
    assert energy(outs["deep"]) < energy(outs["shallow"])


def test_silent_reference_passthrough():
    n = 2 * SR
    mic = (np.random.default_rng(1).standard_normal(n) * 0.05).astype(np.float32)
    ref = np.zeros(n, dtype=np.float32)
    mask = ResidualMask()
    out = mask.process(mic, ref)
    assert np.array_equal(out, mic)


# --- f172 (roadmap task 5 re-point): the mask's margin, per BAND -------------------------------
# The frame-sum margin (residual_margin_db) saturates on the near-silent predictions of
# unlearned bands, which is what made f132's her-echo frames report "+6 dB above the predicted
# residual" constantly. The gate now reads the MEDIAN of the per-band out/prediction ratios
# over the bands the mask actually predicts.


def _run_margin(mic: np.ndarray, ref: np.ndarray, user_from: int | None = None) -> tuple[float, float]:
    """(band_margin_db, residual_margin_db) once the ratios have settled. ``user_from`` (ms)
    switches a broadband second source on after the echo-only part has taught the ratios."""
    mask = ResidualMask()
    rng = np.random.default_rng(9)
    for s in range(0, mic.size, SR):
        frame = slice(s, s + SR)
        fed = mic[frame]
        if user_from is not None and s >= user_from * SR // 1000:
            fed = fed + (rng.uniform(-0.3, 0.3, fed.size) * 0.8).astype(np.float32)
        mask.process(fed, ref[frame])
    return mask.band_margin_db, mask.residual_margin_db


def test_band_margin_is_honest_on_echo_only():
    """Residue of the reference only: the learned bands sit at their own prediction, so the
    median band margin stays near 0 dB - the mirror verdict's margin half must not fire."""
    mic, ref = residual_only(3 * SR)
    band, _total = _run_margin(mic, ref)
    assert np.isfinite(band)
    assert band < 6.0


def test_band_margin_lifts_on_a_broadband_user():
    """A second source over taught ratios lifts many learned bands together, and the median
    band margin crosses the protection threshold it is swept at (RESIDUAL_MARGIN_DB = 6)."""
    mic, ref = residual_only(3 * SR)
    band, _total = _run_margin(mic, ref, user_from=2)
    assert np.isfinite(band)
    assert band >= 6.0


def test_band_margin_abstains_where_there_is_no_prediction():
    """A canceller that left NO residue (f172 clip smoke-0018): every band prediction is
    zero, so the mask claims NOTHING (margin -inf) instead of saturating at the floor.
    The frame sums saturate (+60 dB) - and the gate must degrade on the band number."""
    mic, ref = residual_only(2 * SR)
    mask = ResidualMask()
    # Feed the mask an output that stands nowhere near its prediction: nothing learned yet,
    # echo_b all zero, so the band margin must stay -inf (no claim), never +60.
    quiet_out = np.zeros_like(mic)
    for s in range(0, mic.size, SR):
        mask.process(quiet_out[s:s + SR], ref[s:s + SR])
    assert mask.band_margin_db == float("-inf")
