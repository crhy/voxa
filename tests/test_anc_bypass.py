"""Q1 (issue #88): the microphone must be untouched when the speakers are silent, and the user's
voice must never be cancelled. Each required behaviour has a test here."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals  # noqa: E402

from voxa.anc import EchoCanceller, erle_db, estimate_delay

SR = 16000


def feed(canceller, mic, ref, chunk=160):
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    return np.concatenate(outs)


def _learned(canceller, ref, echo, delay):
    """Warm the filter up on 2 s of clean echo so it has a real learned filter to lose."""
    feed(canceller, echo[:2 * SR], ref[:2 * SR], 160)


# ---------------------------------------------------------------------------
# 1. Reference silent: output bit-identical to the mic, filter does not adapt,
#    no state lost (sound returns -> cancels at once from the learned filter).
# ---------------------------------------------------------------------------


def test_silent_reference_is_bit_identical_to_mic():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    w_before = c.weights.copy()
    mic = (echo + speech).astype(np.float32)
    silent = np.zeros(160, dtype=np.float32)
    for _ in range(20):  # a block of near-silence right after loud playback
        out = c.process(mic[:160], silent)
        assert np.array_equal(out, mic[:160].astype(np.float32))
        assert np.array_equal(c.weights, w_before)


def test_silent_reference_below_60dbfs_is_bit_identical():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    mic = (echo + speech).astype(np.float32)
    rng = np.random.default_rng(7)
    faint = (1e-4 * rng.standard_normal(160)).astype(np.float32)  # ~ -80 dBFS
    for _ in range(10):
        out = c.process(mic[:160], faint)
        assert np.array_equal(out, mic[:160].astype(np.float32))


def test_silent_reference_does_not_adapt_filter():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    w_before = c.weights.copy()
    silent = np.zeros(160, dtype=np.float32)
    mic = (echo + speech).astype(np.float32)
    for _ in range(50):
        c.process(mic[:160], silent)
    assert np.array_equal(c.weights, w_before)


def test_no_state_lost_when_sound_returns():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    before = c.converged_db
    silent = np.zeros(160, dtype=np.float32)
    mic = (echo + speech).astype(np.float32)
    for _ in range(50):  # ~0.5 s of silence
        c.process(mic[:160], silent)
    # music returns: it must cancel at once from the learned filter
    out = feed(c, echo[:SR], ref[:SR], 160)
    assert erle_db(echo[:SR], out) >= 25.0
    assert c.converged_db >= before


# ---------------------------------------------------------------------------
# 2. Near-end speech with a silent reference: unchanged. Near-end speech DURING
#    playback (double talk): adaptation freezes, speech survives (<1 dB loss),
#    echo still reduced by >=15 dB.
# ---------------------------------------------------------------------------


def test_near_end_speech_with_silent_reference_unchanged():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    mic = speech.astype(np.float32)  # near-end speech, no echo at all
    out = feed(c, mic, np.zeros_like(mic), 160)
    assert np.array_equal(out, mic)


def test_double_talk_speech_survives_under_1db():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    mic = (echo + speech).astype(np.float32)
    cleaned = feed(c, mic, ref, 160)
    mask = anc_signals.speech_mask(speech)
    p_cleaned = float(np.mean(cleaned[mask] ** 2))
    p_speech = float(np.mean(speech[mask] ** 2))
    loss = 10.0 * np.log10(p_cleaned / p_speech)
    assert abs(loss) < 1.0
    corr = float(np.corrcoef(cleaned[mask], speech[mask])[0, 1])
    assert corr >= 0.98


def test_double_talk_echo_still_reduced_15db():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    mic = (echo + speech).astype(np.float32)
    cleaned = feed(c, mic, ref, 160)
    mask = anc_signals.speech_mask(speech)
    tail = slice(2 * SR, None)
    sel = ~mask[tail]
    assert erle_db(mic[tail][sel], cleaned[tail][sel]) >= 15.0


def test_double_talk_freezes_adaptation():
    """During double talk the filter must not be dragged toward the user's voice: a clean-echo
    block right after a speech burst must still cancel as well as before."""
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    mic = (echo + speech).astype(np.float32)
    cleaned = feed(c, mic, ref, 160)
    # the second burst ends near 4.9 s; check the clean stretch before the third (6.1 s)
    start = int(5.2 * SR)
    stop = int(5.8 * SR)
    assert erle_db(mic[start:stop], cleaned[start:stop]) >= 25.0


# ---------------------------------------------------------------------------
# 3. The canceller never turns off: after 60 s of silence and then music, ERLE
#    within 1 s is at least what it was before.
# ---------------------------------------------------------------------------


def test_never_turns_off_after_60s_silence():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    before = c.converged_db
    silence = np.zeros(60 * SR, dtype=np.float32)
    feed(c, silence, silence, 160)
    out = feed(c, echo[:SR], ref[:SR], 160)
    assert erle_db(echo[:SR], out) >= before


def test_never_turns_off_silence_then_double_talk():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    _learned(c, ref, echo, d)
    before = c.converged_db
    silence = np.zeros(60 * SR, dtype=np.float32)
    feed(c, silence, silence, 160)
    mic = (echo + speech).astype(np.float32)
    out = feed(c, mic[:SR], ref[:SR], 160)
    mask = anc_signals.speech_mask(speech[:SR])
    tail = slice(int(0.5 * SR), None)
    sel = ~mask[tail]
    assert erle_db(mic[:SR][tail][sel], out[tail][sel]) >= 15.0
    assert c.converged_db >= before


# ---------------------------------------------------------------------------
# 4. Output is never louder than the microphone (existing rule) and never NaN.
# ---------------------------------------------------------------------------


def test_output_never_louder_than_mic():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    mic = (echo + speech).astype(np.float32)
    cleaned = feed(c, mic, ref, 160)
    assert erle_db(mic, cleaned) >= -0.5


def test_output_never_nan():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    mic = (echo + speech).astype(np.float32)
    cleaned = feed(c, mic, ref, 160)
    assert np.all(np.isfinite(cleaned))
    # a pathological reference (huge spikes) must not poison the output either
    rng = np.random.default_rng(3)
    wild = (1e6 * rng.standard_normal(len(mic))).astype(np.float32)
    wild_out = feed(EchoCanceller(delay=d), mic, wild, 160)
    assert np.all(np.isfinite(wild_out))
