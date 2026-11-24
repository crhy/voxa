"""Hearing 2 (task h04): the DTLN-aec live echo stage - available, real-time, and correct.

The stage runs the shipped ONNX weights under voxa/data/dtln with onnxruntime. The tests are offline:
no microphone, no playback. If onnxruntime or the weights are not here the whole module skips, exactly
like the aec3 tests skip without pywebrtc-audio - and test_dtln_falls_back below proves the app then
still runs Voxa's own canceller instead of crashing.
"""

from __future__ import annotations

import time

import anc_signals
import numpy as np
import pytest

from voxa import config, dtln
from voxa.audio import AudioCapture

if not dtln.dtln_available()[0]:
    pytest.skip("DTLN stage unavailable: " + dtln.dtln_available()[1], allow_module_level=True)

SR = dtln.SR
FRAME_SHIFT = dtln.BLOCK_SHIFT  # 128 samples = 8 ms of the 16 kHz stream


def test_the_stage_is_available_and_says_what_it_runs():
    ok, why = dtln.dtln_available()
    assert ok, why
    assert "onnxruntime" in why and "DTLN-aec" in why


def test_new_profiles_default_to_the_neural_stage(monkeypatch):
    """config default is dtln; an unknown/legacy echo_mode still normalises to Voxa's own stage."""
    assert config.Settings().echo_mode == "dtln"
    assert "dtln" in config.ECHO_MODES
    assert config.Settings(echo_mode="not-a-mode").normalized().echo_mode == "voxa"


def test_falls_back_to_voxa_canceller_when_the_stage_cannot_run(monkeypatch):
    """dtln is the default but must never crash a machine without onnxruntime/weights (header rule)."""
    monkeypatch.setattr("voxa.audio.dtln_available", lambda: (False, "weights missing (test)"))
    cap = AudioCapture()
    cap.set_echo_mode("dtln")
    from voxa.anc import EchoCanceller
    assert isinstance(cap.canceller, EchoCanceller)


def _feed(canceller, mic, ref, chunk):
    out = []
    for i in range(0, len(mic), chunk):
        out.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    # flush the reconstruction tail
    out.append(canceller.process(np.zeros(0, "float32"), np.zeros(0, "float32")))
    return np.concatenate(out)


def test_streaming_is_chunk_size_invariant():
    """A live canceller sees whatever chunk sizes the audio server hands it; the output must not
    depend on that. Feeding the stream one chunk at a time equals feeding it in 320-sample pieces."""
    ref, echo, speech, _delay = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    one = _feed(dtln.DtlnCanceller(SR), mic, ref, len(mic))
    pieces = _feed(dtln.DtlnCanceller(SR), mic, ref, 320)
    n = min(len(one), len(pieces))
    assert n > 0
    assert np.allclose(one[:n], pieces[:n], atol=1e-3), (
        f"chunk-dependent output: max diff {np.max(np.abs(one[:n] - pieces[:n]))}")


def test_process_returns_one_sample_per_input_sample():
    ref, echo, speech, _delay = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    c = dtln.DtlnCanceller(SR)
    got = c.process(mic, ref)
    assert got.shape == mic.shape
    assert np.all(np.isfinite(got))


def test_removes_a_synthetic_room_echo():
    """On broadband music + a 37 ms delayed room, the stage drives the residual well below the
    echo it was handed (the same energy-ratio the bench ERLE quotes)."""
    ref, echo, speech, _delay = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    c = dtln.DtlnCanceller(SR)
    cleaned = _feed(c, mic, ref, 320)
    n = min(len(mic), len(cleaned))
    mic_pow = float(np.mean(mic[:n] ** 2))
    res_pow = float(np.mean(cleaned[:n] ** 2))
    assert mic_pow > 1e-6, "the synthetic echo must be loud before cancellation"
    assert res_pow < 0.5 * mic_pow, (
        f"no echo removed: residual {10*np.log10(res_pow+1e-20):.1f} dB vs mic "
        f"{10*np.log10(mic_pow+1e-20):.1f} dB")


def test_per_block_latency_stays_under_the_frame_shift():
    """Step 4: the stage must never fall behind - one block (stage 1 + stage 2) has to finish in less
    than the 8 ms the stream advances per block, p99. Reported in docs/hearing/H04-DTLN.md."""
    c = dtln.DtlnCanceller(SR)
    rng = np.random.default_rng(0)
    lat = []
    try:
        for _ in range(400):
            block = rng.standard_normal(SR).astype("float32")  # 1 s of stream = many blocks
            t0 = time.perf_counter()
            c.process(block, block)
            lat.append((time.perf_counter() - t0) * 1000.0 / (SR / dtln.BLOCK_SHIFT))
    finally:
        c.close()
    lat = np.array(lat)
    p50 = float(np.percentile(lat, 50))
    p99 = float(np.percentile(lat, 99))
    frame_shift_ms = dtln.BLOCK_SHIFT / SR * 1000.0  # 8.0 ms
    print(f"DTLN per-block latency p50={p50:.2f}ms p99={p99:.2f}ms (frame shift {frame_shift_ms:.1f}ms)")
    assert p99 < frame_shift_ms, f"p99 {p99:.2f} ms exceeds the {frame_shift_ms:.1f} ms frame shift"
