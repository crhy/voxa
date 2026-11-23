"""Hearing 2 (task h01): the SpeexDSP bench arm, on synthetic signals - fast, offline, no sound.

The library under test is the system libspeexdsp (the MDF echo canceller); when it is not on this
machine the whole module skips, exactly like the aec3 tests skip without pywebrtc-audio.
"""

from __future__ import annotations

import anc_signals
import numpy as np
import pytest

from voxa.bench import modes

if not modes.speex_available()[0]:
    pytest.skip("libspeexdsp not importable: " + modes.speex_available()[1], allow_module_level=True)


def _run(canceller, mic, ref, frame=modes.SpeexCanceller.BLOCK):
    out = np.zeros_like(mic)
    for i in range(0, len(mic) - frame + 1, frame):
        cleaned = canceller.process(mic[i:i + frame], ref[i:i + frame])
        out[i:i + len(cleaned)] = cleaned[:len(out) - i]
    return out


def _db(x) -> float:
    return 10.0 * float(np.log10(max(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)), 1e-20)))


def test_speex_reduces_a_synthetic_room_echo():
    ref, echo, speech, delay = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    canceller = modes.SpeexCanceller(16000, taps=2048)
    try:
        cleaned = _run(canceller, mic, ref)
    finally:
        canceller.close()
    # The MDF filter has to earn its echo removal on the far-end-only part of the clip.
    far_end = np.zeros(len(mic), dtype=bool)
    far_end[:int(2.0 * 16000)] = True
    erle = _db(mic[far_end]) - _db(cleaned[far_end])
    assert erle > 6.0, f"SpeexDSP MDF removed only {erle:.1f} dB"


def test_speex_keeps_the_user_who_talks_over_the_echo():
    ref, echo, speech, delay = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    canceller = modes.SpeexCanceller(16000, taps=2048)
    try:
        cleaned = _run(canceller, mic, ref)
    finally:
        canceller.close()
    spoken = anc_signals.speech_mask(speech)
    # The user is still there: the cleaned stream keeps most of her energy where she spoke.
    kept = _db(cleaned[spoken]) - _db(mic[spoken])
    assert kept > -6.0, f"SpeexDSP deleted the user by {-kept:.1f} dB"
    # ...and the echo it removed on the far-end-only region does not come back as residual.
    far_end = np.zeros(len(mic), dtype=bool)
    far_end[:int(2.0 * 16000)] = True
    assert _db(cleaned[far_end] - speech[far_end]) < _db(mic[far_end] - speech[far_end]) - 3.0


def test_speex_reports_the_settings_this_library_build_reaches():
    canceller = modes.SpeexCanceller(16000, taps=1024, alpha=0.1)
    try:
        assert canceller.settings["settings_reachable"]["alpha"] is True
        assert canceller.settings["settings_reachable"]["max_filter_length"] is True
        # Measured on this machine: these requests are NOT in this build's echo_ctl switch, and
        # request 29 (upstream TAU) corrupts the state, so the arm never sends them.
        assert "double_talk" in canceller.settings["settings_unreachable"]
        assert "tau" in canceller.settings["settings_unreachable"]
        assert canceller.delay_ctl_supported is False
        assert canceller.continuous is True
        assert canceller.filter_span_ms == pytest.approx(1024 / 16000 * 1000.0)
    finally:
        canceller.close()


def test_speex_power_properties_are_measured_and_close_is_safe():
    ref, echo, speech, _ = anc_signals.make()
    mic = (echo + speech).astype(np.float32)
    canceller = modes.SpeexCanceller(16000, taps=1024)
    canceller.note_double_talk(20.0)
    _run(canceller, mic, ref)
    assert np.isfinite(canceller.mic_power_db)
    assert np.isfinite(canceller.residual_power_db)
    assert np.isfinite(canceller.echo_explained_db)
    assert canceller.settings["double_talk_notes"] == 1
    canceller.close()
    canceller.close()          # idempotent: the runner also closes on the way out
