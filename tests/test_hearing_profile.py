"""The hearing profile saved per speaker/microphone pair (issue #72, saved-profile half).

Everything runs on synthetic numpy pairs through the offline capture object: no GStreamer, no sound
server, no recordings, and no volume is touched. What must hold is that Voxa starts from the filter she
learned for this pair, that a filter belonging to another pair or volume is not used, and that starting
from the saved filter is never worse than starting blank.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals  # noqa: E402

from voxa import anccal
from voxa.anc import EchoCanceller, erle_db, estimate_delay
from voxa.audio import REFERENCE_RATE, REFERENCE_SECONDS, AudioCapture, ReferenceRing

SR = 16000
SINK = "alsa_output.pci-0_0.monitor"
MIC = "alsa:7C:device_2"


def feed(canceller, mic, ref, chunk=160):
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    return np.concatenate(outs)


def pcm16(x) -> bytes:
    samples = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 32767.0), -32768, 32767)
    return samples.astype("<i2").tobytes()


def pair():
    """(ref, mic, delay) for one synthetic speaker/room/microphone path.

    The microphone carries only what the computer plays, as in a calibration recording: the residual
    is what she can still hear of herself.
    """
    ref, echo, _speech, delay = anc_signals.make()
    return ref, echo, max(0, estimate_delay(echo, ref) - 256)


def learned_state(ref, mic, delay):
    """A canceller warmed up on the pair, as a saved profile would be."""
    canceller = EchoCanceller(delay=delay)
    feed(canceller, mic, ref, 160)
    return canceller


def test_profile_path_is_per_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    a = anccal.profile_path(SINK, MIC)
    b = anccal.profile_path(SINK, "alsa:7C:device_9")
    assert a != b
    assert a.parent == tmp_path / "voxa" / "hearing"
    assert anccal.profile_path("weird name/with spaces", MIC).name == "weird_name_with_spaces__alsa_7C_device_2.json"


def test_saved_profile_carries_the_pair_and_the_level(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    canceller = learned_state(ref, mic, delay)
    path = anccal.save_live_state(canceller.export_state(), delay, SINK, MIC, 0.3)
    profile = json.loads(path.read_text())
    assert set(profile) == {"delay", "state", "sink", "source", "level", "saved_at"}
    assert profile["delay"] == delay
    assert profile["sink"] == SINK and profile["source"] == MIC
    assert profile["level"] == 0.3 and profile["saved_at"]
    assert len(profile["state"]["weights"]) == 1024


def test_apply_uses_the_saved_filter_and_its_delay(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    anccal.save_live_state(learned_state(ref, mic, delay).export_state(), delay, SINK, MIC, 0.3)
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    assert capture.apply_hearing_profile(SINK, MIC) is True
    assert capture.hearing_profile_applied is True
    assert capture.canceller._delay == delay
    assert np.max(np.abs(capture.canceller.weights)) > 0.0


def test_profile_of_another_pair_is_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    anccal.save_live_state(learned_state(ref, mic, delay).export_state(), delay, SINK, MIC, 0.3)
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    assert capture.apply_hearing_profile(SINK, "alsa:7C:device_9") is False
    assert capture.hearing_profile_applied is False
    assert capture.canceller.weights is None
    assert anccal.load_profile(SINK, "alsa:7C:device_9") is None


def test_profile_learned_at_a_very_different_volume_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    anccal.save_live_state(learned_state(ref, mic, delay).export_state(), delay, SINK, MIC, 0.4)
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    assert capture.apply_hearing_profile(SINK, MIC, level=0.01) is False
    assert capture.hearing_profile_stale is True
    assert capture.canceller.weights is None
    # The same pair at a similar volume is still current.
    assert capture.apply_hearing_profile(SINK, MIC, level=0.3) is True


def test_profile_is_current_bounds():
    profile = {"delay": 100, "state": {}, "sink": SINK, "source": MIC, "level": 0.4}
    assert anccal.profile_is_current(profile, SINK, MIC) is True            # volume unknown
    assert anccal.profile_is_current(profile, SINK, MIC, 0.4) is True
    assert anccal.profile_is_current(profile, SINK, MIC, 0.25) is True      # about one stop up
    assert anccal.profile_is_current(profile, SINK, MIC, 0.8) is True       # one stop up
    assert anccal.profile_is_current(profile, SINK, MIC, 0.02) is False     # far quieter
    assert anccal.profile_is_current(profile, SINK, MIC, 4.0) is False      # far louder
    assert anccal.profile_is_current(profile, "other.sink.monitor", MIC) is False
    assert anccal.profile_is_current(None, SINK, MIC) is False
    # A profile written by an older build: no pair, no level, still usable for the pair it is read from.
    assert anccal.profile_is_current({"delay": 0, "state": {}}, SINK, MIC, 0.3) is True


def test_old_shape_profile_still_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    state = learned_state(ref, mic, delay).export_state()
    anccal.profile_path(SINK, MIC).parent.mkdir(parents=True)
    anccal.profile_path(SINK, MIC).write_text(json.dumps({"delay": delay, "state": state}))
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    assert capture.apply_hearing_profile(SINK, MIC) is True
    assert np.max(np.abs(capture.canceller.weights)) > 0.0


def test_starting_from_the_saved_filter_is_never_worse(tmp_path, monkeypatch):
    """The regression the issue asks for: a filter change must not make the residual worse."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    anccal.save_live_state(learned_state(ref, mic, delay).export_state(), delay, SINK, MIC, 0.3)
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    capture.apply_hearing_profile(SINK, MIC)
    from_saved = feed(capture.canceller, mic, ref, 160)
    from_blank = feed(EchoCanceller(), mic, ref, 160)
    assert erle_db(mic, from_saved) >= erle_db(mic, from_blank) - 0.5
    assert erle_db(mic, from_saved) >= 25.0
    # The saved filter also survives a restart in the middle of the pair.
    restarted = AudioCapture()
    restarted.canceller = EchoCanceller()
    restarted.apply_hearing_profile(SINK, MIC)
    assert erle_db(mic, feed(restarted.canceller, mic, ref, 160)) >= erle_db(mic, from_saved) - 0.5


def test_stop_saves_what_the_live_canceller_learned(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    capture = AudioCapture()
    capture.canceller = EchoCanceller(delay=delay)
    feed(capture.canceller, mic[:2 * SR], ref[:2 * SR], 160)
    capture.hearing_sink, capture.hearing_microphone = SINK, MIC
    capture.hearing_level = 0.25
    capture.stop()
    profile = anccal.load_profile(SINK, MIC)
    assert profile is not None
    assert profile["delay"] == delay
    assert len(profile["state"]["weights"]) == 1024
    assert profile["level"] == 0.25
    # And the next session starts from it.
    fresh = AudioCapture()
    fresh.canceller = EchoCanceller()
    assert fresh.apply_hearing_profile(SINK, MIC) is True


def test_stop_keeps_a_filter_that_learned_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    capture.hearing_sink, capture.hearing_microphone = SINK, MIC
    capture.stop()
    assert anccal.load_profile(SINK, MIC) is None


def test_stop_survives_a_canceller_that_is_not_ours(tmp_path, monkeypatch):
    """The capture object is also driven by fakes and by webrtc mode; saving must never break stopping."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    capture = AudioCapture()
    capture.canceller = object()
    capture.hearing_sink, capture.hearing_microphone = SINK, MIC
    capture.stop()
    assert anccal.load_profile(SINK, MIC) is None


def _feed_delay_estimation(capture, mic, ref, chunk=1600):
    """Drive _maybe_estimate_delay the way the capture callbacks do."""
    capture._ref_ring = ReferenceRing(len(ref))
    capture._ref_ring.append(ref)
    capture._ref_anchor = 0
    capture._mic_counter = 0
    capture._delay_done = False
    capture._delay_mic = []
    capture._delay_ref = []
    pcm = pcm16(mic)
    step = chunk * 2
    for i in range(0, len(pcm), step):
        capture._maybe_estimate_delay(pcm[i:i + step])
        capture._mic_counter += chunk


def test_remeasuring_the_delay_keeps_the_learned_filter(tmp_path, monkeypatch):
    """A re-measured delay must not throw the filter away: she keeps adapting from what she has."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, _delay = pair()
    span = REFERENCE_SECONDS * REFERENCE_RATE
    capture = AudioCapture()
    capture.canceller = EchoCanceller()
    capture.canceller.load_state(learned_state(ref[:span], mic[:span], 0).export_state())
    before = capture.canceller.weights.copy()
    _feed_delay_estimation(capture, mic[:span], ref[:span])
    assert capture._delay_done is True
    assert capture.canceller._delay > 0
    assert np.array_equal(capture.canceller.weights, before)


def test_set_echo_mode_voxa_starts_from_the_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    ref, mic, delay = pair()
    anccal.save_live_state(learned_state(ref, mic, delay).export_state(), delay, SINK, MIC, 0.3)
    capture = AudioCapture()
    capture.hearing_microphone = MIC
    capture.set_echo_mode("voxa", SINK)
    assert capture.hearing_profile_applied is True
    assert capture.canceller._delay == delay
    assert np.max(np.abs(capture.canceller.weights)) > 0.0
