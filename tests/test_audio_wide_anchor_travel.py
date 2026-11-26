"""Task v11d: the wide anchor parks the reference where the exact lock cannot confirm.

On the live microphone (the v11d capture) the echo is only a few percent of the captured
energy, so the exact lock's confidence gates - built for the bench's strong echo - never
confirm, and DTLN stayed handed the reference at travel 0, 31 ms off the travel the chirp
probe measured and outside its own 32 ms learning window. The whole-reply wide energy
anchor is the honest unconfident reading; it must park the reference, not claim a lock.
"""

import io
import types
import wave

import numpy as np

from voxa import anccal
from voxa.audio import AudioCapture
from voxa.speech import pilot_wav_bytes


def _reply_pair(travel_s, seconds=2.8, rate=16000, echo=0.02, noise=8e-3, seed=5):
    """A reply's render map and a WEAK echo of it on a noisy microphone, lead/lagging by
    travel_s (negative = the microphone carries the map's samples before the anchor said
    they were audible: the live geometry)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    ref = (rng.standard_normal(n) * 0.2).astype(np.float32)
    ref[: rate // 4] = 0.0
    ref[-rate // 8:] = 0.0
    lag = int(round(travel_s * rate))
    mic = np.zeros(n, np.float32)
    if lag < 0:
        mic[:n + lag] = echo * ref[-lag:]
    else:
        mic[lag:] = echo * ref[:n - lag]
    mic = mic + rng.standard_normal(n).astype(np.float32) * noise
    return mic.astype(np.float32), ref


def _run_anchor(capture, mic, ref, chunk=3200):
    """Feed one reply frame by frame through the shipped anchor, reference aligned."""
    pos = {"i": 0}

    def fake_reference(samples):
        i = pos["i"]
        pos["i"] += samples
        window = np.zeros(samples, np.float32)
        stop = min(i + samples, ref.size)
        window[:stop - i] = ref[i:stop]
        return window

    capture._reference_window = fake_reference
    for a in range(0, mic.size, chunk):
        frame = (mic[a:a + chunk] * 32767.0).astype("<i2")
        capture._observe_wide_anchor(frame.tobytes())
        if capture._anchor["done"]:
            return
    capture._finish_wide_anchor()


def test_the_anchor_parks_the_live_negative_travel_when_the_lock_cannot():
    """Without the v11d wiring this returns None: the anchor never touched the read."""
    capture = AudioCapture()
    mic, ref = _reply_pair(-0.020)
    _run_anchor(capture, mic, ref)
    travel = capture.echo_travel_ms()
    assert travel is not None
    assert abs(travel + 20.0) <= 3.0


def test_the_anchor_parks_a_positive_travel_too():
    capture = AudioCapture()
    mic, ref = _reply_pair(+0.025)
    _run_anchor(capture, mic, ref)
    travel = capture.echo_travel_ms()
    assert travel is not None
    assert abs(travel - 25.0) <= 3.0


def test_a_confident_exact_lock_outvotes_the_anchor():
    capture = AudioCapture()
    capture._delay_lock = types.SimpleNamespace(delay_confident=True)
    mic, ref = _reply_pair(-0.020)
    _run_anchor(capture, mic, ref)
    assert capture.echo_travel_ms() is None


def test_the_anchor_converges_and_never_jumps_a_whole_window():
    """The anchor parks where the last parking left off plus its residual reading; a
    second reply whose residual lands far outside the current parking is refused."""
    capture = AudioCapture()
    capture._echo_travel = -0.020
    capture._apply_anchor_travel(-0.110)
    assert capture._echo_travel == -0.020
    capture._apply_anchor_travel(-0.035)
    assert capture._echo_travel == -0.035
    capture._apply_anchor_travel(-0.0345)
    assert capture._echo_travel == -0.035  # inside the dead band: no churn


def test_the_anchor_parks_mid_reply_once_the_reply_spans_the_cap():
    """A reply longer than the cap anchors when the cap fills, NOT only after the reply
    ends: the travel must be parked before attribution reads the reply's end."""
    capture = AudioCapture()
    mic, ref = _reply_pair(-0.020, seconds=2.8)
    pos = {"i": 0}

    def fake_reference(samples):
        i = pos["i"]
        pos["i"] += samples
        window = np.zeros(samples, np.float32)
        stop = min(i + samples, ref.size)
        window[:stop - i] = ref[i:stop]
        return window

    capture._reference_window = fake_reference
    fed = 0
    for a in range(0, mic.size, 3200):
        frame = (mic[a:a + 3200] * 32767.0).astype("<i2")
        capture._observe_wide_anchor(frame.tobytes())
        fed += 1
        if fed * 3200 >= 1.5 * 16000:
            break
    assert capture._anchor["done"] is True  # the cap finished it, no reply-end gap needed
    travel = capture.echo_travel_ms()
    assert travel is not None
    assert abs(travel + 20.0) <= 3.0


def test_an_unresolved_peak_does_not_park_anything():
    """Pure noise on both microphones: the peak is junk and the reference stays put."""
    capture = AudioCapture()
    rng = np.random.default_rng(9)
    n = int(2.8 * 16000)
    mic = rng.standard_normal(n).astype(np.float32) * 8e-3
    ref = rng.standard_normal(n).astype(np.float32) * 0.2
    _run_anchor(capture, mic, ref)
    assert capture.echo_travel_ms() is None


# --- the echo pilot (task v11d) ---------------------------------------------------

def _pilot_pcm():
    with wave.open(io.BytesIO(pilot_wav_bytes())) as wav:
        return np.frombuffer(wav.readframes(-1), "<i2").astype(np.float32) / 32768.0


def _feed_pilot(capture, mic, ref):
    pos = {"i": 0}

    def fake_reference(samples):
        i = pos["i"]
        pos["i"] += samples
        window = np.zeros(samples, np.float32)
        stop = min(i + samples, ref.size)
        window[:stop - i] = ref[i:stop]
        return window

    capture._reference_window = fake_reference
    capture.note_pilot_playback()
    for a in range(0, mic.size, 3200):
        frame = (mic[a:a + 3200] * 32767.0).astype("<i2")
        capture._observe_pilot(frame.tobytes())
        if capture._pilot["done"]:
            break


def test_the_pilot_parks_the_reference_before_the_first_reply():
    """The live geometry: a 0.35 s pilot audible 0.1 s into the window, its WEAK echo
    (2% of the captured energy, like the live microphone) carried 18 ms BEFORE the map
    said it was audible. Content never resolves that; the chirp must."""
    capture = AudioCapture()
    pilot = _pilot_pcm()
    n = int(1.6 * 16000)
    ref = np.zeros(n, np.float32)
    audible = 1600
    ref[audible:audible + pilot.size] = pilot
    lag = -288  # -18 ms
    rng = np.random.default_rng(3)
    mic = rng.standard_normal(n).astype(np.float32) * 8e-3
    mic[:n + lag] += np.float32(0.02) * ref[-lag:]
    _feed_pilot(capture, mic, ref)
    travel = capture.echo_travel_ms()
    assert travel is not None
    assert abs(travel + 18.0) <= 1.0


def test_the_pilot_refuses_to_park_on_noise():
    capture = AudioCapture()
    n = int(1.6 * 16000)
    rng = np.random.default_rng(17)
    ref = rng.standard_normal(n).astype(np.float32) * 0.2
    mic = rng.standard_normal(n).astype(np.float32) * 8e-3
    _feed_pilot(capture, mic, ref)
    assert capture.echo_travel_ms() is None


def test_the_pilot_wav_is_deterministic_and_playable():
    audio = pilot_wav_bytes()
    assert audio == pilot_wav_bytes()
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        frames = wav.readframes(-1)
    assert abs(len(frames) / 2 / 16000 - 0.35) < 0.01


# --- the pilot SEED (task v11d, rhY's 21:48 note) -----------------------------------

def test_a_clean_pilot_seeds_the_exact_lock_confidently_issue_72_v11d(tmp_path, monkeypatch):
    """The pilot's matched-filter peak (the probe measured ~53 dB, repeats within 0.3 ms)
    is the PROOF: the aligner starts confident at its travel, and the travel is stored
    for this device pair so the next conversation locks instantly."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    capture = AudioCapture()
    capture.hearing_sink, capture.hearing_microphone = "sinkT", "micT"
    pilot = _pilot_pcm()
    n = int(1.6 * 16000)
    ref = np.zeros(n, np.float32)
    audible = 1600
    ref[audible:audible + pilot.size] = pilot
    lag = -288  # -18 ms
    rng = np.random.default_rng(3)
    mic = rng.standard_normal(n).astype(np.float32) * 8e-3
    mic[:n + lag] += np.float32(0.3) * ref[-lag:]  # a CLEAN echo: the chirp sees it
    _feed_pilot(capture, mic, ref)
    assert abs(capture.echo_travel_ms() + 18.0) <= 1.0
    assert capture._echo_seed is not None
    assert abs(capture._echo_seed * 1000.0 + 18.0) <= 1.0
    lock = capture._delay_lock
    assert lock is not None
    assert lock.delay_confident                       # confident AT ONCE
    assert abs(lock.delay_frac + 288) <= 2
    # stored for the pair...
    seed = anccal.echo_seed(anccal.load_profile("sinkT", "micT"), 16000)
    assert seed is not None and abs(seed + 18.0) <= 1.0
    # ...and a later conversation locks from it before its own chirp resolves
    fresh = AudioCapture()
    fresh.hearing_sink, fresh.hearing_microphone = "sinkT", "micT"
    fresh.note_pilot_playback()
    assert fresh._echo_seed is not None
    assert abs(fresh._echo_seed * 1000.0 + 18.0) <= 1.0


def test_an_epoch_clears_the_seed_for_the_next_chirp_issue_72_v11d(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    capture = AudioCapture()
    capture.hearing_sink, capture.hearing_microphone = "sinkT", "micT"
    capture._echo_seed = -0.018
    capture.begin_epoch(output_class="speakers", reason="device-test")
    assert capture._echo_seed is None


def test_the_echo_seed_survives_the_profile_rewrites(tmp_path, monkeypatch):
    """The filter state and the chirp travel share one profile file; neither writer may
    clobber the other's key."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    anccal.save_echo_seed("sinkT", "micT", -18.0, 100.0, 16000)
    profile = anccal.load_profile("sinkT", "micT")
    assert abs(anccal.echo_seed(profile, 16000) + 18.0) < 1e-9
    assert anccal.echo_seed(profile, 48000) is None      # another sample rate, no seed
    assert anccal.echo_seed({"echo_seed": {"travel_ms": -18.0, "prominence": 5.0,
                                           "sample_rate": 16000}}, 16000) is None
    anccal.save_live_state({"w": [1.0]}, 12, "sinkT", "micT", level=0.2)
    profile = anccal.load_profile("sinkT", "micT")
    assert abs(anccal.echo_seed(profile, 16000) + 18.0) < 1e-9   # the filter saved, seed kept
    assert profile["delay"] == 12
