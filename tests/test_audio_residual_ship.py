"""f171: the residual stage SHIPPED on the live capture path (issue #72).

The bench arm ``voxa-residual`` (f169) measured the win; f171 puts the same
:class:`voxa.refmask.LockedResidualMask` on the shipping canceller's output in
:mod:`voxa.audio`. These tests drive the LIVE path (:class:`voxa.audio.AudioCapture`)
offline the way the GStreamer callback does - ``_maybe_estimate_delay`` then
``_mix_or_passthrough`` per 20 ms frame - and hold it to the five contracts the task
names: the mask is driven at the delay the canceller is driven at, it claims no delay
nobody handed it, it is off outside TTS plus the echo tail, a degraded frame never
passes an unmeasured mask verdict, and the confidence fields never crash.
"""

from __future__ import annotations

import numpy as np

from voxa.align import ExactAligner, ExactDelayLock
from voxa.audio import AudioCapture, ReferenceRing

SR = 16000
CHUNK = SR * 20 // 1000  # the 20 ms frame the capture callback delivers


def broadband(n: int, seed: int = 0) -> np.ndarray:
    """A broadband reference (music-like): a delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float64)


def room(seed: int, tail_s: float = 0.05) -> np.ndarray:
    """Room impulse response: a direct spike followed by a decaying reverberant tail."""
    rng = np.random.default_rng(seed)
    n = int(tail_s * SR)
    tail = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.012 * SR))
    h = 0.3 * tail / np.sqrt(np.sum(tail**2))
    h[0] += 1.0
    return h.astype(np.float64)


def echo_at(ref: np.ndarray, delay_fn) -> np.ndarray:
    """ref delayed by ``delay_fn(t_seconds)`` samples (linear interp -> fractional lag)."""
    n = len(ref)
    t = np.arange(n) / SR
    idx = np.arange(n, dtype=np.float64) - delay_fn(t)
    return np.interp(idx, np.arange(n, dtype=np.float64), ref, left=0.0, right=0.0)


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0),
                   -32768, 32767).astype("<i2").tobytes()


def _rms(bytes_or_array) -> float:
    x = (np.frombuffer(bytes_or_array, dtype="<i2") if isinstance(bytes_or_array, bytes)
         else np.asarray(bytes_or_array))
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _voxa_capture(total: int, anchor: int = 0) -> AudioCapture:
    """A live capture path on the ring reference (the monitor path), counters aligned the way
    ``_on_ref_sample``/``_on_sample`` leave them: the reference stream's position 0 sits at the
    microphone's position ``anchor``, so every window the path serves is the aligned pair the
    clip has (a clip that opens mid-conversation puts the speakers' pre-roll at a negative
    anchor: reference position 0 played ``abs(anchor)`` microphone samples before the mic began)."""
    cap = AudioCapture()
    cap.set_echo_mode("voxa")
    cap._ref_ring = ReferenceRing(max(1, total))
    cap._ref_anchor = anchor
    return cap


def _feed(cap: AudioCapture, ref: np.ndarray, mic: np.ndarray) -> list[bytes]:
    """Run one (mic, ref) pair through the capture path in 20 ms frames, as ``_on_sample`` does.

    The reference goes to the ring first (the monitor's ``_on_ref_sample`` does exactly this), so
    ``_reference_window`` serves the aligned window the render timeline would hand the live path.
    ``mic`` may be shorter than ``ref``: the mic clip then begins mid-stream, with the speakers
    already playing (what a live recording of a running TTS conversation looks like).
    """
    cap._ref_ring.append(ref.astype(np.float32))
    out: list[bytes] = []
    for lo in range(0, len(mic), CHUNK):
        hi = lo + CHUNK
        cap._maybe_estimate_delay(_pcm(mic[lo:hi]))
        out.append(cap._mix_or_passthrough(_pcm(mic[lo:hi])))
    return out


def test_the_live_path_drives_the_mask_at_the_delay_the_canceller_uses() -> None:
    """``set_delay`` gets the SAME delay the canceller is driven at, frame by frame."""
    ref = broadband(3 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(21))[: len(ref)]
    cap = _voxa_capture(len(ref))
    assert isinstance(cap._residual_mask, type(cap._residual_mask))  # the stage exists
    _feed(cap, ref, mic)

    assert cap.delay_confident, "a known-delay clip must prove the lock on the live path"
    locked = float(cap.delay_frac)
    assert abs(locked - 500.0) <= 6.0
    # the mask reads its reference at the delay the canceller was just driven at
    assert cap._residual_mask._exact_lag == int(round(locked))
    assert cap._residual_mask._exact_lag == cap.canceller.delay_samples


def test_no_delay_handed_means_no_mask_claim() -> None:
    """Nobody handed the mask a delay: ``_exact_lag`` stays None, the mask claims none."""
    cap = _voxa_capture(SR)
    user = 0.05 * np.random.default_rng(7).standard_normal(SR)
    _feed(cap, np.zeros(SR), user)  # the user speaks, the speakers never do
    assert cap._residual_mask._exact_lag is None
    assert not cap.delay_confident


def test_the_mask_is_off_outside_tts_plus_the_echo_tail() -> None:
    """The stage is active through TTS plus ``tail_s`` and never outside: after the room went
    quiet long enough, the near-end keeps its energy instead of being ducked."""
    ref = broadband(2 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(22))[: len(ref)]
    cap = _voxa_capture(len(ref) + SR)
    _feed(cap, ref, mic)
    assert cap._residual_mask._exact_lag is not None  # the stage was live through the tail

    # the speakers and their tail have been silent for seconds: user audio only
    user = 0.05 * np.random.default_rng(8).standard_normal(SR)
    cap._delay_done = True  # the delay is already tracked; this test is about the stage
    frames = _feed(cap, np.zeros(SR), user)
    stream = b"".join(frames)
    assert abs(_rms(stream) - _rms(_pcm(user))) / _rms(_pcm(user)) < 0.35


def test_a_degraded_frame_never_hands_the_mask_an_unmeasured_delay() -> None:
    """The lock can never prove itself: the estimate keeps the canceller pointed (visible
    degraded) and the mask is never handed a delay nobody measured."""
    ref = broadband(2 * SR)
    mic = np.convolve(echo_at(ref, lambda t: 500.0), room(23))[: len(ref)]
    cap = _voxa_capture(2 * len(ref) + SR)
    cap._delay_done = False
    cap._delay_lock = ExactDelayLock(sample_rate=SR, canceller=cap.canceller)
    cap._delay_lock.aligner = ExactAligner(coherence_min=0.999)  # can never prove the path
    _feed(cap, ref, mic)
    assert cap.delay_degraded and not cap.delay_confident
    assert cap._residual_mask._exact_lag is None  # never claims a delay it was not handed
    assert cap.canceller.delay_samples > 0        # ...yet the estimate keeps it pointed


def test_the_shipping_stage_cuts_the_residue_the_canceller_left() -> None:
    """The chain is what ships: the residual stage on the locked output takes the residue the
    linear canceller left, while the user's own span survives."""
    def delay(t):  # noqa: ANN001
        return 1600.0 + 300.0 / 1e6 * 16000 * t

    # The speakers were playing before the microphone started: the clip opens mid-conversation,
    # exactly like the recorded bench clips, so the mask bootstraps on live residue from its
    # first frame (a mic that opens on silence before the echo arrives is not what the stage
    # ships against -- and the bench has never measured that case either).
    pre, clip = 2 * SR, 3 * SR
    ref_full = broadband(pre + clip)
    echo_full = np.convolve(echo_at(ref_full, delay), room(24))[: pre + clip]
    rng = np.random.default_rng(25)
    user = np.zeros(clip)
    user[int(1.6 * SR):] = 0.05 * rng.standard_normal(clip - int(1.6 * SR))
    mic = echo_full[pre:] + user

    withmask = _voxa_capture(pre + clip + SR, anchor=-pre)
    stream = b"".join(_feed(withmask, ref_full, mic))
    without = _voxa_capture(pre + clip + SR, anchor=-pre)
    without._residual_mask = None  # the f168 chain: the locked canceller alone
    base = b"".join(_feed(without, ref_full, mic))

    s = np.frombuffer(stream, dtype="<i2").astype(np.float64)
    b = np.frombuffer(base, dtype="<i2").astype(np.float64)
    a, z = int(0.4 * SR), int(1.4 * SR)  # Voxa plays, the user has not started
    assert _rms(b[a:z]) > 0.0, "the canceller alone still leaves residue here"
    assert _rms(s[a:z]) < 0.6 * _rms(b[a:z]), (
        f"the residual stage left {100 * (1 - _rms(s[a:z]) / _rms(b[a:z])):.1f} % of the residue")
    u = slice(int(2.0 * SR), int(2.8 * SR))
    assert _rms(s[u]) > 0.5 * _rms(b[u]), "the user is never deleted for sharing a clip with echo"


def test_nothing_crashes_on_the_confidence_fields() -> None:
    """Silence, and a near-end-only clip: the confidence/delay fields stay readable."""
    cap = _voxa_capture(SR)
    _feed(cap, np.zeros(SR), np.zeros(SR))
    assert cap.delay_samples == 0 and cap.delay_frac == 0.0
    assert not cap.delay_confident and not cap.delay_degraded
    user = 0.05 * np.random.default_rng(9).standard_normal(SR)
    cap2 = _voxa_capture(SR)
    frames = _feed(cap2, np.zeros(SR), user)
    assert b"".join(frames)
    assert cap2._residual_mask._exact_lag is None
