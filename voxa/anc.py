"""Measure how late an echo is and how much of the mic it accounts for."""

from __future__ import annotations

import contextlib
import threading
import weakref

import numpy as np

# --- one echo canceller owns the microphone at a time -------------------------------------------
# Handoff 7 ("Use one owner for AEC"): cascading unrelated echo cancellers makes the result
# undiagnosable and can remove the near-end speaker. The claim is held by the live canceller (or
# capture path) that owns the echo path; a second live claim is a bug, so it raises.
_AEC_LOCK = threading.Lock()
_AEC_OWNER = None  # weakref to the owner, or None


def claim_aec_owner(owner, kind: str) -> None:
    """Take sole ownership of the echo path for ``owner``; AssertionError if someone else holds it."""
    global _AEC_OWNER
    with _AEC_LOCK:
        current = _AEC_OWNER() if _AEC_OWNER is not None else None
        if current is not None and current is not owner:
            holding = getattr(current, "_aec_kind", "unknown")
            raise AssertionError(
                f"echo canceller {holding!r} already owns the echo path; refusing to run "
                f"{kind!r} as a second one")
        _AEC_OWNER = weakref.ref(owner)
        owner._aec_kind = kind


def release_aec_owner(owner) -> None:
    """Give up ownership if ``owner`` still holds it (a stale release never clears another's claim)."""
    global _AEC_OWNER
    with _AEC_LOCK:
        current = _AEC_OWNER() if _AEC_OWNER is not None else None
        if current is owner:
            _AEC_OWNER = None


def aec_owner_kind() -> str | None:
    """Which canceller owns the echo path right now (``None`` = nobody)."""
    with _AEC_LOCK:
        current = _AEC_OWNER() if _AEC_OWNER is not None else None
        return getattr(current, "_aec_kind", None) if current is not None else None


def erle_db(mic, cleaned) -> float:
    """Energy removed by cleaning, in dB. 0.0 when the mic is silent."""
    mic_ms = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
    if mic_ms < 1e-12:
        return 0.0
    cleaned_ms = max(float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2)), 1e-20)
    return 10.0 * float(np.log10(mic_ms / cleaned_ms))


def estimate_delay(mic, ref, max_delay: int = 4800) -> int:
    """Samples the reference must be delayed to line up with its echo in the mic."""
    n = min(len(mic), len(ref), 5 * 16000)
    if n <= 0:
        return 0
    mic_n = np.asarray(mic[:n], dtype=np.float64)
    ref_n = np.asarray(ref[:n], dtype=np.float64)
    if float(np.mean(mic_n**2)) < 1e-8 or float(np.mean(ref_n**2)) < 1e-8:
        return 0
    size = 1
    while size < 2 * n:
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(mic_n, size) * np.conj(np.fft.rfft(ref_n, size)))
    window = np.abs(corr[: max_delay + 1])
    peak = int(np.argmax(window))
    if window[peak] < 8.0 * float(np.median(window)):
        return 0
    return peak


def nlms_cancel(mic, ref, taps: int = 1024, mu: float = 0.5, delay: int = 0,
                weights=None, history=None, freeze=None):
    """Normalised least-mean-squares echo cancellation, sample by sample.
    Returns (cleaned, weights, history). `weights` (taps,) and `history` (a dict with "buf" (taps,) and
    "pending" (delay,)) carry the filter and the most recent reference samples from one call to the next so
    audio can be processed in chunks; None starts from zeros. `freeze`: optional boolean array, True where
    the filter must NOT adapt (the user is speaking)."""
    mic = np.asarray(mic, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    w = np.zeros(taps) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    if history is None:
        buf = np.zeros(taps)
        pending = np.zeros(delay)
    else:
        buf = np.asarray(history["buf"], dtype=np.float64).copy()
        pending = np.asarray(history["pending"], dtype=np.float64).copy()
    out = np.empty(len(mic))
    for i in range(len(mic)):
        if delay > 0:
            push = pending[-1]
            pending[1:] = pending[:-1]
            pending[0] = ref[i]
        else:
            push = ref[i]
        buf[1:] = buf[:-1]
        buf[0] = push
        e = mic[i] - w @ buf
        out[i] = e
        if freeze is None or not freeze[i]:
            # Regularised by the filter length: with a near-silent reference (the pauses in a voice) a tiny
            # denominator would make the step enormous and the filter blow up.
            w += mu * e * buf / (buf @ buf + taps * 1e-5)
    return out.astype(np.float32), w, {"buf": buf, "pending": pending}


MIN_REFERENCE_POWER = 1e-5  # about -50 dB: below this the computer is effectively silent
SILENT_REFERENCE_POWER = 1e-6  # about -60 dB: the speakers are off, the mic must be untouched


class EchoCanceller:
    """Streaming echo canceller with double-talk protection.

    Audio arrives as consecutive chunks of any length; internally it is cut into
    160-sample (10 ms) blocks aligned to the global stream. Each block is first
    cleaned with the filter frozen; if the leftover looks like double-talk (the
    user speaking) the block is emitted as-is and the filter stays put, otherwise
    the filter adapts on the block.
    """

    BLOCK = 160

    def __init__(self, sample_rate: int = 16000, taps: int = 1024, mu: float = 0.5, delay: int = 0):
        self._sample_rate = sample_rate
        self._taps = taps
        self._mu = mu
        self._delay = delay
        self.weights = None
        self.history = None
        self._mic_rem = np.zeros(0)
        self._ref_rem = np.zeros(0)
        self._ref_hist = np.zeros(0)
        self._pos = 0
        self._floor = 1.0
        self._blocks_adapted = 0
        self._converged = 0.0
        self._speaking_left = 0
        self._floor_pow = None
        self._floor_known = False
        self._last_mic_power = 0.0
        self._last_residual_power = 0.0
        self._last_reference_power = 0.0

    def _delayed_ref(self, length: int) -> np.ndarray:
        start = self._pos - self._delay
        if start >= 0:
            return self._ref_hist[start:start + length]
        head = np.zeros(-start)
        return np.concatenate([head, self._ref_hist[:length + start]])

    def _block(self, mic_b: np.ndarray, ref_b: np.ndarray) -> np.ndarray:
        length = len(mic_b)
        self._ref_hist = np.concatenate([self._ref_hist, ref_b])
        w0 = self.weights.copy() if self.weights is not None else None
        h0 = dict(self.history) if self.history is not None else None
        trial, w1, h1 = nlms_cancel(mic_b, ref_b, taps=self._taps, mu=self._mu,
                                    delay=self._delay, weights=w0, history=h0,
                                    freeze=np.ones(length, dtype=bool))
        delayed = self._delayed_ref(length)
        p_mic = float(np.mean(mic_b ** 2))
        p_res = float(np.mean(trial ** 2))
        # Kept for attribution (handoff 6.3, evidence stream 1): what the echo path explains and
        # what is left over, measured on the frozen-filter trial, in dB. Read-only.
        self._last_mic_power = p_mic
        self._last_residual_power = p_res
        self._last_reference_power = float(np.mean(delayed ** 2))
        p_ref = self._last_reference_power
        p_now = float(np.mean(ref_b ** 2))  # power of the reference block itself (what the speakers play now)
        speaking = (p_ref < 1e-7 and p_mic > 1e-6) or (
            self._blocks_adapted > 100
            and p_res / max(p_mic, 1e-12) > max(8.0 * self._floor, 0.05)
            and p_res > 1e-6)
        if speaking:
            self._speaking_left = 25
        # Nothing (or almost nothing) is being played: there is no echo to learn from, so do not adapt.
        reference_quiet = p_now < MIN_REFERENCE_POWER
        silent = p_now < SILENT_REFERENCE_POWER
        if silent:
            # The speakers are off: hand the microphone through untouched, bit for bit, keep the
            # learned filter, and still advance the reference history so alignment is not lost.
            self._floor_pow = p_mic if self._floor_pow is None else (
                0.5 * self._floor_pow + 0.5 * p_mic if p_mic < self._floor_pow
                else 0.98 * self._floor_pow + 0.02 * p_mic)
            self._floor_known = True
            self.weights = w1
            self.history = h1
            self._pos += length
            return np.asarray(mic_b, dtype=np.float32)
        if self._floor_known and p_mic < 2.0 * self._floor_pow:
            # The speakers are playing but the microphone does not hear them above its own noise
            # floor: there is nothing to cancel, so leave the microphone alone and do not adapt.
            self.weights = w1
            self.history = h1
            self._pos += length
            return np.asarray(mic_b, dtype=np.float32)
        if self._speaking_left > 0 or reference_quiet:
            if self._speaking_left > 0:
                self._speaking_left -= 1
            self.weights = w1
            self.history = h1
            self._pos += length
            return self._never_worse(mic_b, trial, p_mic)
        result, w2, h2 = nlms_cancel(mic_b, ref_b, taps=self._taps, mu=self._mu,
                                     delay=self._delay, weights=w0, history=h0, freeze=None)
        self.weights = w2
        self.history = h2
        self._blocks_adapted += 1
        self._floor = 0.95 * self._floor + 0.05 * (p_res / max(p_mic, 1e-12))
        self._converged = 0.9 * self._converged + 0.1 * erle_db(mic_b, result)
        self._pos += length
        return self._never_worse(mic_b, result, p_mic)

    def _never_worse(self, mic_b, cleaned, p_mic: float):
        """Safety net: the canceller may never hand on something louder than the microphone itself.

        If it does (a filter gone wrong), pass the raw microphone through for this block and pull the filter
        back towards zero so it relearns."""
        if p_mic > 0.0 and float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2)) > 1.5 * p_mic:
            if self.weights is not None:
                self.weights = self.weights * 0.5
            return np.asarray(mic_b, dtype=np.float32)
        if not np.all(np.isfinite(cleaned)):
            if self.weights is not None:
                self.weights = self.weights * 0.5
            return np.asarray(mic_b, dtype=np.float32)
        return cleaned

    def process(self, mic, ref) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        mic_all = np.concatenate([self._mic_rem, mic])
        ref_all = np.concatenate([self._ref_rem, ref])
        outs = []
        i = 0
        while i + self.BLOCK <= len(mic_all):
            outs.append(self._block(mic_all[i:i + self.BLOCK], ref_all[i:i + self.BLOCK]))
            i += self.BLOCK
        self._mic_rem = mic_all[i:]
        self._ref_rem = ref_all[i:]
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs).astype(np.float32)

    def reset(self) -> None:
        self.__init__(sample_rate=self._sample_rate, taps=self._taps, mu=self._mu, delay=self._delay)

    def export_state(self) -> dict:
        weights = self.weights if self.weights is not None else np.zeros(self._taps)
        return {"taps": self._taps, "mu": self._mu, "delay": self._delay,
                "weights": [float(x) for x in weights]}

    def load_state(self, state: dict) -> None:
        if int(state["taps"]) != self._taps:
            return
        self.weights = np.asarray(state["weights"], dtype=np.float64).copy()
        self.history = None

    @property
    def converged_db(self) -> float:
        return self._converged

    @property
    def delay_samples(self) -> int:
        """The delay this path was fitted with (0 = never estimated: alignment unhealthy)."""
        return int(self._delay)

    @property
    def mic_power_db(self) -> float:
        """Energy the microphone delivered on the last block, in dB."""
        return 10.0 * float(np.log10(self._last_mic_power)) if self._last_mic_power > 1e-20 else -120.0

    @property
    def residual_power_db(self) -> float:
        """Energy left after the echo-path fit on the last block, in dB."""
        return 10.0 * float(np.log10(self._last_residual_power)) if self._last_residual_power > 1e-20 else -120.0

    @property
    def echo_explained_db(self) -> float:
        """How much of the microphone the delayed reference explains, in dB (0.0 = all of it)."""
        if self._last_mic_power <= 1e-20:
            return -120.0
        explained = max(self._last_mic_power - self._last_residual_power, 0.0)
        if explained <= 1e-20:
            return -120.0
        return 10.0 * float(np.log10(explained / self._last_mic_power))


# --- WebRTC AEC3 -------------------------------------------------------------------------------
# What is actually linked, verified on 2026-11-19 rather than read off the wrapper's name
# (handoff 5.6/7: "do not infer AEC3 from a wrapper name"):
#   * pywebrtc-audio 0.2.0, Apache-2.0, cp314 manylinux x86_64 wheel, build-id
#     b6706e833285f1b05139081fea76112d82c26a55;
#   * the extension module statically links WebRTC's AEC3: its symbol table carries
#     webrtc::EchoCanceller3 (ProcessCapture/RenderWriter/SetAudioBufferDelay), EchoRemover,
#     Subtractor, ErleEstimator, DelayEstimator, AlignmentMixer and ClockdriftDetector - 34
#     AEC3-only symbols;
#   * the wrapper vendors its WebRTC sources from ewan-xu/AEC3 (see its
#     vendor/webrtc_audio/MODIFICATIONS.md), whose upstream repository was last pushed
#     2022-02-24: `audio_processing/aec3/echo_remover.cc` in the v0.2.0 wheel is byte-identical to
#     that repository (sha256 4b2d0565293dbe6a...) and differs from webrtc/src main today
#     (sha256 946e604a5fa0a879...). So this is AEC3 of early-2022 vintage, not the 2026 upstream,
#     with WebRTC's modern NS/AGC2/RNN-VAD added on top by the wrapper's authors.
#   * the wrapper exposes only EchoCanceller(sample_rate, num_channels, stream_delay_ms),
#     process(near, far), reset() and stream_delay_ms. EchoCanceller3Config is NOT reachable, so
#     `echo_removal_control.has_clock_drift` cannot be pushed into AEC3 from here; AEC3's own
#     ClockdriftDetector still runs inside it. Voxa therefore keeps the flag itself (below) and
#     uses it in its own layers.
AEC3_PACKAGE = "pywebrtc-audio"
AEC3_BLOCK = 160  # 10 ms at 16 kHz: the frame size AEC3 requires
AEC3_RATES = (8000, 16000, 32000, 48000)


def aec3_available() -> bool:
    """True when the WebRTC AEC3 binding can actually be imported and built."""
    try:
        from pywebrtc_audio import EchoCanceller
    except Exception:  # noqa: BLE001 - an import failure is simply "not available here"
        return False
    with contextlib.suppress(Exception):
        EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)
        return True
    return False


def aec3_build_info() -> dict:
    """What AEC3 backend this process really linked - for the log, the bench report and the notes."""
    info = {"package": AEC3_PACKAGE, "version": None, "module": None, "aec3_symbols": None,
            "config_exposed": False, "clock_drift_exposed": False}
    try:
        import importlib.metadata

        info["version"] = importlib.metadata.version(AEC3_PACKAGE)
    except Exception:  # noqa: BLE001 - metadata is nice to have, not required
        pass
    try:
        import pywebrtc_audio

        info["module"] = getattr(pywebrtc_audio, "__file__", None)
        canceller = pywebrtc_audio.EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)
        info["aec3_symbols"] = "EchoCanceller3 (verified in the wheel's symbol table)"
        info["clock_drift_exposed"] = hasattr(canceller, "has_clock_drift")
        info["config_exposed"] = hasattr(canceller, "config")
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


class Aec3Canceller:
    """WebRTC AEC3 behind Voxa's canceller interface (roadmap task 2).

    ``process(mic, ref)`` is the same call shape as :class:`EchoCanceller`: consecutive chunks of
    any length, float samples in -1..1, cleaned float32 out. Internally the streams are cut into
    the 160-sample (10 ms) frames AEC3 requires, aligned to the global stream, and AEC3 is fed
    CONTINUOUSLY - silence and the echo tail included - because its delay estimator, its filter and
    its near-end suppressor all keep state across frames (handoff 6.2: never reset per sentence).

    AEC3 owns the delay: ``stream_delay_ms`` stays 0 and its internal estimator searches, so
    :attr:`delay_samples` reports 0 and :attr:`delay_managed_internally` says why.

    One instance owns the echo path: constructing a second live AEC3 (or running one while the
    GStreamer ``webrtcdsp`` path holds the claim) raises AssertionError.
    """

    BLOCK = AEC3_BLOCK
    # The capture path must not bypass chunks whose reference is silent: AEC3 needs every frame.
    continuous = True
    delay_managed_internally = True
    # AEC3's linear filter covers num_filters * 64 samples: 832 samples = 52 ms at 16 kHz.
    filter_span_ms = 52.0

    def __init__(self, sample_rate: int = 16000, stream_delay_ms: int = 0, *,
                 clock_drift: bool = False, claim: bool = True,
                 double_talk_guard: bool = True) -> None:
        if int(sample_rate) not in AEC3_RATES:
            raise ValueError(f"AEC3 runs at {AEC3_RATES} Hz, not {sample_rate}")
        from pywebrtc_audio import EchoCanceller

        if claim:
            claim_aec_owner(self, "aec3")
        self._sample_rate = int(sample_rate)
        self._stream_delay_ms = int(stream_delay_ms)
        self._double_talk_guard = bool(double_talk_guard)
        # The residual-ratio guard cannot see through AEC3 on this machine's rooms: the user's
        # residual measures only 1.5x above the echo path's own floor, while the microphone holds
        # 40 % of its energy that no echo model can explain. The guard therefore asks the
        # REFERENCE that question (the same tail-held model the bench chain uses), not AEC3.
        from voxa.refmask import ReferenceMask

        self._detect = (ReferenceMask(rate=self._sample_rate, tail=True)
                        if double_talk_guard else None)
        self._dt_excess = 0.1
        self._impl = EchoCanceller(sample_rate=self._sample_rate, num_channels=1,
                                   stream_delay_ms=self._stream_delay_ms)
        self._mic_rem = np.zeros(0, dtype=np.float32)
        self._ref_rem = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._frames = 0
        self._guard_frames = 0
        self._double_talk_frames = 0
        self._converged = 0.0
        self._clock_drift = bool(clock_drift)
        self._last_mic_power = 0.0
        self._last_residual_power = 0.0
        self._last_reference_power = 0.0

    def process(self, mic, ref) -> np.ndarray:
        """Clean one chunk against the reference covering the same span; both float -1..1."""
        mic = np.asarray(mic, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError("AEC3 cancels one mono stream at a time")
        length = min(mic.size, ref.size)
        if length == 0:
            return np.zeros(0, dtype=np.float32)
        self._mic_rem = np.concatenate([self._mic_rem, mic[:length]])
        self._ref_rem = np.concatenate([self._ref_rem, ref[:length]])
        out = []
        i = 0
        while i + self.BLOCK <= self._mic_rem.size:
            mic_b = self._mic_rem[i:i + self.BLOCK]
            ref_b = self._ref_rem[i:i + self.BLOCK]
            out.append(self._frame(mic_b, ref_b))
            i += self.BLOCK
        self._mic_rem = self._mic_rem[i:]
        self._ref_rem = self._ref_rem[i:]
        if self._mic_rem.size:
            # A chunk that is not a multiple of 160 still owes its caller its own length: the
            # partial frame is handed on as the microphone delivered it (AEC3 only ever sees whole
            # 10 ms frames, and feeding it a padded frame would teach its filter silence).
            out.append(self._mic_rem)
            self._mic_rem = self._mic_rem[0:0]
            self._ref_rem = self._ref_rem[0:0]
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def _frame(self, mic_b: np.ndarray, ref_b: np.ndarray) -> np.ndarray:
        """One 10 ms frame through AEC3, with the same never-worse guard the NLMS path has."""
        p_mic = float(np.mean(mic_b.astype(np.float64) ** 2))
        p_ref = float(np.mean(ref_b.astype(np.float64) ** 2))
        try:
            cleaned = np.asarray(self._impl.process(mic_b, ref_b), dtype=np.float32)
        except Exception:  # noqa: BLE001 - a frame AEC3 cannot take must not stop the microphone
            self._guard_frames += 1
            self._note(p_mic, p_mic, p_ref)
            return mic_b
        if cleaned.size != mic_b.size or not np.all(np.isfinite(cleaned)):
            # AEC3 gone wrong on this frame: hand the microphone through and keep going.
            self._guard_frames += 1
            self._note(p_mic, p_mic, p_ref)
            return mic_b
        p_res = float(np.mean(cleaned.astype(np.float64) ** 2))
        if p_mic > 0.0 and p_res > p_mic:
            # Never hand on something louder than the microphone itself (AEC3's high-pass filter or
            # a diverged filter). The frame passes through raw; the filter keeps its state. The
            # bound is 1.0x, not the NLMS path's 2.0x: AEC3's own suppression already shrinks the
            # residual, so anything above the microphone is the filter's doing, and the bench
            # measures exactly that frame in erle_fst (f129: a +1 dB tail frame measured 0 dB).
            self._guard_frames += 1
            self._note(p_mic, p_mic, p_ref)
            return mic_b
        self._frames += 1
        if self._detect is not None and p_ref >= MIN_REFERENCE_POWER and p_mic > 1e-7:
            # AEC3's suppressor pushes the near end down with the echo (ITU-T P.340 allows at most
            # 3 dB of near-end attenuation in double talk - this binding does tens of dB), so the
            # frame is handed to the reference model first: more than a tenth of this frame's
            # energy unexplained by the held echo model means the user is speaking over Voxa, and
            # the frame goes on as the microphone delivered it. Nothing is discarded - the raw
            # frame is the safest copy - and AEC3 keeps adapting (handoff 6.2). The model's first
            # frames learn the coupling, so it gets 4 frames of history before it may decide.
            self._detect.process(mic_b.astype(np.float64), ref_b.astype(np.float64))
            if (self._detect.frames_done >= 4
                    and self._detect.last_excess_frac > self._dt_excess):
                self._double_talk_frames += 1
                self._note(p_mic, p_res, p_ref)
                return mic_b
        self._note(p_mic, p_res, p_ref)
        self._converged = 0.9 * self._converged + 0.1 * erle_db(mic_b, cleaned)
        return cleaned

    def _note(self, p_mic: float, p_res: float, p_ref: float) -> None:
        self._last_mic_power = p_mic
        self._last_residual_power = p_res
        self._last_reference_power = p_ref
        self._pos += self.BLOCK

    def reset(self) -> None:
        """Drop AEC3's internal state (a route change, an explicit epoch) - never per sentence."""
        with contextlib.suppress(Exception):
            self._impl.reset()
        self._mic_rem = np.zeros(0, dtype=np.float32)
        self._ref_rem = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._frames = 0
        self._guard_frames = 0
        self._double_talk_frames = 0
        self._converged = 0.0
        if self._detect is not None:
            self._detect.reset()

    def close(self) -> None:
        """Give up echo-path ownership (the capture path calls this when it changes mode)."""
        release_aec_owner(self)

    def export_state(self) -> dict:
        """AEC3 keeps no readable filter, so there is nothing to save per device (task 3 changes that)."""
        return {"backend": "aec3", "sample_rate": self._sample_rate,
                "stream_delay_ms": self._stream_delay_ms,
                "saved_filter": False, "note": "AEC3 exposes no filter state in pywebrtc-audio 0.2.0"}

    def load_state(self, state: dict) -> None:
        """Nothing to load: AEC3 starts from its own defaults. Kept so callers stay canceller-agnostic."""
        return None

    def set_clock_drift(self, detected: bool) -> None:
        """Record that capture and playback run on different clocks (mic and sink are different devices).

        AEC3's ``echo_removal_control.has_clock_drift`` is not reachable through this binding, so the
        flag lives here: the capture path and the attribution layer read it, and roadmap task 3 acts
        on it. AEC3's own ClockdriftDetector still runs inside the library.
        """
        self._clock_drift = bool(detected)

    @property
    def has_clock_drift(self) -> bool:
        return self._clock_drift

    @property
    def converged_db(self) -> float:
        return self._converged

    @property
    def delay_samples(self) -> int:
        """Always 0 for AEC3: its own estimator owns the delay and does not report it (task 3)."""
        return 0

    @property
    def frames_processed(self) -> int:
        return self._frames

    @property
    def guard_frames(self) -> int:
        """Frames passed through raw because AEC3's output failed the never-worse guard."""
        return self._guard_frames

    @property
    def double_talk_frames(self) -> int:
        """Frames passed through raw because the residual was near-end speech, not echo."""
        return self._double_talk_frames

    @property
    def mic_power_db(self) -> float:
        return 10.0 * float(np.log10(self._last_mic_power)) if self._last_mic_power > 1e-20 else -120.0

    @property
    def residual_power_db(self) -> float:
        return 10.0 * float(np.log10(self._last_residual_power)) if self._last_residual_power > 1e-20 else -120.0

    @property
    def echo_explained_db(self) -> float:
        if self._last_mic_power <= 1e-20:
            return -120.0
        explained = max(self._last_mic_power - self._last_residual_power, 0.0)
        if explained <= 1e-20:
            return -120.0
        return 10.0 * float(np.log10(explained / self._last_mic_power))

