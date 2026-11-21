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


GCCPHAT_SEARCH_MS = 400.0  # roadmap task 3: the 0-400 ms delay search


def gccphat_delay(mic, ref, max_delay_ms: float = GCCPHAT_SEARCH_MS,
                  sample_rate: int = 16000):
    """GCC-PHAT: the delay the echo reaches the microphone at, in samples (fractional).

    The phase transform whitens the cross spectrum, so the peak sits on the DIRECT path even
    when the room rings (RT60 0.2-0.8 s) and the window is only 0.5 s of a TTS utterance
    (roadmap task 3). Returns None when there is nothing to correlate. The peak is refined
    to sub-sample by a 3-point parabolic fit, which is what makes the +-1 ms gate possible.
    """
    mic = np.asarray(mic, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = min(mic.size, ref.size)
    if n < max(1600, int(0.1 * sample_rate)):
        return None
    mic, ref = mic[:n], ref[:n]
    if float(np.mean(mic * mic)) < 1e-10 or float(np.mean(ref * ref)) < 1e-10:
        return None
    max_delay = int(max_delay_ms / 1000.0 * sample_rate)
    size = 1
    while size < n + max_delay:
        size <<= 1
    cross = np.fft.rfft(mic, size) * np.conj(np.fft.rfft(ref, size))
    cross /= np.abs(cross) + 1e-12  # phase transform: flat spectrum, a sharp peak
    corr = np.fft.irfft(cross, size)
    window = corr[: max_delay + 1]
    peak = int(np.argmax(window))
    floor = float(np.median(window))
    if floor > 0.0:
        if window[peak] < 8.0 * floor:
            return None
    elif window[peak] <= 0.0:
        return None
    if peak == 0 or peak == max_delay:
        return float(peak)
    left, top, right = window[peak - 1], window[peak], window[peak + 1]
    denom = left - 2.0 * top + right
    shift = 0.5 * (left - right) / denom if denom != 0.0 else 0.0
    return float(peak) + float(np.clip(shift, -1.0, 1.0))


COHERENCE_MIN = 0.12  # overlap correlation at the estimated lag: echo present, not a near-end accident


class DelayTracker:
    """Track the echo delay and the clock drift of the echo path, continuously.

    Roadmap task 3. At the onset of every TTS utterance the first 0.5 s give a GCC-PHAT
    delay over 0-400 ms: the coarse offset the canceller's reference must carry. The
    estimate repeats every 0.5 s while the utterance plays, so a drifting echo path is
    tracked WITHIN one reply, not only across replies:

    * hysteresis: the coarse offset (``delay_samples``, what the metrics and the canceller
      report) moves only on a shift of more than 1 ms confirmed by the next estimate;
    * slope: the estimates' slope across utterances is the clock drift in ppm; above 50
      ppm ``has_clock_drift`` latches, and the piecewise-linear offset between estimates
      IS the reference resampling the drift needs (the anchors ``observe`` returns);
    * an abrupt jump (a route change, a device switch) is an epoch: it is accepted at
      once, an event of its own, and the slope regression restarts from it - a slow drift
      never looks like a jump, and a jump is never smeared into the drift estimate.

    The tracker never touches the canceller's filter state: keeping it is the point.
    """

    def __init__(self, sample_rate: int = 16000, search_ms: float = GCCPHAT_SEARCH_MS,
                 window_s: float = 0.5, hysteresis_ms: float = 1.0,
                 epoch_ms: float = 8.0, ppm_threshold: float = 50.0,
                 gap_s: float = 0.35) -> None:
        self._sr = int(sample_rate)
        self._search_ms = float(search_ms)
        self._window = max(160, int(round(window_s * self._sr)))
        self._hyst = float(hysteresis_ms) / 1000.0 * self._sr
        self._epoch = float(epoch_ms) / 1000.0 * self._sr
        self._ppm_threshold = float(ppm_threshold)
        self._gap = max(1, int(round(gap_s * self._sr)))  # quiet shorter than this stays inside
        self._pos_samples = 0  # global audio position observe() has seen
        self._in_utterance = False
        self._utt_start = 0
        self._utt_mic: list = []
        self._utt_ref: list = []
        self._utt_next = self._window
        self._gap_run = 0  # consecutive quiet samples inside the open utterance
        self.coarse = 0  # the hysteresis-stabilised integer offset
        self._have_coarse = False
        self._est_ts: list = []  # (t_s, estimate) since the last epoch
        self._est_d: list = []
        self._candidate = None  # an unconfirmed >1 ms shift
        self._epoch_candidate: float | None = None  # an unconfirmed epoch jump
        self.drift_ppm = 0.0
        self.has_clock_drift = False
        self.epochs = 0
        self.last_update: dict | None = None

    def observe(self, mic, ref) -> dict | None:
        """Feed one aligned chunk of (microphone, reference) floats -1..1.

        Returns an update when a 0.5 s window produced an estimate, else None::
        {"kind": "utterance" | "track" | "shift" | "epoch", "t_s", "delay_samples",
         "offset", "anchors": (t0, off0, t1, off1), "drift_ppm"}
        """
        mic = np.asarray(mic, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        n = min(mic.size, ref.size)
        if n == 0:
            return None
        mic, ref = mic[:n], ref[:n]
        playing = float(np.mean(ref * ref)) > SILENT_REFERENCE_POWER
        update = None
        if playing:
            if not self._in_utterance:
                self._in_utterance = True
                self._utt_start = self._pos_samples
                self._utt_mic, self._utt_ref = [], []
                self._utt_next = self._window
            self._utt_mic.append(mic)
            self._utt_ref.append(ref)
            self._gap_run = 0
            while self._utt_next <= self._pos_samples + n - self._utt_start:
                update = self._estimate_at(self._utt_next)
                self._utt_next += self._window
        elif self._in_utterance:
            # A TTS reply pauses between words; closing on every dip would restart the
            # 0.5 s windows and the first estimate would land after playback ends, so the
            # canceller never gets its offset while the echo plays. Hold the reply open
            # across inter-word pauses; only a sustained silence ends it (then the tail
            # is not estimated).
            self._gap_run += n
            if self._gap_run > self._gap:
                self._in_utterance = False
            else:
                self._utt_mic.append(mic)
                self._utt_ref.append(ref)
        self._pos_samples += n
        if update is not None:
            self.last_update = update
        return update

    def _estimate_at(self, utt_sample: int) -> dict | None:
        """GCC-PHAT over the aligned 0.5 s window ending at ``utt_sample`` of the utterance."""
        mic_all = np.concatenate(self._utt_mic)
        ref_all = np.concatenate(self._utt_ref)
        end = min(utt_sample, ref_all.size)
        start = max(0, end - self._window)
        t_s = (self._utt_start + end) / self._sr
        est = gccphat_delay(mic_all[start:end], ref_all[start:end],
                            max_delay_ms=self._search_ms, sample_rate=self._sr)
        if est is None:
            return None
        if not self._coherent(mic_all[start:end], ref_all[start:end], int(round(est))):
            return None  # the peak is an accident of the near-end speaker, not an echo
        return self._accept(float(est), t_s)

    @staticmethod
    def _coherent(mic_w: np.ndarray, ref_w: np.ndarray, lag: int) -> bool:
        """Does the microphone window actually carry a copy of the reference at ``lag``?

        GCC-PHAT whitens everything: when the user speaks over her own reply the peak can land
        on a coincidence of the near-end signal, and a jump to that lag wrecks the canceller.
        A real echo keeps a plain overlap correlation at its own lag well above the accident
        level; near-end-dominated windows do not (the double-talk guard downstream keeps the
        filter, and a missing window costs one 0.5 s delay, not a wrong offset).
        """
        if lag <= 0 or lag >= len(mic_w):
            return False
        m = mic_w[lag:]
        r = ref_w[:len(mic_w) - lag]
        mm = float(np.dot(m, m))
        rr = float(np.dot(r, r))
        if mm <= 0.0 or rr <= 0.0:
            return False
        return float(np.dot(m, r)) ** 2 / (mm * rr) >= COHERENCE_MIN ** 2

    def _accept(self, est: float, t_s: float) -> dict:
        kind = "track"
        if not self._have_coarse:
            self.coarse = int(round(est))
            self._have_coarse = True
            kind = "utterance"
        else:
            delta = est - self.coarse
            if abs(delta) > self._epoch:
                # An abrupt path change is an explicit epoch that skips the hysteresis and
                # restarts the slope regression, but one window is not a jump: a lone
                # near-end spike or a window cut in half by the change itself gets
                # confirmed by the next window agreeing before the coarse offset jumps.
                if (self._epoch_candidate is not None and delta * self._epoch_candidate > 0.0
                        and abs(est - self._epoch_candidate) <= 2.0 * self._hyst):
                    self.coarse = int(round(est))
                    self.epochs += 1
                    self._epoch_candidate = None
                    self._est_ts, self._est_d, self._candidate = [], [], None
                    kind = "epoch"
                else:
                    self._epoch_candidate = est
            elif abs(delta) > self._hyst:
                if self._candidate is not None and delta * self._candidate > 0.0:
                    self.coarse = int(round(est))
                    self._candidate = None
                    kind = "shift"
                else:
                    self._candidate = delta
        self._est_ts.append(t_s)
        self._est_d.append(est)
        if len(self._est_ts) > 24:  # bounded memory: the regression only needs the recent past
            self._est_ts.pop(0)
            self._est_d.pop(0)
        if len(self._est_ts) >= 4:
            t = np.asarray(self._est_ts)
            d = np.asarray(self._est_d)
            slope = float(np.polyfit(t, d, 1)[0])  # samples of extra delay per second
            self.drift_ppm = slope / self._sr * 1e6
            if abs(self.drift_ppm) >= self._ppm_threshold:
                self.has_clock_drift = True
        off0 = self._est_d[-2] if len(self._est_d) >= 2 else est
        t0 = self._est_ts[-2] if len(self._est_ts) >= 2 else t_s
        return {"kind": kind, "t_s": t_s, "delay_samples": self.coarse, "offset": est,
                "anchors": (t0, off0, t_s, est), "drift_ppm": self.drift_ppm}


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

    # The bench rooms decay over hundreds of milliseconds (roadmap task 3 strata): a filter
    # shorter than the echo tail cannot cancel it at all - the bench asks for taps spanning
    # 256 ms. The default stays the short-path length the unit tests warm up in two seconds.
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
        self._ref_base = 0  # global SAMPLE index of _ref_hist[0] (the reference stream, bounded)
        self._pos = 0  # global SAMPLE index of the next microphone sample process() will see
        self._floor = 1.0
        self._blocks_adapted = 0
        self._converged = 0.0
        self._speaking_left = 0
        self._floor_pow = None
        self._floor_known = False
        self._last_mic_power = 0.0
        self._last_residual_power = 0.0
        self._last_reference_power = 0.0
        # Roadmap task 3: the reference is read at a fractional offset that the delay tracker
        # re-estimates every 0.5 s. Between estimates the offset interpolates the two nearest
        # ones, which IS the reference resampling the clock drift needs (+-200 ppm). None = the
        # plain integer delay, exactly the behaviour every legacy caller had.
        self._offset_anchors = None
        self._clock_drift = False

    def set_offset(self, coarse: int, t0: float = 0.0, off0: float = 0.0,
                   t1: float = 0.0, off1: float = 0.0) -> None:
        """Re-point the reference offset WITHOUT touching the filter (roadmap task 3).

        ``coarse`` is the hysteresis-stabilised integer offset the canceller and the metrics
        report; the (t, offset) anchors from :class:`DelayTracker` position the read precisely,
        interpolating between estimates and holding outside them. The weights, the reference
        buffer, the convergence and double-talk state all survive - only an explicit epoch
        (a route change, a device switch, :meth:`reset`) may drop them.
        """
        self._delay = int(coarse)
        if t1 <= t0 and t0 == 0.0 and off0 == 0.0:
            # All anchor defaults: the caller means the plain integer coarse offset.
            self._offset_anchors = None
        elif t1 <= t0:
            self._offset_anchors = None if off0 == float(self._delay) else (
                float(t0), float(off0), float(t0), float(off0))
        else:
            self._offset_anchors = (float(t0), float(off0), float(t1), float(off1))

    def set_clock_drift(self, detected: bool) -> None:
        """Record that capture and playback run on different clocks (read by the capture path)."""
        self._clock_drift = bool(detected)

    @property
    def has_clock_drift(self) -> bool:
        return self._clock_drift

    def _offset_at(self, times_s: np.ndarray) -> np.ndarray:
        """The offset in samples to read the reference at at these global times."""
        if self._offset_anchors is None:
            return np.full(times_s.shape, float(self._delay))
        t0, off0, t1, off1 = self._offset_anchors
        return np.interp(times_s, (t0, t1), (off0, off1))

    def _trim_ref_hist(self) -> None:
        """Keep the reference history bounded: the read window plus the filter span."""
        keep = self._taps + max(self._delay, int(GCCPHAT_SEARCH_MS / 1000.0 * self._sample_rate))
        keep += 4 * self.BLOCK + 2048
        if self._ref_hist.size > keep:
            drop = self._ref_hist.size - keep
            self._ref_hist = self._ref_hist[drop:]
            self._ref_base += drop

    def _shifted_ref(self, length: int) -> np.ndarray:
        """The reference samples the same span as this block, read at the tracked offset.

        Fractional reads interpolate two neighbouring samples; positions before the stream
        began read as silence (what the old pending buffer was zero-initialised to).
        """
        hist = self._ref_hist
        pos = self._pos + np.arange(length)
        times_s = pos / float(self._sample_rate)
        idx = pos - self._offset_at(times_s) - self._ref_base
        idx = np.clip(idx, -1.0, float(hist.size))
        lo = np.floor(idx).astype(np.int64)
        frac = idx - lo
        padded = np.concatenate(([0.0], hist, [0.0]))
        a = padded[lo + 1]
        b = padded[lo + 2]
        return a + (b - a) * frac

    def _delayed_ref(self, length: int) -> np.ndarray:
        """The delayed reference for this block (the offset the filter is fitted with)."""
        return self._shifted_ref(length)

    def _block(self, mic_b: np.ndarray, ref_b: np.ndarray) -> np.ndarray:
        length = len(mic_b)
        self._ref_hist = np.concatenate([self._ref_hist, ref_b])
        self._trim_ref_hist()
        w0 = self.weights.copy() if self.weights is not None else None
        h0 = dict(self.history) if self.history is not None else None
        # The offset (coarse + interpolated fine correction) is applied to the REFERENCE read,
        # not to a pending buffer inside the filter: that is what lets it be fractional, move
        # every 0.5 s and survive a coarse re-point without losing one sample of filter state.
        ref_s = self._shifted_ref(length)
        trial, w1, h1 = nlms_cancel(mic_b, ref_s, taps=self._taps, mu=self._mu,
                                    delay=0, weights=w0, history=h0,
                                    freeze=np.ones(length, dtype=bool))
        delayed = ref_s
        p_mic = float(np.mean(mic_b ** 2))
        p_res = float(np.mean(trial ** 2))
        # Kept for attribution (handoff 6.3, evidence stream 1): what the echo path explains and
        # what is left over, measured on the frozen-filter trial, in dB. Read-only.
        self._last_mic_power = p_mic
        self._last_residual_power = p_res
        self._last_reference_power = float(np.mean(delayed ** 2))
        p_ref = self._last_reference_power
        p_now = float(np.mean(ref_b ** 2))  # power of the reference block itself (what the speakers play now)
        # Tail-aware double-talk: these rooms ring up to 0.8 s after her own words (roadmap
        # task 2 learned the same lesson about AEC3's guard), so microphone energy where the
        # DELAYED reference reads quiet is the tail - the user may only be called when the
        # recent reference history has been quiet for longer than the longest room too.
        hist_ener = float(np.mean(self._ref_hist ** 2)) if self._ref_hist.size else 0.0
        speaking = (p_ref < 1e-7 and hist_ener < 1e-7 and p_mic > 1e-6) or (
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
        result, w2, h2 = nlms_cancel(mic_b, ref_s, taps=self._taps, mu=self._mu,
                                     delay=0, weights=w0, history=h0, freeze=None)
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

