"""Measure how late an echo is and how much of the mic it accounts for."""

from __future__ import annotations

import numpy as np


def erle_db(mic, cleaned) -> float:
    """Energy removed by cleaning, in dB. 0.0 when the mic is silent."""
    mic_ms = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
    if mic_ms < 1e-12:
        return 0.0
    cleaned_ms = max(float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2)), 1e-20)
    return 10.0 * float(np.log10(mic_ms / cleaned_ms))


def estimate_delay(mic, ref, max_delay: int = 8000) -> int:
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
        p_ref = float(np.mean(delayed ** 2))
        speaking = (p_ref < 1e-7 and p_mic > 1e-6) or (
            self._blocks_adapted > 100
            and p_res / max(p_mic, 1e-12) > max(8.0 * self._floor, 0.05)
            and p_res > 1e-6)
        if speaking:
            self._speaking_left = 25
        # Nothing (or almost nothing) is being played: there is no echo to learn from, so do not adapt.
        reference_quiet = p_ref < MIN_REFERENCE_POWER
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
