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
    # The reference samples the filter can ever need: `taps` of filter history plus the delay it
    # was fitted with plus the block being judged, with a margin so the bound is a stated rule
    # rather than an off-by-one tight fit. Anything older cannot appear in the microphone any more
    # (handoff 5.7: the reference history must be a bounded working set, not an append-only list).
    REFERENCE_MARGIN = 1024
    # A room keeps an echo alive after the speakers stop: reflections for ~100 ms. Judging "is an
    # echo in flight" on the render block alone is what let the tail through (handoff 5.7).
    ECHO_TAIL_MS = 128.0

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
        # Global stream position of ``_ref_hist[0]``: the working set slides, the positions do not.
        self._ref_first = 0
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
        # Diagnostics for the health state: blocks handed on as the raw microphone because no
        # echo could be in them (never because the CURRENT render block happened to be silent).
        self._pass_through_blocks = 0

    def _reference_capacity(self) -> int:
        """How many reference samples the working set holds (see ``REFERENCE_MARGIN``)."""
        return (self._taps + max(int(self._delay), 0) + self.BLOCK
                + self._echo_tail_samples() + self.REFERENCE_MARGIN)

    def _echo_tail_samples(self) -> int:
        """How far before the microphone's moment a room can still be delivering an echo."""
        return int(self.ECHO_TAIL_MS / 1000.0 * self._sample_rate)
        return self._taps + max(int(self._delay), 0) + self.BLOCK + self.REFERENCE_MARGIN

    def _push_reference(self, ref_b: np.ndarray) -> None:
        """Append one block to the bounded reference working set.

        Identical to the old append-everything history for every sample the filter can still be
        asked about: the oldest position any block can read is ``_pos - delay``, and the working set
        keeps everything from ``_pos + length - capacity`` back, which is older than that whenever
        ``capacity >= delay + BLOCK`` (true by :meth:`_reference_capacity`). Older samples cannot
        reach the microphone any more, so dropping them changes no output - and a 2 h session stops
        growing at 737 MB.
        """
        self._ref_hist = np.concatenate([self._ref_hist, ref_b])
        capacity = self._reference_capacity()
        if self._ref_hist.size > capacity:
            self._ref_hist = self._ref_hist[-capacity:]
        self._ref_first = self._pos + len(ref_b) - self._ref_hist.size

    def _delayed_ref(self, length: int, lookback: int = 0) -> np.ndarray:
        """Reference samples for the microphone span, zero-padded where nothing is held.

        ``lookback`` extends the window BEFORE the microphone's moment, which is where the room's
        reflections come from: an echo can still be arriving from reference samples the speakers
        stopped playing.
        """
        start = self._pos - self._delay - lookback
        span = length + lookback
        out = np.zeros(span)
        lo = max(start, self._ref_first, 0)
        hi = min(start + span, self._ref_first + self._ref_hist.size)
        if hi > lo:
            out[lo - start: hi - start] = self._ref_hist[lo - self._ref_first: hi - self._ref_first]
        return out

    def _block(self, mic_b: np.ndarray, ref_b: np.ndarray) -> np.ndarray:
        length = len(mic_b)
        self._push_reference(ref_b)
        w0 = self.weights.copy() if self.weights is not None else None
        h0 = dict(self.history) if self.history is not None else None
        trial, w1, h1 = nlms_cancel(mic_b, ref_b, taps=self._taps, mu=self._mu,
                                    delay=self._delay, weights=w0, history=h0,
                                    freeze=np.ones(length, dtype=bool))
        delayed = self._delayed_ref(length)
        # ``in_flight`` is the same window widened by the room's reflection tail: the reference samples
        # that can still be producing sound IN THIS BLOCK. This, not the render block being fed, is
        # what decides whether there is an echo to cancel and whether the microphone's energy is the
        # user rather than her own voice.
        in_flight = self._delayed_ref(length, self._echo_tail_samples())
        p_mic = float(np.mean(mic_b ** 2))
        p_res = float(np.mean(trial ** 2))
        # Kept for attribution (handoff 6.3, evidence stream 1): what the echo path explains and
        # what is left over, measured on the frozen-filter trial, in dB. Read-only.
        self._last_mic_power = p_mic
        self._last_residual_power = p_res
        self._last_reference_power = float(np.mean(delayed ** 2))
        p_in_flight = float(np.mean(in_flight ** 2))
        # ``_last_reference_power`` is the power of the reference AT THE MICROPHONE'S MOMENT: what the
        # speakers played ``delay`` samples ago, the only thing that can be in this block at all - including
        # the echo of a reply that has already stopped (the tail). ``p_in_flight`` widens that window by the
        # room's reflections. The old code asked instead what the speakers play NOW, and handed the raw
        # microphone through whenever the answer was silence: that is the bypass that leaks her own last
        # words back as user words (handoff 5.7: silence in the current render block does not imply silence
        # in the room).
        speaking = (p_in_flight < 1e-7 and p_mic > 1e-6) or (
            self._blocks_adapted > 100
            and p_res / max(p_mic, 1e-12) > max(8.0 * self._floor, 0.05)
            and p_res > 1e-6)
        if speaking:
            self._speaking_left = 25
        # Nothing (or almost nothing) of the speaker path can be in this block: there is no echo
        # to cancel and nothing to learn from, so do not adapt. Judged on the delayed reference, so
        # the echo tail keeps being cancelled after the speakers went quiet.
        reference_quiet = p_in_flight < MIN_REFERENCE_POWER
        silent = p_in_flight < SILENT_REFERENCE_POWER
        if silent:
            # Nothing of the speaker path can be in this block (the echo tail has finished as well
            # as the speakers went quiet): hand the microphone through untouched, bit for bit, keep
            # the learned filter, and still advance the reference history so alignment is not lost.
            self._floor_pow = p_mic if self._floor_pow is None else (
                0.5 * self._floor_pow + 0.5 * p_mic if p_mic < self._floor_pow
                else 0.98 * self._floor_pow + 0.02 * p_mic)
            self._floor_known = True
            self.weights = w1
            self.history = h1
            self._pos += length
            self._pass_through_blocks += 1
            return np.asarray(mic_b, dtype=np.float32)
        if self._floor_known and p_mic < 2.0 * self._floor_pow:
            # The speakers are playing but the microphone does not hear them above its own noise
            # floor: there is nothing to cancel, so leave the microphone alone and do not adapt.
            self.weights = w1
            self.history = h1
            self._pos += length
            self._pass_through_blocks += 1
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
    def pass_through_blocks(self) -> int:
        """Blocks handed on as the raw microphone because no echo could be in them."""
        return int(self._pass_through_blocks)

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

