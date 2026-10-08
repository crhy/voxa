"""Keep the reference (monitor) stream aligned with the microphone stream.

The monitor and the microphone come from different clocks, so the delay between
them drifts (sample-rate drift) and one of the streams can drop samples (a
dropout).  This module measures the delay over time, fits the drift as a line,
and resamples the reference onto the microphone timeline in front of the
canceller in voxa/anc.py.
"""

from __future__ import annotations

import numpy as np

SR = 16000


def band_coherence(mic, ref, fmin: float = 300.0, fmax: float = 3000.0) -> float:
    """Magnitude-squared coherence between two equal-length streams in a frequency band."""
    n = min(len(mic), len(ref))
    if n < 4:
        return 0.0
    m = np.fft.rfft(np.asarray(mic, dtype=np.float64)[:n])
    r = np.fft.rfft(np.asarray(ref, dtype=np.float64)[:n])
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    band = (freqs >= fmin) & (freqs <= fmax)
    power_m = np.abs(m[band]) ** 2
    power_r = np.abs(r[band]) ** 2
    den = float(power_m.sum() * power_r.sum())
    if den <= 0.0:
        return 0.0
    num = float(np.abs(np.sum(m[band] * np.conj(r[band]))) ** 2)
    return num / den


def correlation_peak(mic, ref, max_delay: int = 8000) -> tuple[float, float]:
    """Best echo delay in samples and how far the peak stands above correlation noise."""
    n = min(len(mic), len(ref), 5 * SR)
    if n <= 0:
        return 0, 0.0
    mic_n = np.asarray(mic[:n], dtype=np.float64)
    ref_n = np.asarray(ref[:n], dtype=np.float64)
    if float(np.mean(mic_n**2)) < 1e-8 or float(np.mean(ref_n**2)) < 1e-8:
        return 0, 0.0
    size = 1
    while size < 2 * n:
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(mic_n, size) * np.conj(np.fft.rfft(ref_n, size)))
    window = np.abs(corr[:max_delay + 1])
    med = float(np.median(window))
    peak = int(np.argmax(window))
    ratio = window[peak] / med if med > 0 else float("inf")
    if 0 < peak < len(window) - 1:
        best_o, best_v = 0.0, window[peak]
        for o in np.linspace(-1.0, 1.0, 41):
            p = peak + o
            lo = int(np.ceil(p))
            hi = min(len(mic_n), len(ref_n))
            if hi <= lo:
                continue
            t = np.arange(lo, hi)
            pos = t - p
            i0 = np.floor(pos).astype(np.int64)
            w = pos - i0
            i0c = np.clip(i0, 0, len(ref_n) - 1)
            i1 = np.minimum(i0 + 1, len(ref_n) - 1)
            rv = ref_n[i0c] * (1.0 - w) + ref_n[i1] * w
            v = abs(float(np.dot(mic_n[t], rv)))
            if v > best_v:
                best_v, best_o = v, o
        peak = peak + best_o
    return peak, ratio


def resample_to(x, length: int, ratio: float = 1.0, frac: float = 0.0) -> np.ndarray:
    """Resample ``x`` to ``length`` samples, taking output i from input position ``i * ratio + frac``.

    Linear interpolation between neighbours; positions outside ``x`` come out zero.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(length, dtype=np.float64)
    if length <= 0 or len(x) == 0:
        return out
    pos = np.arange(length, dtype=np.float64) * ratio + frac
    valid = (pos >= 0.0) & (pos < len(x))
    i0 = np.floor(pos).astype(np.int64)
    i0c = np.clip(i0, 0, len(x) - 1)
    w = pos - i0
    contrib = x[i0c] * (1.0 - w)
    hi = np.minimum(i0 + 1, len(x) - 1)
    contrib = contrib + x[hi] * w * ((i0 + 1) < len(x))
    return np.where(valid, contrib, 0.0)


def fit_drift(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares line through measured (mic position, delay) points, clamped to +-3000 ppm."""
    ts = np.array([p[0] for p in points], dtype=np.float64)
    ds = np.array([p[1] for p in points], dtype=np.float64)
    tc = ts - float(ts.mean())
    tt = float(np.dot(tc, tc))
    if tt <= 0.0:
        return 0.0, float(ds.mean())
    slope = float(np.dot(tc, ds - float(ds.mean())) / tt)
    slope = max(-3000e-6, min(3000e-6, slope))
    return slope, float(np.mean(ds - slope * ts))


class DriftTracker:
    """Tracks the delay between the microphone and its reference over time.

    ``delay(t)`` is how many reference samples the echo in the microphone at
    microphone position ``t`` trails its source.  The tracker probes the streams
    periodically, fits a line through the measured delays (the slope is the
    clock drift in samples per sample), and re-anchors when a measurement jumps
    by more than ``jump`` samples (a dropout in one of the streams).
    """

    def __init__(self, sample_rate: int = SR, probe_every: int = SR, window: int = 8192,
                 max_delay: int = 8000, min_ref_power: float = 1e-6, max_ppm: float = 3000.0,
                 jump: int = 512, min_ratio: float = 4.0):
        self._probe_every = max(1, probe_every)
        self._window = max(1024, window)
        self._max_delay = max_delay
        self._min_ref_power = min_ref_power
        self._max_ppm = max_ppm
        self._jump = jump
        self._min_ratio = min_ratio
        self._points: list[tuple[float, float]] = []
        self._slope = 0.0
        self._offset = 0.0
        self._last_probe = -(1 << 60)

    def observe(self, mic, ref, mic_start: int) -> None:
        """Feed one chunk pair covering microphone positions [mic_start, mic_start + len(mic)).

        The reference chunk must cover the same wall-clock span as the microphone
        chunk (not the shifted echo span).  Probes periodically at the end of the
        chunk; quiet references and unclear correlations are skipped.
        """
        end = mic_start + len(mic)
        if end - self._last_probe < self._probe_every:
            return
        self._last_probe = end
        w = min(len(mic), len(ref), self._window)
        if w < 1024:
            return
        m = np.asarray(mic[len(mic) - w:], dtype=np.float64)
        r = np.asarray(ref[len(ref) - w:], dtype=np.float64)
        if float(np.mean(r * r)) < self._min_ref_power:
            return
        peak, ratio = correlation_peak(m, r, self._max_delay)
        if ratio < self._min_ratio:
            return
        t = float(end - w)
        if self._points and abs(peak - (self._offset + self._slope * t)) > self._jump:
            self._points = []  # dropout: the old timeline no longer holds
        self._points.append((t, float(peak)))
        self._points = self._points[-24:]
        self._refit()

    def _refit(self) -> None:
        ts = [p[0] for p in self._points]
        if len(self._points) >= 3 and ts[-1] - ts[0] >= 2 * self._probe_every:
            self._slope, self._offset = fit_drift(self._points)
        else:
            self._offset = float(np.median([d - self._slope * t for t, d in self._points]))

    def delay_at(self, mic_pos: float) -> float:
        return self._offset + self._slope * mic_pos

    def ref_span(self, mic_start: int, length: int) -> tuple[float, float]:
        """Start and length of the reference span that produced the echo of a mic span."""
        d0 = self.delay_at(mic_start)
        d1 = self.delay_at(mic_start + length)
        return mic_start - d0, length - (d1 - d0)

    def aligned_ref(self, mic, ref, mic_start: int) -> np.ndarray:
        """Resample a reference chunk covering the mapped span onto the microphone timeline."""
        return resample_to(ref, len(mic), 1.0 - self._slope)


def align_pair(mic, ref, probe_every: int = 4000, window: int = 4096) -> np.ndarray:
    """Offline: run the tracker over a whole pair and return the reference on the mic timeline.

    The returned array has len(mic) samples; reference positions the tracker maps
    outside the input are zero.
    """
    n = min(len(mic), len(ref))
    tracker = DriftTracker(probe_every=probe_every, window=window)
    step = tracker._probe_every
    for i in range(0, n, step):
        tracker.observe(mic[i:i + step], ref[i:i + step], i)
    ratio = 1.0 - tracker._slope
    d_0 = tracker.delay_at(0.0)
    if n >= 1024:
        best_v = -1.0
        best_sl, best_d = tracker._slope, d_0
        for dsl in np.arange(-50e-6, 50e-6, 2e-6):
            rr = 1.0 - (tracker._slope + dsl)
            for cand in np.arange(d_0 - 2.0, d_0 + 2.0, 0.1):
                al = resample_to(ref, n, rr, frac=-cand)
                v = abs(float(np.dot(mic[:n], al)))
                if v > best_v:
                    best_v, best_sl, best_d = v, tracker._slope + dsl, cand
        ratio = 1.0 - best_sl
        d_0 = best_d
    lo = int(max(0, np.floor(-d_0)))
    hi = int(min(n, np.ceil(ratio * (n - 1) - d_0) + 1))
    span = ref[lo:hi]
    return resample_to(span, len(mic), ratio, frac=float(-d_0 - lo))
