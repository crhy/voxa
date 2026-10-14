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


def _band_limit(x, fmin: float, fmax: float, rate: int) -> np.ndarray:
    """Zero the FFT of ``x`` outside [fmin, fmax) Hz and return the inverse transform."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    X[(f < fmin) | (f >= fmax)] = 0.0
    return np.fft.irfft(X, len(x))


def fine_lag_q(mic, ref, lo: int, hi: int, fmin: float = 300.0, fmax: float = 3000.0,
               rate: int = SR) -> tuple[float, float]:
    """Band-limited cross-correlation peak in [lo, hi) samples with parabolic (sub-sample) refinement.

    Returns ``(lag, quality)``: the delay in samples by which the echo in ``mic`` trails ``ref``
    (mic = ref delayed by the returned amount), and the peak height normalised by the geometric
    mean of the two band-limited stream energies (0..1; near 1 for a clean echo, near 0 for a
    random peak from a silent or band-starved block). The streams are band-limited to
    [fmin, fmax) Hz before correlating so quiet bands cannot steal the peak.
    """
    n = min(len(mic), len(ref))
    if n < 4 or hi <= lo:
        return float(max(0, lo)), 0.0
    a = _band_limit(np.asarray(mic[:n], dtype=np.float64), fmin, fmax, rate)
    b = _band_limit(np.asarray(ref[:n], dtype=np.float64), fmin, fmax, rate)
    size = 1
    while size < 2 * n:
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    hi = min(hi, len(corr))
    if hi <= lo:
        return float(max(0, lo)), 0.0
    window = corr[lo:hi]
    k = int(np.argmax(window)) + lo
    lag = float(k)
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            lag = float(k + 0.5 * (y0 - y2) / denom)
    den = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    q = abs(float(corr[k])) / den if den > 0.0 else 0.0
    return lag, float(min(1.0, q))


def fine_lag(mic, ref, lo: int, hi: int, fmin: float = 300.0, fmax: float = 3000.0,
             rate: int = SR) -> float:
    """Band-limited cross-correlation lag only; see :func:`fine_lag_q` for the quality too."""
    return fine_lag_q(mic, ref, lo, hi, fmin, fmax, rate)[0]


_ANALYSIS_BANDS = ((300.0, 3000.0), (1000.0, 6000.0), (100.0, 1000.0))
_BAND_POWER_FLOOR = 10.0 ** (-50.0 / 10.0)  # -50 dBFS as mean-square power


def fit_drift_fine_info(mic, ref, rate: int = SR, block_s: float = 2.0, hop_s: float = 1.0,
                        search: tuple[int, int] = (0, 4800)) -> tuple[float, float, float, dict]:
    """Robust drift fit; same as :func:`fit_drift_fine` but also returns block counts.

    Returns ``(lag_at_t0, drift_samples_per_second, residual_rms, info)`` with
    ``info = {"blocks_kept", "blocks_total"}``.
    """
    n = min(len(mic), len(ref))
    if n < 4:
        return 0.0, 0.0, 0.0, {"blocks_kept": 0, "blocks_total": 0}
    block = max(1024, int(round(block_s * rate)))
    hop = max(1, int(round(hop_s * rate)))
    coarse_lag, coarse_q = fine_lag_q(mic, ref, search[0], search[1], 300.0, 3000.0, rate)
    if coarse_q < 0.1:
        for fmin, fmax in _ANALYSIS_BANDS[1:]:
            lag, q = fine_lag_q(mic, ref, search[0], search[1], fmin, fmax, rate)
            if q > coarse_q:
                coarse_q, coarse_lag = q, lag
    lo_c = max(search[0], int(coarse_lag) - 40)
    hi_c = min(search[1], int(coarse_lag) + 41)
    ts: list[float] = []
    ls: list[float] = []
    qs: list[float] = []
    total = 0
    for i in range(0, n - block + 1, hop):
        total += 1
        seg_r = np.asarray(ref[i:i + block], dtype=np.float64)
        chosen = None
        for fmin, fmax in _ANALYSIS_BANDS:
            bl = _band_limit(seg_r, fmin, fmax, rate)
            if float(np.mean(bl * bl)) >= _BAND_POWER_FLOOR:
                chosen = (fmin, fmax)
                break
        if chosen is None:
            continue
        lag, q = fine_lag_q(mic[i:i + block], ref[i:i + block], lo_c, hi_c,
                            chosen[0], chosen[1], rate)
        if q < 0.1:
            continue
        ts.append((i + block / 2.0) / rate)
        ls.append(lag)
        qs.append(q)
    info = {"blocks_kept": len(ts), "blocks_total": total}
    kept = len(ts)
    if kept < 4:
        med = float(np.median(ls)) if kept else 0.0
        return med, 0.0, 0.0, info
    t = np.asarray(ts, dtype=np.float64)
    lags = np.asarray(ls, dtype=np.float64)
    qual = np.asarray(qs, dtype=np.float64)
    if t[-1] - t[0] + block_s < 6.0:
        return float(np.median(lags)), 0.0, 0.0, info
    # Robust slope: median of adjacent-block slopes (a gap-safe Theil-Sen variant;
    # all-pairs medians are corrupted by region jumps and silence gaps).
    adj = [(lags[j] - lags[i]) / (t[j] - t[i])
           for i, j in zip(range(kept - 1), range(1, kept), strict=True) if t[j] > t[i]]
    if not adj:
        return float(np.median(lags)), 0.0, 0.0, info
    slope_ts = float(np.median(adj))
    resid0 = lags - (slope_ts * t + float(np.median(lags - slope_ts * t)))
    keep = np.abs(resid0) <= 1.5
    if int(keep.sum()) < 4:
        return float(np.median(lags)), 0.0, 0.0, info
    tk, lk, qk = t[keep], lags[keep], qual[keep]
    w = qk / float(qk.sum())
    slope = slope_ts
    max_slope = 300e-6 * rate
    if abs(slope) > max_slope:
        return float(np.median(lags)), 0.0, 0.0, info
    intercept = float(np.dot(w, lk - slope * tk))
    resid = float(np.sqrt(np.dot(w, (lk - (slope * tk + intercept)) ** 2)))
    return float(intercept), float(slope), resid, info


def fit_drift_fine(mic, ref, rate: int = SR, block_s: float = 2.0, hop_s: float = 1.0,
                   search: tuple[int, int] = (0, 4800)) -> tuple[float, float, float]:
    """Fit the mic/ref delay as a line using sub-sample per-block lags.

    One coarse lag is measured on the whole clip, then each ``block_s`` block (stepped by
    ``hop_s``) is measured with :func:`fine_lag_q` within +-40 samples of that coarse lag, in
    the analysis band where the reference has energy. Blocks that are silent or band-starved
    (quality below 0.1) are dropped; the slope is the median of adjacent-block slopes (a gap-safe
    Theil-Sen variant), blocks more than 1.5 samples from that line are dropped, and the intercept
    is refit by quality-weighted least squares.

    Returns ``(lag_at_t0_samples, drift_samples_per_second, residual_rms_samples)``.
    """
    lag0, drift, resid, _info = fit_drift_fine_info(mic, ref, rate, block_s, hop_s, search)
    return lag0, drift, resid


def fractional_delay(x, d: float, rate: int = SR) -> np.ndarray:
    """Delay ``x`` by ``d`` samples (fractional) via an FFT phase ramp; same length out."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d), len(x))


def _analysis_band(ref, rate: int) -> tuple[float, float]:
    """The analysis band where the reference has energy: 300-3000, else 1000-6000, else 100-1000."""
    for fmin, fmax in _ANALYSIS_BANDS:
        bl = _band_limit(np.asarray(ref, dtype=np.float64), fmin, fmax, rate)
        if float(np.mean(bl * bl)) >= _BAND_POWER_FLOOR:
            return fmin, fmax
    return _ANALYSIS_BANDS[0]


def follow_drift(mic, ref, rate: int = SR, block_s: float = 2.0,
                 ppm: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align the reference to the microphone by following the fitted clock drift block by block.

    The reference is delayed block by block along the fitted drift line (with a 10 ms crossfade
    between blocks so there are no clicks). Band coherence in the band where the reference has
    energy is measured for both the fixed-delay pair and the drift-followed pair, and whichever
    is higher is returned, so following a bad fit can never make things worse. Returns
    ``(mic_aligned, ref_aligned, info)`` of equal length where
    ``info = {"lag", "ppm", "residual", "used", "blocks_kept", "blocks_total"}`` and ``used``
    is ``"drift"`` or ``"fixed"``.

    When ``ppm`` is given the slope is fixed to that drift (parts per million) and only the lag
    is fitted: the fitted mid-clip delay is re-anchored onto the fixed slope. This lets a session
    drift measured on long clips be applied to short clips or clips whose own fit failed.
    """
    n = min(len(mic), len(ref))
    lag0, drift, resid, binfo = fit_drift_fine_info(mic, ref, rate, block_s)
    if ppm is not None:
        t_mid = (n / 2.0) / rate
        mid_delay = lag0 + drift * t_mid
        drift = ppm * 1e-6 * rate
        lag0 = mid_delay - drift * t_mid
    mic_a = np.zeros(n, dtype=np.float64)
    ref_a = np.zeros(n, dtype=np.float64)
    if n > 0:
        mic_a[:] = np.asarray(mic[:n], dtype=np.float64)
    block = max(1024, int(round(block_s * rate)))
    src = np.asarray(ref[:n], dtype=np.float64)

    def delayed(start: int, length: int, d: float) -> np.ndarray:
        # Whole samples by indexing (no wrap-around), only the fraction by phase ramp.
        whole = int(np.floor(d))
        seg = np.zeros(length, dtype=np.float64)
        lo, hi = start - whole, start - whole + length
        a, b = max(lo, 0), min(hi, n)
        if b > a:
            seg[a - lo:b - lo] = src[a:b]
        return fractional_delay(seg, d - whole, rate)

    for i in range(0, n, block):
        length = min(block, n - i)
        ref_a[i:i + length] = delayed(i, length, lag0 + drift * ((i + length / 2.0) / rate))
    fmin, fmax = _analysis_band(ref[:n] if n else ref, rate)
    ref_fixed = delayed(0, n, lag0 + drift * ((n / 2.0) / rate)) if n else ref_a.copy()
    # The segment-averaged estimate is the trustworthy one on real recordings; band_coherence
    # reads about 0.1 on pairs whose true coherence is 0.9.
    coh_fixed = band_coherence_segments([(mic_a, ref_fixed)], fmin, fmax) if n >= 8192 else 0.0
    coh_drift = band_coherence_segments([(mic_a, ref_a)], fmin, fmax) if n >= 8192 else 0.0
    used = "drift"
    if coh_fixed > coh_drift:
        used = "fixed"
        ref_a = ref_fixed
    info = {"lag": lag0, "ppm": drift / rate * 1e6, "residual": resid,
            "used": used, "blocks_kept": binfo["blocks_kept"], "blocks_total": binfo["blocks_total"]}
    return mic_a, ref_a, info


def predictable_share(mic, ref, rate: int = SR, fmin: float = 100.0, fmax: float = 7000.0,
                      nseg: int = 8192) -> float:
    """Power-weighted share of the microphone's power that the reference can predict.

    Welch (segment-accumulated) estimate: Hann windows of ``nseg`` samples stepped by
    ``nseg // 2`` (half overlap); the per-frequency power and cross terms are summed across
    all windows before the ratio is formed. With x = reference and y = microphone, the share
    is ``sum over bins [fmin, fmax] of |Pxy|^2 / Pxx`` divided by ``sum of Pyy`` over the same
    bins, i.e. the fraction of microphone power explainable by the reference. Returns 0.0 for
    clips shorter than ``nseg``. Achievable ERLE is ``-10*log10(1 - share)``.
    """
    x = np.asarray(ref, dtype=np.float64)
    y = np.asarray(mic, dtype=np.float64)
    m = min(len(x), len(y))
    if m < nseg:
        return 0.0
    w = np.hanning(nseg)
    f = np.fft.rfftfreq(nseg, 1.0 / rate)
    band = (f >= fmin) & (f < fmax)
    pxx = np.zeros(nseg // 2 + 1)
    pyy = np.zeros(nseg // 2 + 1)
    pxy = np.zeros(nseg // 2 + 1, dtype=np.complex128)
    for i in range(0, m - nseg + 1, nseg // 2):
        X = np.fft.rfft(x[i:i + nseg] * w)
        Y = np.fft.rfft(y[i:i + nseg] * w)
        pxx += np.abs(X) ** 2
        pyy += np.abs(Y) ** 2
        pxy += X * np.conj(Y)
    den = float(pyy[band].sum())
    if den <= 0.0:
        return 0.0
    num = float(np.sum(np.abs(pxy[band]) ** 2 / (pxx[band] + 1e-30)))
    return float(min(1.0, max(0.0, num / den)))


def band_coherence_segments(pairs, lo: float, hi: float, nseg: int = 8192,
                            rate: int = SR) -> float:
    """Mean magnitude-squared coherence over ``pairs`` in [lo, hi) Hz using Hann windows.

    Windows are ``nseg`` samples long, stepped by ``nseg // 2`` (half overlap); the per-frequency
    power and cross terms are accumulated across all windows before the coherence is formed.
    """
    w = np.hanning(nseg)
    f = np.fft.rfftfreq(nseg, 1.0 / rate)
    band = (f >= lo) & (f < hi)
    pxx = np.zeros(nseg // 2 + 1)
    pyy = np.zeros(nseg // 2 + 1)
    pxy = np.zeros(nseg // 2 + 1, dtype=np.complex128)
    for x, y in pairs:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = min(len(x), len(y))
        for i in range(0, m - nseg + 1, nseg // 2):
            X = np.fft.rfft(x[i:i + nseg] * w)
            Y = np.fft.rfft(y[i:i + nseg] * w)
            pxx += np.abs(X) ** 2
            pyy += np.abs(Y) ** 2
            pxy += X * np.conj(Y)
    den = pxx[band] * pyy[band]
    num = np.abs(pxy[band]) ** 2
    return float(np.mean(num / (den + 1e-30)))
