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
