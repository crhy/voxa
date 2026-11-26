#!/usr/bin/env python3
"""Full-buffer fit decision: what does _fit_echo_travel see on the live capture?

Prints the signed PHAT gate chain over the WHOLE capture (the EchoWindow buffer scale)
and coherence at the peak, plus the peak lag per non-overlapping content chunk.
"""

from __future__ import annotations

import sys

import numpy as np

RATE = 16000


def signed_phat(mic, ref, max_delay):
    n = min(mic.size, ref.size)
    mic, ref = mic[:n].astype(np.float64), ref[:n].astype(np.float64)
    size = 1
    while size < n + 2 * max_delay:
        size <<= 1
    cross = np.fft.rfft(mic, size) * np.conj(np.fft.rfft(ref, size))
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.irfft(cross, size)
    return np.concatenate((corr[size - max_delay:], corr[:max_delay + 1]))


def coh_at(mic, ref, lag):
    if lag == 0 or abs(lag) >= len(mic) or abs(lag) >= len(ref):
        return -1.0
    if lag > 0:
        end, start = len(mic), lag
    else:
        end = min(len(mic), len(ref) + lag)
        start = 0
    m, r = mic[start:end], ref[start - lag:end - lag]
    mm, rr = float(np.dot(m, m)), float(np.dot(r, r))
    if mm <= 0 or rr <= 0:
        return -1.0
    return float(np.dot(m, r)) ** 2 / (mm * rr)


def main() -> int:
    data = np.load(sys.argv[1])
    mic, ref = data["mic"], data["ref"]
    max_delay = int(0.4 * RATE)
    signed = signed_phat(mic, ref, max_delay)
    peak_i = int(np.argmax(signed))
    lag = peak_i - max_delay
    floor = float(np.median(signed))
    top = float(signed[peak_i])
    band = RATE // 25
    rest = np.concatenate((signed[:peak_i - band], signed[peak_i + band:]))
    print(f"FULL n={min(mic.size, ref.size)} peak_lag={lag} ({lag / RATE * 1000:.1f} ms) "
          f"top={top:.3e} floor={floor:.3e} ratio={top / floor:.1f}")
    print(f"  prominence_ok(8x)={top >= 8 * floor} second_path_refused="
          f"{float(rest.max()) >= 0.5 * top} (rest.max={float(rest.max()):.3e} "
          f"half_top={0.5 * top:.3e})")
    print(f"  coh@peak={coh_at(mic, ref, lag):.5f} need>=0.0144 (COHERENCE_MIN^2)")
    # peak lag over 1.5 s content chunks (is the echo lag stable / resolved per chunk?)
    n = RATE * 3 // 2
    lags, topr, refr = [], [], []
    for a in range(0, mic.size - n, n):
        m, r = mic[a:a + n], ref[a:a + n]
        if float(np.mean(r * r)) <= 1e-9:
            continue
        s = signed_phat(m, r, max_delay)
        p = int(np.argmax(s)) - max_delay
        f = float(np.median(s))
        rr = np.concatenate((s[:p + max_delay - band], s[p + max_delay + band:]))
        lags.append(p)
        topr.append(float(s[p + max_delay]) / f if f > 0 else 0.0)
        refr.append(float(rr.max()) >= 0.5 * float(s[p + max_delay]))
    print(f"1.5s chunks: lags={lags}")
    print(f"           ratios={[f'{x:.0f}' for x in topr]}")
    print(f"           refused={refr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
