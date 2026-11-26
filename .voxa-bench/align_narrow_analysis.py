#!/usr/bin/env python3
"""Can a narrow-search energy correlation over the whole reply find the live travel?

Full-buffer (and per-reply) RAW cross-correlation, search narrowed to around the wide
anchor's coarse reading (+-80 ms): does the peak land at the chirp-measured -496..-510?
"""

from __future__ import annotations

import sys

import numpy as np

RATE = 16000


def main() -> int:
    data = np.load(sys.argv[1])
    mic, ref = data["mic"].astype(np.float64), data["ref"].astype(np.float64)
    n = min(mic.size, ref.size)
    mic, ref = mic[:n], ref[:n]
    nfft = 1
    while nfft < 2 * n - 1:
        nfft <<= 1
    corr = np.fft.irfft(np.fft.rfft(mic, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    # lag k (0..n-1) pairs mic[i]*ref[i-k]; negative lag at nfft-k.
    print("energy corr, full buffer, peak per search radius:")
    for rad_ms in (400, 150, 80, 40):
        span = int(rad_ms * RATE / 1000)
        idx = list(range(1, span + 1)) + list(range(nfft - span, nfft))
        w = corr[idx]
        p = int(np.argmax(w))
        lag = idx[p]
        if lag >= nfft // 2:
            lag -= nfft
        med = float(np.median(np.abs(w)))
        print(f"  +-{rad_ms} ms: lag={lag} ({lag / RATE * 1000:.1f} ms) "
              f"prominence={float(w[p]) / med:.1f}")
    # coarse-then-fine: wide (RAW, +-1.5 s) then refine +-80 ms around it
    span = int(1500 * RATE / 1000)
    idx = list(range(1, span + 1)) + list(range(nfft - span, nfft))
    w = corr[idx]
    p = int(np.argmax(w))
    lag = idx[p]
    if lag >= nfft // 2:
        lag -= nfft
    print(f"coarse wide: lag={lag} ({lag / RATE * 1000:.1f} ms)")
    for c in range(lag - 60, lag + 61, 15):
        span2 = 60
        idx2 = list(range(c - span2, c + span2 + 1))
        idx2 = [(i % nfft) for i in idx2]
        w2 = corr[idx2]
        p2 = int(np.argmax(w2))
        print(f"  refine around {c}: lag={idx2[p2] % nfft if idx2[p2] < nfft // 2 else idx2[p2] - nfft} "
              f"peak={float(w2[p2]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
