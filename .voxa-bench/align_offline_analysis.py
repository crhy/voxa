#!/usr/bin/env python3
"""Offline: why does the shipping _measure never lock on LIVE (mic, ref) capture?

Loads .voxa-bench capture npz (starts, lens, mic, ref) and compares candidate
delay estimators x window lengths:
  PHAT      = shipping gccphat (phase-normalized)
  RAW       = plain cross-spectrum (Wiener-ish, energy weighted)
  BAND      = PHAT but only frequencies where the REFERENCE has energy stay unit;
              noise-only bands contribute nothing (kills the noise lobes)
For each window: does the signed +-400 ms peak land near the true lag (-503 = -31.4 ms),
does the second-path gate pass, and what is the coherence at the peak?
"""

from __future__ import annotations

import sys

import numpy as np

RATE = 16000
SEARCH_MS = 400.0
TRUE_LAG = -503  # chirp probe: -31.4 ms


def signed_corr(mic, ref, max_delay, estimator):
    n = min(mic.size, ref.size)
    mic, ref = mic[:n].astype(np.float64), ref[:n].astype(np.float64)
    size = 1
    while size < n + 2 * max_delay:
        size <<= 1
    cross = np.fft.rfft(mic, size) * np.conj(np.fft.rfft(ref, size))
    if estimator == "PHAT":
        cross = cross / (np.abs(cross) + 1e-12)
    elif estimator == "BAND":
        rp = np.abs(np.fft.rfft(ref, size)) ** 2
        mask = rp > np.median(rp)
        w = np.where(mask, 1.0 / (np.abs(cross) + 1e-12), 0.0)
        cross = cross * w
    corr = np.fft.irfft(cross, size)
    return np.concatenate((corr[size - max_delay:], corr[:max_delay + 1]))


def gates(mic, ref, max_delay, estimator):
    signed = signed_corr(mic, ref, max_delay, estimator)
    peak_i = int(np.argmax(signed))
    floor = float(np.median(signed))
    top = float(signed[peak_i])
    lag = peak_i - max_delay
    band = RATE // 25
    far_lo, far_hi = max(0, peak_i - band), min(signed.size, peak_i + band)
    rest = np.concatenate((signed[:far_lo], signed[far_hi:]))
    refused = bool(rest.size and float(rest.max()) >= 0.5 * top)
    prom_ok = bool(top >= 8.0 * floor) if floor > 0 else top > 0
    return lag, top, floor, prom_ok, refused


def coh(mic, ref, lag):
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
    starts, lens = data["starts"], data["lens"]
    mic_cat, ref_cat = data["mic"], data["ref"]
    mic_w, ref_w, off = [], [], 0
    for ln in lens:
        mic_w.append(mic_cat[off:off + ln])
        ref_w.append(ref_cat[off:off + ln])
        off += ln
    mic_all = np.concatenate(mic_w)
    ref_all = np.concatenate(ref_w)

    max_delay = int(SEARCH_MS / 1000.0 * RATE)
    print(f"total samples={mic_all.size} ({mic_all.size / RATE:.2f} s)")

    for estimator in ("PHAT", "RAW", "BAND"):
        for win_s in (0.52, 1.0, 2.0, 4.0, 8.0):
            n = int(win_s * RATE)
            if n > mic_all.size:
                continue
            hits = total = refused = 0
            lags = []
            # slide the whole reply: step 0.16 s, only windows with reference energy
            for a in range(0, mic_all.size - n, RATE // 6):
                m, r = mic_all[a:a + n], ref_all[a:a + n]
                if float(np.mean(r * r)) <= 1e-9:
                    continue
                total += 1
                lag, top, floor, prom, ref2 = gates(m, r, max_delay, estimator)
                lags.append(lag)
                if abs(lag - TRUE_LAG) <= 60:
                    hits += 1
                if ref2:
                    refused += 1
            med = int(np.median(lags)) if lags else 0
            print(f"{estimator:5s} win={win_s:4.2f}s total={total:3d} "
                  f"near_true={hits:3d} second_refused={refused:3d} median_lag={med}")
    print()
    # coherence at TRUE lag on content windows (the real echo test)
    for win_s in (0.52, 1.0, 2.0, 4.0):
        n = int(win_s * RATE)
        vals = []
        for a in range(0, mic_all.size - n, RATE // 6):
            m, r = mic_all[a:a + n], ref_all[a:a + n]
            if float(np.mean(r * r)) <= 1e-9:
                continue
            vals.append(coh(m, r, TRUE_LAG))
        vals = np.array(vals)
        print(f"coh@true win={win_s:4.2f}s: n={len(vals)} median={np.median(vals):.4f} "
              f"p90={np.percentile(vals, 90):.4f} (need >= 0.0144)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
