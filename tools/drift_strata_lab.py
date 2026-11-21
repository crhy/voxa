#!/usr/bin/env python3
"""Drift strata lab (roadmap task 3, issue #72).

The OFFLINE bench-strata companion to the smoke suite. Where the smoke suite draws one random
impairment per clip, this lab FIXES one variable per stratum - clock drift (+-50/100/200 ppm)
or an abrupt echo-path step (20-80 ms) - and reads the three task-3 metrics off the SAME
pipeline the bench uses:

  delay_est_err_ms    the delay error of the tracker's estimate at the moment it is made,
  drift_erle_loss_db  the ERLE a drifting path loses against the same path not drifting,
  reconverge_s        the seconds after an abrupt step until ERLE is back at 15 dB.

It reuses voxa/bench/render.py's frozen echo operators (make_rir / fft_convolve /
resample_clock) and voxa/bench/metrics.echo_delay_at_seconds (the truth delay at an instant),
so a stratum here is measured exactly the way the smoke suite measures a clip - only with the
impairment pinned instead of random.

No microphone, no playback, no audio devices: pure numpy on synthetic paths.

Usage:  /Voxa/.venv/bin/python3 tools/drift_strata_lab.py            # per-stratum table
        /Voxa/.venv/bin/python3 tools/drift_strata_lab.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxa.anc import DelayTracker, EchoCanceller, erle_db
from voxa.bench.metrics import echo_delay_at_seconds
from voxa.bench.render import ECHO_RMS, SR, fft_convolve, make_rir, resample_clock

CHUNK = 320  # 20 ms frames, like the bench
DRIFT_STRATA = [0.0, 50.0, -50.0, 100.0, -100.0, 200.0, -200.0]
STEP_STRATA = [20.0, 40.0, 60.0, 80.0]


def tts_like(n: int, seed: int) -> np.ndarray:
    """A broadband speech/music-like reference: a delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (f / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float32)


def run_case(seconds: float, delay0_ms: float, drift_ppm: float, step_at_s: float | None,
             step_ms: float, seed: int, rt60_s: float = 0.4) -> dict:
    """One synthetic clip through the real echo operators: continuous canceller + delay
    tracker fed 20 ms frames; return the three task-3 metrics for this stratum."""
    n = int(seconds * SR)
    ref = tts_like(n, seed)
    delay_samples = int(round(delay0_ms / 1000.0 * SR))
    step_samples = int(round(step_ms / 1000.0 * SR)) if step_ms else 0
    step_at = int(round(step_at_s * SR)) if step_at_s is not None else None

    rng = np.random.default_rng(seed)
    echo_raw = fft_convolve(ref, make_rir(rng, rt60_s))
    # render.py applies drift/step to the already-reverberated echo (the far-end keeps its clock).
    echo = resample_clock(echo_raw, n, delay_samples, step_at, step_samples, drift_ppm)
    rms = float(np.sqrt(np.mean(echo**2)))
    mic = (echo * (ECHO_RMS / max(rms, 1e-9))).astype(np.float32)

    echo_truth = {
        "delay_samples": delay_samples, "clip_seconds": n / SR,
        "step_at_s": (step_at / SR) if step_at else None,
        "delay_step_ms": (step_ms or 0.0), "drift_ppm": drift_ppm,
    }

    canceller = EchoCanceller(sample_rate=SR, taps=4096, mu=0.7)
    tracker = DelayTracker(sample_rate=SR)
    est_err: list[float] = []     # coarse (hysteresis-stabilised) integer offset vs truth
    offset_err: list[float] = []  # fine fractional offset the anchors interpolate (what the filter reads)
    curve: list[tuple[float, float]] = []
    for i in range(0, n, CHUNK):
        mc, rc = mic[i:i + CHUNK], ref[i:i + CHUNK]
        upd = tracker.observe(mc, rc)
        if upd is not None:
            canceller.set_offset(upd["delay_samples"], *upd["anchors"])
            truth = echo_delay_at_seconds(echo_truth, upd["t_s"])
            if truth is not None:
                est_err.append(abs(float(upd["delay_samples"]) - truth))
                offset_err.append(abs(float(upd["offset"]) - truth))
        cleaned = canceller.process(mc, rc)
        curve.append(((i + CHUNK / 2.0) / SR, erle_db(mc, cleaned)))
    curve = np.array(curve)

    settle = curve[:, 0] >= 2.0
    if step_at is not None:
        settle &= curve[:, 0] < (step_at / SR) - 0.5
    if not settle.any():
        settle = curve[:, 0] >= 2.0
    steady = round(float(np.mean(curve[settle, 1])), 3) if settle.any() else 0.0

    reconverge = None
    if step_at is not None:
        for t_s, erle in curve[curve[:, 0] > step_at / SR]:
            if erle >= 15.0:
                reconverge = round(float(t_s - step_at / SR), 3)
                break

    return {
        "drift_ppm": drift_ppm, "step_ms": step_ms, "step_at_s": step_at_s,
        "delay_est_err_ms": round(float(np.mean(est_err)) / SR * 1000.0, 3) if est_err else None,
        "delay_offset_err_ms": round(float(np.mean(offset_err)) / SR * 1000.0, 3) if offset_err else None,
        "delay_estimates": len(est_err),
        "steady_erle_db": steady,
        "reconverge_s": reconverge,
    }


def table() -> list[dict]:
    rows, ref_erle = [], None
    for drift in DRIFT_STRATA:
        r = run_case(10.0, 100.0, drift, None, 0.0, seed=7)
        if drift == 0.0:
            ref_erle = r["steady_erle_db"]
        r["drift_erle_loss_db"] = round(ref_erle - r["steady_erle_db"], 3) if ref_erle is not None else None
        rows.append(r)
    for step in STEP_STRATA:
        rows.append(run_case(10.0, 100.0, 0.0, 4.0, step, seed=11))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="roadmap task 3 drift/path-change strata lab")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    rows = table()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    head = (f"{'drift_ppm':>9} {'step_ms':>8} {'step_at_s':>9} | {'coarse_err_ms':>13} "
            f"{'fine_err_ms':>11} {'#est':>4} {'steady_erle_db':>14} {'drift_erle_loss_db':>18} {'reconverge_s':>12}")
    print(head)
    for r in rows:
        print(f"{r['drift_ppm']:>9.0f} {str(r['step_ms']):>8} {str(r['step_at_s']):>9} | "
              f"{str(r['delay_est_err_ms']):>13} {str(r['delay_offset_err_ms']):>11} {r['delay_estimates']:>4} "
              f"{r['steady_erle_db']:>14} {str(r.get('drift_erle_loss_db')):>18} {str(r['reconverge_s']):>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
