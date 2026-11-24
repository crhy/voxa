#!/usr/bin/env python3
"""Device/Bluetooth strata lab (roadmap task 10, issue #72).

The OFFLINE bench-strata companion to the smoke suite. The frozen clips may not change (B3), so
task 10's two impairments are pinned here instead, and read off the SAME operators the bench uses:

  * a 300 ms BULK delay (the report's "A2DP widens the delay search to 400 ms": a Bluetooth link
    buffers hundreds of ms on top of the room, and a search window the size of the room cannot
    find it), and
  * a MID-CLIP DEVICE SWITCH: a different room (new RIR) and a different delay, at a known frame
    - one explicit epoch, not a per-sentence rebuild.

It reuses voxa/bench/render.py's frozen echo operators (make_rir / fft_convolve), voxa/anc's
canceller and delay tracker, voxa/onset.py's onset rule and voxa/voicedetect.py's joint gate, all
driven by the per-output table in voxa/echo.py - so a stratum here is measured the way the live
pipeline would be.

What it measures, per stratum:

  delay_est_err_ms    the delay error of the tracker's estimate at the moment it is made
  erle_db             the steady ERLE the output holds
  erle_loss_vs_50ms_db  how much ERLE the bulk delay costs against the same path at 50 ms
  reconverge_s        the seconds after the switch until ERLE is back at 15 dB
  false_onsets        barge-in onsets raised while the user is SILENT (must stay 0)
  unqualified_s       audio inside the recovery window the canceller could not explain yet

No microphone, no playback, no sound device: pure numpy on synthetic paths. It reads nothing from
the running desktop and changes nothing.

Usage:  /Voxa/.venv/bin/python3 tools/device_strata_lab.py            # per-stratum table
        /Voxa/.venv/bin/python3 tools/device_strata_lab.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxa.anc import DelayTracker, EchoCanceller, estimate_delay, erle_db
from voxa.bench.render import ECHO_RMS, SR, fft_convolve, make_rir
from voxa.echo import (OUTPUT_BT_A2DP, OUTPUT_HEADPHONES, OUTPUT_SPEAKERS, OutputTuning,
                       output_tuning)
from voxa.onset import OnsetDetector, level_dbfs

CHUNK = 320  # 20 ms frames, like the bench
CHUNK_S = CHUNK / SR
SWITCH_AT_S = 4.0
RECONVERGE_ERLE_DB = 15.0


def speech_like(n: int, seed: int) -> np.ndarray:
    """A broadband speech/music-like reference: a delay cannot be measured from pure tones."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / np.sqrt(1.0 + (f / 1500.0) ** 2)
    ref = np.fft.irfft(spec, n)
    return (0.5 * ref / np.max(np.abs(ref))).astype(np.float32)


def path(ref: np.ndarray, seed: int, rt60_s: float, delay_samples: int) -> np.ndarray:
    """One device's own acoustic path: the room it is in and how far its sound travels."""
    rng = np.random.default_rng(seed)
    echo = fft_convolve(ref, make_rir(rng, rt60_s))[:len(ref)]
    wave = np.zeros_like(echo)
    wave[delay_samples:] = echo[:-delay_samples] if delay_samples else echo
    rms = float(np.sqrt(np.mean(wave**2)))
    return (wave * (ECHO_RMS / max(rms, 1e-9))).astype(np.float32)


def run_case(seconds: float, delay0_ms: float, delay1_ms: float, tuning: OutputTuning,
             seed: int, switch_at_s: float | None = SWITCH_AT_S) -> dict:
    """One synthetic clip: the pipeline starts on one output and the device changes mid-clip.

    The switch is an EXPLICIT epoch exactly as task 10 defines it: the reference history, the
    canceller and the delay estimate restart together, the thresholds are re-read from the new
    output's table, and the recovery window opens. Nothing else is rebuilt anywhere.
    """
    n = int(seconds * SR)
    ref = speech_like(n, seed)
    delay0 = int(round(delay0_ms / 1000.0 * SR))
    delay1 = int(round(delay1_ms / 1000.0 * SR))
    switch = int(round((switch_at_s if switch_at_s is not None else seconds + 1.0) * SR))

    # BEFORE the switch: one room, one delay. AFTER: a different device (different room, different
    # travel time). The user says nothing at any point: every onset this clip raises is false.
    mic = path(ref, seed, 0.35, delay0)
    if switch_at_s is not None:
        after = path(ref, seed + 1, 0.55, delay1)
        mic = np.where(np.arange(n) < switch, mic, after).astype(np.float32)

    # The pipeline's own alignment step, with the search window THIS output allows (task 10):
    # the room-sized 300 ms window cannot see a Bluetooth-delayed echo at all, and returns 0.
    search_samples = int(round(tuning.delay_search_ms / 1000.0 * SR))
    bulk = int(2.0 * SR)
    delay_est = int(estimate_delay(mic[:bulk], ref[:bulk], max_delay=search_samples))
    found_first = delay_est > 0
    canceller = (EchoCanceller(sample_rate=SR, taps=4096, mu=0.7, delay=delay_est)
                 if found_first else EchoCanceller(sample_rate=SR, taps=4096, mu=0.7))
    tracker = DelayTracker(sample_rate=SR)
    onset = OnsetDetector(margin_db=tuning.onset_margin_db, evidence_ms=tuning.onset_evidence_ms)

    est_err: list[float] = []
    est0_err: list[float] = []
    curve: list[tuple[float, float]] = []
    false_onsets = 0
    unqualified_s = 0.0
    epoch_at_s: list[float] = []
    recovery_until_s = 0.0
    aligned = found_first

    def align(segment_start: int, truth: int) -> int:
        """The epoch's re-estimate: the delay is the one thing that changed, so it is measured."""
        window = slice(segment_start, min(segment_start + bulk, n))
        return int(estimate_delay(mic[window], ref[window], max_delay=search_samples))

    for i in range(0, n, CHUNK):
        t_s = (i + CHUNK / 2.0) / SR
        mc, rc = mic[i:i + CHUNK], ref[i:i + CHUNK]

        # one explicit epoch at the switch: never per sentence, always per device. The reference
        # history and the canceller go with it; the delay estimate restarts.
        if switch_at_s is not None and i >= switch and not epoch_at_s:
            epoch_at_s.append(t_s)
            delay1_est = align(switch, delay1)
            est0_err.append(abs(delay1_est - delay1) / SR * 1000.0)
            canceller = (EchoCanceller(sample_rate=SR, taps=4096, mu=0.7, delay=delay1_est)
                         if delay1_est > 0 else EchoCanceller(sample_rate=SR, taps=4096, mu=0.7))
            tracker = DelayTracker(sample_rate=SR)
            onset.reset()
            aligned = delay1_est > 0
            recovery_until_s = t_s + tuning.recovery_s

        upd = tracker.observe(mc, rc)
        if upd is not None:
            canceller.set_offset(upd["delay_samples"], *upd["anchors"])
            truth = delay1 if (epoch_at_s and t_s >= (switch_at_s or 0.0)) else delay0
            est_err.append(abs(float(upd["delay_samples"]) - truth) / SR * 1000.0)
            aligned = True

        if t_s < recovery_until_s and not aligned:
            unqualified_s += CHUNK_S

        cleaned = canceller.process(mc, rc)
        curve.append((t_s, erle_db(mc, cleaned)))

        # exactly the way the bench hands the onset rule its two numbers (runner.py:1002): the
        # microphone's int16 RMS in dBFS, and the reference's own power in dBFS. The delay the
        # rule is told about is the one the pipeline actually believes (None = unaligned).
        mic_db = level_dbfs(float(np.sqrt(np.mean(mc * mc))) * 32768.0)
        ref_power = float(np.mean(rc * rc))
        ref_db = 10.0 * math.log10(ref_power) if ref_power > 1e-12 else -140.0
        onset.note_delay(canceller.delay_samples / SR * 1000.0 if canceller.delay_samples else None)
        outcome = onset.observe(mic_db, ref_db, dt_ms=CHUNK_S * 1000.0)
        if outcome.onset:
            false_onsets += 1
            onset.reset()

    est0_err.insert(0, abs(delay_est - delay0) / SR * 1000.0)

    curve = np.array(curve)
    before = curve[curve[:, 0] < (switch_at_s if switch_at_s is not None else seconds) - 0.5]
    steady = round(float(np.mean(before[:, 1])), 2) if before.size else 0.0

    after = curve[(curve[:, 0] >= (switch_at_s or 0.0) + 1.5) &
                  (curve[:, 0] <= (switch_at_s or 0.0) + 4.0)] if switch_at_s is not None else curve
    erle_after = round(float(np.mean(after[:, 1])), 2) if after.size else None

    reconverge = None
    if switch_at_s is not None:
        for t_s, erle in curve[curve[:, 0] >= switch_at_s]:
            if erle >= RECONVERGE_ERLE_DB:
                reconverge = round(float(t_s - switch_at_s), 3)
                break

    return {
        "delay0_ms": delay0_ms, "delay1_ms": delay1_ms, "switch_at_s": switch_at_s,
        "output_class": tuning.output_class, "delay_search_ms": tuning.delay_search_ms,
        "residual_margin_db": tuning.residual_margin_db, "onset_margin_db": tuning.onset_margin_db,
        "delay_est_err_ms": round(float(np.mean(est_err)), 1) if est_err else None,
        "bulk_est_err_ms": round(float(np.mean(est0_err)), 1) if est0_err else None,
        "aligned_from_bulk": found_first,
        "delay_estimates": len(est_err),
        "erle_db": steady,
        "erle_after_switch_db": erle_after,
        "reconverge_s": reconverge,
        "false_onsets": false_onsets,
        "unqualified_s": round(unqualified_s, 2),
    }


STRATA: list[tuple[str, float, float, OutputTuning, int, float | None]] = [
    # task 10's delay stratum: the same bulk delay seen through the two search windows. The room
    # can only reach ~300 ms, so its window is 300 ms; a Bluetooth link adds codec buffering on
    # TOP of that, and a window that size is the only one that finds the echo at all.
    ("bt-delay-300ms/speakers-search", 300.0, 120.0, output_tuning(OUTPUT_SPEAKERS), 5, None),
    ("bt-delay-300ms/a2dp-search", 300.0, 120.0, output_tuning(OUTPUT_BT_A2DP), 5, None),
    ("bt-delay-380ms/speakers-search", 380.0, 120.0, output_tuning(OUTPUT_SPEAKERS), 5, None),
    ("bt-delay-380ms/a2dp-search", 380.0, 120.0, output_tuning(OUTPUT_BT_A2DP), 5, None),
    # the report's own comparison: the SAME bulk delay costs how much ERLE against the 50 ms
    # room-only path (task 10's gate: within 3 dB)
    ("no-switch-300ms", 300.0, 0.0, output_tuning(OUTPUT_BT_A2DP), 5, None),
    ("no-switch-50ms-floor", 50.0, 0.0, output_tuning(OUTPUT_BT_A2DP), 5, None),
    # the mid-clip device switch, one epoch per device
    ("switch-to-a2dp", 300.0, 90.0, output_tuning(OUTPUT_BT_A2DP), 5, SWITCH_AT_S),
    ("switch-to-headphones", 120.0, 60.0, output_tuning(OUTPUT_HEADPHONES), 5, SWITCH_AT_S),
    ("switch-to-speakers", 120.0, 60.0, output_tuning(OUTPUT_SPEAKERS), 5, SWITCH_AT_S),
]


def table() -> list[dict]:
    rows = []
    for name, delay0, delay1, tuning, seed, switch_at in STRATA:
        row = run_case(9.0, delay0, delay1, tuning, seed, switch_at)
        row["stratum"] = name
        rows.append(row)
    floor = next((r for r in rows if r["stratum"] == "no-switch-50ms-floor"), None)
    for row in rows:
        row["erle_loss_vs_50ms_db"] = (round(floor["erle_db"] - row["erle_db"], 2)
                                       if floor else None)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="roadmap task 10 device/Bluetooth strata lab")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    rows = table()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    head = (f"{'stratum':>20} {'delay_ms':>9} {'search_ms':>10} {'bulk_err_ms':>11} "
            f"{'coarse_err_ms':>13} {'#est':>4} {'erle_db':>8} {'reconverge_s':>12} "
            f"{'false_onsets':>12} {'unqualified_s':>13}")
    print(head)
    for r in rows:
        span = f"{r['delay0_ms']:.0f}->{r['delay1_ms']:.0f}"
        print(f"{r['stratum']:>20} {span:>9} "
              f"{r['delay_search_ms']:>10.0f} {str(r['bulk_est_err_ms']):>11} "
              f"{str(r['delay_est_err_ms']):>13} "
              f"{r['delay_estimates']:>4} {r['erle_db']:>8.2f} {str(r['reconverge_s']):>12} "
              f"{r['false_onsets']:>12} {r['unqualified_s']:>13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
