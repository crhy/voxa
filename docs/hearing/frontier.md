# Voxa Hearing 2 - the echo-cancellation frontier (task h01, issue #72)

Every serious candidate for Voxa's echo stage, measured on Voxa's own bench: synthetic clips, seed 0,
offline (no microphone, no playback, no audio devices). This is the frontier Hearing 2 asked for:
what echo removal is achievable in Voxa's rooms today, at what cost to the user's voice, per
candidate - written before anyone writes another chain gate.

How to regenerate every number in this file:
`/Voxa/.venv/bin/python3 -m voxa.bench run --suite full --seed 0 --echo-mode <mode> --out
.voxa-bench/frontier-<mode>-full.json --no-gates --quiet` for each of `voxa`, `aec3`, `speex`,
`hybrid`, `dtln`, `nkf`, then `tools/hearing_frontier_table.py .voxa-bench/frontier-<mode>-full.json`.
The measured rows are pasted from that script's output; only `aec3full` (never run) is a hand-written
probe reason. Arms: `voxa`, `aec3`, `speex`, `hybrid`, `aec3full`, `dtln`, `nkf`.

## The table (full suite: 480 clips, seed 0)

| mode | erle_fst_median_db (dB) | erle_fst_p10_db (dB) | nearend_atten_dt_db (dB) | echo_leak_word_rate (frac) | dt_wer_penalty_pts (pts) | user_recall_nst (frac) | delay_est_err_ms (ms) | cpu_core_share (cores) | erle_music_loss_db (dB) | echo_ms_per_audio_s | echo_core_share | this machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxa | 6.92 [6.11, 7.61] | -5.00 [-5.30, -4.75] | 9.37 [8.21, 10.47] | 0.1911 [0.1799, 0.2028] | 128.64 [111.73, 147.69] | 0.9507 [0.9351, 0.9640] | 9.29 [7.22, 11.46] | 0.0660 [0.0634, 0.0688] | 0.1263 [0.0837, 0.1748] | 65.6 | 0.118 | Voxa's NLMS (f130 delay tracking) + f131 residual suppressor |
| aec3 | 31.43 [30.35, 32.39] | 0.2290 [-0.1341, 0.7504] | 24.96 [23.45, 26.56] | 0.0498 [0.0431, 0.0563] | 84.17 [73.69, 95.75] | 0.9267 [0.9075, 0.9447] | 6.36 [3.74, 9.89] | 0.0337 [0.0329, 0.0345] | unmeasured | 33.8 | 0.154 | WebRTC AEC3 as the `pywebrtc-audio` wheel ships it |
| speex | 2.14 [2.04, 2.24] | 0.6200 [0.5700, 0.6500] | -0.5871 [-1.0131, -0.1733] | 0.4669 [0.4512, 0.4821] | 149.04 [114.95, 191.03] | 0.9507 [0.9351, 0.9640] | 6.36 [3.74, 9.89] | 0.0124 [0.0120, 0.0129] | unmeasured | 12.8 | 0.078 | libspeexdsp (system .so) via ctypes |
| hybrid | 31.43 [30.35, 32.39] | 0.2290 [-0.1341, 0.7504] | 24.96 [23.45, 26.56] | 0.0498 [0.0431, 0.0563] | 84.17 [73.69, 95.75] | 0.9267 [0.9075, 0.9447] | 6.36 [3.74, 9.89] | 0.0337 [0.0329, 0.0346] | unmeasured | 33.9 | 0.194 | linear aec3 (installed wheel) + Voxa's residual suppressor |
| aec3full | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | not published for this interpreter: no AEC3-full shim could be built (set `VOXA_AEC3FULL_SHIM`); the freedesktop webrtc-audio-processing 1.3 source's `echo_remover.h` needs Abseil C++ headers (`libabsl-dev`) absent here and un-installable offline, and no PyPI binding exposing the full `EchoCanceller3Config` is published for Python 3.14 (tried `pyaec`, `webrtc-audio-processing`) — see `/home/rhy/voxa-bench-data/third_party/SOURCES.md` |
| dtln | 24.41 [23.72, 25.03] | 2.19 [1.74, 2.87] | 7.86 [6.77, 9.04] | 0.0786 [0.0708, 0.0862] | 98.64 [80.89, 120.47] | 0.9339 [0.9159, 0.9495] | 10.98 [7.56, 14.48] | 0.0894 [0.0881, 0.0907] | n/a | 90.2 | 0.232 | DTLN-aec 512 (MIT) weights via `ai_edge_litert`; streaming validated vs the repo's `*_processed.wav` |
| nkf | 1.33 [0.86, 1.79] | -7.39 [-7.84, -7.04] | 3.03 [2.39, 3.74] | 0.3904 [0.3764, 0.4047] | 145.54 [94.63, 216.48] | 0.9471 [0.9315, 0.9615] | 10.98 [7.56, 14.48] | 0.8620 [0.8399, 0.8850] | n/a | 850.7 | 2.102 | NKF-AEC (fjiang9, `nkf_epoch70.pt`) linear-only Kalman via torch; streaming == offline (corr 1.0) but ~850 ms audio/s and 2.1 cores — not real-time here |

Bracketed ranges are bootstrap 95 % intervals over clips (300 resamples, `tools/hearing_frontier_table.py`).
`cpu_core_share` is the whole pipeline (what the gates record). The echo stage's own cost is the last
two columns: `echo_ms_per_audio_s` is milliseconds of CPU time per second of audio, `echo_core_share`
its share of one core. Every arm that runs here is real time on this CPU by a wide margin (worst:
Voxa's own NLMS + suppressor at 0.118 of a core, 65.6 ms per second of audio, on 16 threads).

Runtime dependency size, measured on this machine (what the Flatpak would carry): Voxa's own NLMS and
f131 suppressor 0 (pure Python, already shipped); `libspeexdsp.so.1` 84 KB; the `pywebrtc-audio` wheel
952 KB (its extension module statically links WebRTC AEC3; the system
`libwebrtc-audio-processing-1.so.3` is 1.1 MB); onnxruntime 67 MB for any neural arm.

### The same arms on smoke (52 clips, seed 0) - the fast arm check

| mode | erle_fst_median_db (dB) | erle_fst_p10_db (dB) | nearend_atten_dt_db (dB) | echo_leak_word_rate (frac) | dt_wer_penalty_pts (pts) | user_recall_nst (frac) | delay_est_err_ms (ms) | cpu_core_share (cores) | erle_music_loss_db (dB) | echo_ms_per_audio_s | echo_core_share | this machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxa | 14.96 [13.49, 16.31] | -1.62 [-2.95, -0.51] | 6.41 [4.33, 9.01] | 0.1151 [0.0754, 0.1548] | 140.03 [82.87, 209.87] | 0.9545 [0.9091, 0.9909] | 0.0044 [0.0027, 0.0065] | 0.0584 [0.0507, 0.0659] | 0.3141 [0.1284, 0.5235] | 51.5 | 0.112 |  |
| aec3 | 35.26 [32.23, 36.95] | 5.77 [3.17, 11.08] | 28.36 [23.41, 33.58] | 0.0595 [0.0357, 0.0873] | 68.36 [43.48, 99.72] | 0.9091 [0.8545, 0.9545] | 5.98 [0.05, 17.79] | 0.0348 [0.0315, 0.0382] | unmeasured | 32.4 | 0.176 |  |
| speex | 2.23 [2.00, 2.56] | 0.7600 [0.6800, 0.8730] | -1.11 [-2.36, 0.07] | 0.3175 [0.2658, 0.3770] | 79.25 [45.63, 130.30] | 0.9455 [0.9000, 0.9818] | 5.98 [0.05, 17.79] | 0.0134 [0.0121, 0.0146] | unmeasured | 13.6 | 0.079 |  |
| hybrid | 35.26 [32.23, 36.95] | 5.77 [3.17, 11.08] | 28.36 [23.41, 33.58] | 0.0595 [0.0357, 0.0873] | 68.36 [43.48, 99.72] | 0.9091 [0.8545, 0.9545] | 5.98 [0.05, 17.79] | 0.0334 [0.0308, 0.0360] | unmeasured | 31.2 | 0.168 |  |
| dtln | 16.14 [11.87, 20.31] | -0.8400 [-1.8374, 0.4500] | 8.07 [4.68, 11.33] | 0.0794 [0.0516, 0.1111] | 142.73 [54.63, 301.47] | 0.9455 [0.9000, 0.9818] | 5.98 [0.05, 17.79] | 0.0771 [0.0749, 0.0794] | n/a | 77.6 | 0.322 | DTLN-aec 512 weights via ai_edge_litert |
| nkf | 1.68 [0.20, 3.34] | -7.15 [-7.76, -6.09] | 3.58 [1.86, 5.77] | 0.2976 [0.2421, 0.3571] | 58.75 [44.38, 72.23] | 0.9545 [0.9091, 0.9909] | 5.98 [0.05, 17.79] | 0.5403 [0.5314, 0.5501] | n/a | 540.7 | 2.069 | NKF-AEC weights /home/rhy/voxa-bench-data/third_party/NKF-AEC-fjiang9/src/nkf_epoch70.pt via torch |

Smoke agrees with the full suite on the **ordering** of the arms on every column, but it flatters the
incumbent on the numbers: 14.96 dB of median echo removal on smoke vs 6.92 dB on the full suite, and
0.0044 ms of delay-estimate error vs 9.29 ms. Smoke is enough to screen a *new* candidate against the
others; it is not enough to quote the incumbent, and the full suite is what this file and any later
gate should be quoted from.

## What each arm actually is on this machine

* `voxa` - Voxa's NLMS canceller (4096 taps, mu 0.7) with continuous GCC-PHAT delay/drift tracking
  (f130) plus the f131 residual suppressor. The incumbent.
* `aec3` - WebRTC AEC3 through `pywebrtc-audio` 0.2.0 (the extension module statically links
  `webrtc::EchoCanceller3`, `EchoRemover`, `Subtractor`, `ErleEstimator`) plus the f131 residual
  suppressor. **The wheel exposes no `EchoCanceller3Config`**, so the delay, suppressor and
  comfort-noise settings are unreachable: this is AEC3 with whatever the wheel baked in, which is why
  it is the *baseline* of this table rather than the frontier of AEC3.
* `hybrid` - the best linear stage this machine has, then the best residual stage it has. On this
  machine that is AEC3 + the f131 suppressor, i.e. **the `aec3` arm itself**, and the table reports
  that instead of inventing a difference: 31.43 dB / 0.0498 / 84.17 on both rows. The hybrid the task
  asked for (linear stage + *neural* residual stage) needs weights that are not reachable here; the
  only measurable difference between the two rows is the bookkeeping cost, 33.9 vs 33.8 ms per second
  of audio and 0.194 vs 0.154 of a core.
* `speex` - the SpeexDSP MDF echo canceller and its own residual suppressor, over the installed
  system `libspeexdsp.so.1` (Debian 1.2.1-4) via ctypes. Measured for real, with two caveats that
  belong to the measurement: this build's `speex_echo_ctl` exposes only `max_filter_length` and
  `alpha` (its delay, double-talk, soft-gain and residual-leak settings are not in the switch), and
  the request upstream calls `TAU` (29) corrupts the state - the process dies in
  `speex_echo_state_destroy` - so the arm never sends it. The MDF therefore searches the delay inside
  its own filter span instead of taking a delay Voxa already knows.
* `dtln` - DTLN-aec (MIT, breizhn, commit 9d24e12) 512-unit model, run offline with the installed
  LiteRT (`ai-edge-litert`) interpreter over the six prebuilt `.tflite` weights (h01 could not run it:
  no `tflite-runtime` wheel for this Python 3.14, no LiteRT then). The bench drives it through the
  repo's own `run_aec.py` contract (16 kHz, block 512 / shift 128, stage-1 + stage-2); the streaming
  port is validated against the repo's `*_processed.wav` (corr 1.0). It removes 24.41 dB median echo
  for 7.86 dB of user attenuation at 0.232 of a core.
* `nkf` - NKF-AEC (fjiang9, commit 8ac58fb; no LICENSE) linear-only complex Kalman filter (L=4, STFT
  1024/256) with the published `nkf_epoch70.pt`, run on CPU with torch. The streaming math is
  *identical* to the repo's offline inference (chunk-independent: streaming == single-call, corr 1.0),
  but the bench returns it one STFT frame late because a 320-sample chunk is shorter than the 1024
  frame. It is measured but expensive: 850 ms of CPU per second of audio, 2.1 cores.
* `aec3full` - **still unmeasured.** No AEC3-full shim could be built here: the freedesktop
  `webrtc-audio-processing` 1.3 source's `echo_remover.h` needs Abseil C++ headers (`libabsl-dev`) that
  are absent and un-installable offline, and no PyPI binding exposing the full `EchoCanceller3Config`
  is published for Python 3.14. `SOURCES.md` under `/home/rhy/voxa-bench-data/third_party/` records the
  tarball, its SHA-256, and the bindings tried. The bench refuses rather than substituting another
  arm: an arm that cannot run here would report the other arm's numbers under this arm's name.

## Pareto frontier

Two axes, as the Hearing 2 brief defines them: echo removal (`erle_fst_median_db` up,
`echo_leak_word_rate` and `dt_wer_penalty_pts` down) against user-voice preservation
(`user_recall_nst` up, `nearend_atten_dt_db` down). A mode is on the frontier when nothing measured
beats it on both.

Full suite, the trade in one line: **speex** leaves the near end almost untouched (-0.59 dB) but
removes almost no echo (2.14 dB median, 0.4669 of the echo words still reach the transcript, 149.04
points of double-talk WER penalty); **aec3/hybrid** removes the echo best (31.43 dB, 0.0498 leak,
84.17 points) but attenuates the user by 24.96 dB; **dtln** now sits between AEC3 and the incumbent on
both axes (24.41 dB median echo removal - only 7 dB short of AEC3 - for 7.86 dB of user attenuation,
0.0786 leak, 0.9339 recall, at 0.232 of a core); **nkf** is near-end-safe (3.03 dB) but removes almost
nothing (1.33 dB median, -7.39 dB p10) and is the most expensive arm here; **voxa** stays in the middle
(6.92 dB, 0.1911 leak, 128.64 points, 9.37 dB of user attenuation) with the best recall of anything
that removes echo (0.9507).

* **The brief's target - ERLE >= 20 dB at <= 6 dB of user attenuation - is not reached by anything
  measured, and `dtln` comes closest:** 24.41 dB echo removal (clears the ERLE bar) at 7.86 dB of user
  attenuation, only 1.86 dB over the attenuation bar. `aec3` clears ERLE by far (31.43) but blows the
  attenuation bar by 4x (24.96 dB); `nkf` clears attenuation easily (3.03 dB) but removes only 1.33 dB
  of echo. So DTLN is the best candidate found: it is the only measured arm with AEC3-class echo
  removal and single-digit user damage, and it is cheap (0.232 of a core, no 67 MB ONNX - LiteRT
  interpreter only).
* The frontier is now a three-ended trade, not two: SpeexDSP (never deletes the user, removes ~nothing),
  AEC3 (best echo removal, deletes the user by 25 dB), and DTLN (near-AEC3 removal at near-incumbent
  cost to the user). DTLN is not beaten on echo removal at its attenuation, and is beaten only by
  AEC3/hybrid, which pay 24.96 dB of user damage to do so.
* All three of `aec3`/`hybrid`, `speex` and `voxa` remain on the frontier as before: `aec3`/`hybrid`
  is not beaten on echo removal, `speex` is not beaten on user preservation, and `voxa` is beaten on
  neither axis - `aec3` damages the user more (24.96 vs 9.37 dB), `speex` removes less echo (2.14 vs
  6.92 dB median, 0.4669 vs 0.1911 leak).
* The incumbent's position is weaker than the smoke set suggested: on the full suite `voxa` leaves
  0.1911 of the echo words in the transcript (four times `aec3`'s 0.0498) and its `erle_fst_p10` is
  **-5.00 dB**, i.e. on the worst tenth of clips the NLMS made the echo *louder*. The `drift_erle_loss_db`
  report for that run is 6.37 dB (over its 0.154 dB threshold): the damage is concentrated in the
  delay-drift clips.

Two measurement notes that must be read with the table, not after it:

* `delay_est_err_ms` is **not an arm-vs-arm comparison**. For `voxa` it measures the canceller's own
  tracked delay (375 of 480 clips), and it is 9.29 ms - worse than the bench's own estimate, not
  better. For `aec3`, `hybrid` and `speex` the arm does not estimate delay at all, so all three rows
  carry the bench's own GCC-PHAT figure (6.36 ms, 303 clips). `dtln` and `nkf` likewise manage no delay
  of their own; their rows carry the bench figure over the clips that arm reached (10.98 ms). Voxa's
  delay tracking is a real advantage in principle and this suite does not show it.
* `user_recall_nst` ties exactly for `voxa` and `speex` (0.9507, identical interval) even though
  their transcripts differ on 20 of 60 NST clips: the metric scores whether the user's utterance was
  heard as a turn, not the transcript's wording, so both arms preserve the near end on clips with no
  echo to cancel. `aec3`'s 0.9267 is the arm that damages the user before any echo is even present.

## Recommendation for Voxa's echo stage

1. **Keep Voxa's own NLMS + f131 residual suppressor (`voxa`) as the shipping echo stage, but do not
   call it solved.** It is on the frontier, it preserves the user best of the arms that remove echo
   (9.37 dB attenuation, 0.9507 recall), and it needs no new dependency. But on the full suite it
   leaves 0.1911 of the echo words in the transcript and -5.00 dB of `erle_fst_p10`: on its worst
   clips it amplifies echo. A leak gate tuned against `aec3`'s 0.0498 would fail it today.
2. **Do not switch to AEC3 as the wheel stands.** Its echo removal is real (31.43 dB, 0.0498 leak,
   84.17 points) but 24.96 dB of `nearend_atten_dt_db` is exactly the failure the Hearing 2 brief is
   about (the target is 3 dB), and with no config exposure there is no knob to pull it back.
   `hybrid` inherits the same problem because on this machine it *is* that chain.
3. **`dtln` is now the strongest measured candidate and the real lead for the echo stage.** DTLN-aec
   512 removes 24.41 dB median (only 7 dB under AEC3) for 7.86 dB of user attenuation, 0.0786 leak and
   0.9339 recall, at 0.232 of a core - the best echo-removal-per-unit-of-user-damage of anything here,
   and it needs only the LiteRT interpreter, not the 67 MB ONNX runtime. It is still 1.86 dB over the
   brief's 6 dB attenuation target, so it does not *pass* the target, but no measured arm is closer.
   Treat it as the next shipping-stage experiment, and as a residual stage on top of the best linear
   stage in the `hybrid` arm.
4. **`nkf` is measured and is not a candidate here.** It preserves the near end (3.03 dB attenuation)
   but removes only 1.33 dB median (-7.39 dB p10) and costs 850 ms of CPU per second of audio, 2.1
   cores - it does not reach real time on this CPU. Its streaming is faithful (identical to offline
   inference), so this is the algorithm, not a porting bug. The linear Kalman filter needs echo it
   simply is not modelling on these clips.
5. **`aec3full` remains the one unfilled row, and it is still the next *measurement*, not a new
   algorithm.** A freedesktop `webrtc-audio-processing` build with `EchoCanceller3Config` reachable
   (delay from the render timeline, suppressor/comfort-noise aimed at double talk) is the candidate
   that could push AEC3's 31.43 dB down toward DTLN's user cost; the gap to close is quantified:
   recover `aec3`'s 0.0498 leak without its 24.96 dB of user damage. It cannot be built on this box
   (Abseil headers absent and un-installable offline; no cp314 config-exposing binding published) -
   see `SOURCES.md`. Until then the `aec3` row is a lower bound on AEC3, not a ceiling.
6. `speex` is not a candidate for Voxa's stage: 2.14 dB of median echo removal and 0.4669 leak is not
   echo cancellation, and this build cannot be given the delay Voxa already knows. It earns its place
   in this table as the near-end-safe end of the trade and as the cheapest stage measured
   (12.8 ms per second of audio, 0.078 of a core) - the arm any future chain should be measured
   against on cost.

## Frozen-file note

`voxa/bench/render.py`, `voxa/bench/metrics.py` and the clip set are untouched by this task; no
threshold in `voxa/bench/gates.toml` was changed, no clip was skipped or re-seeded, and no arm's
numbers were improved by editing the metric that reads them. `dtln` and `nkf` are now measured on the
full suite (h03 fetched DTLN-aec and NKF-AEC weights/code and wired both behind the bench); `aec3full`
is still reported unmeasured with its reason rather than passed off as measured. The new code is
`voxa/bench/modes.py`, the arm wiring in `voxa/bench/runner.py`, `tools/hearing_frontier_table.py` and
the tests in `tests/test_echo_modes_speex.py` / `tests/test_echo_modes_probe.py`.

## h04 shipped: DTLN-aec is now Voxa's live echo stage (issue #72)

h04 promotes the h03 winner to the shipping default. The `dtln` bench arm and the live capture path
are now the **same code** — `voxa/dtln.py`'s streaming `DtlnCanceller`, the far-end reference being the
exact samples Voxa plays (`delay_managed_internally`), 16 kHz, block 512 / shift 128, LSTM state kept
across `process` calls. The weights ship as **ONNX run through `onnxruntime`** (`voxa/data/dtln/`,
MIT, provenance in that folder's `SOURCES.md`), not the LiteRT `.tflite` h03 measured; they are the same
model, and the whole-file `run_aec.py` reconstruction through ONNX reproduces the repo's
`*_processed.wav` (double-talk corr 1.00000, max |diff| 4.8e-4), so every h03 DTLN number carries over.
`dtln` is the default `echo_mode`; when onnxruntime or a weight is missing the app logs once and falls
back to Voxa's own NLMS `voxa` stage (see `docs/hearing/H04-DTLN.md`).

How to regenerate these rows: `/Voxa/.venv/bin/python3 -m voxa.bench run --suite full --seed 0
--echo-mode dtln --out .voxa-bench/frontier-dtln-full.json --no-gates --quiet`, then
`/Voxa/.venv/bin/python3 tools/hearing_frontier_table.py .voxa-bench/frontier-dtln-full.json`
(the smoke rows the same way with `--suite smoke`).

### Live stage on the full suite (480 clips, seed 0) — next to h03's measured DTLN arm

| mode | erle_fst_median_db | erle_fst_p10_db | nearend_atten_dt_db | echo_leak_word_rate | dt_wer_penalty_pts | user_recall_nst | delay_est_err_ms | cpu_core_share | echo_ms_per_audio_s | echo_core_share | what it is |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dtln (h04 live, ONNX) | 24.39 [23.72, 25.00] | 2.26 [1.73, 2.82] | 7.85 [6.76, 9.03] | 0.0742 [0.0666, 0.0818] | 125.07 [81.21, 197.50] | 0.9291 [0.9111, 0.9459] | 10.98 [7.56, 14.48] | 0.0996 [0.0983, 0.1007] | 99.9 | 0.369 | `voxa/dtln.py` DTLN-aec 512 ONNX via onnxruntime, single-thread |
| dtln (h03, LiteRT) | 24.41 [23.72, 25.03] | 2.19 [1.74, 2.87] | 7.86 [6.77, 9.04] | 0.0786 [0.0708, 0.0862] | 98.64 [80.89, 120.47] | 0.9339 [0.9159, 0.9495] | 10.98 [7.56, 14.48] | 0.0894 [0.0881, 0.0907] | 90.2 | 0.232 | DTLN-aec 512 weights via `ai_edge_litert` |

The live ONNX stage lands on h03's measured row inside every bootstrap interval (median ERLE 24.39 vs
24.41, user attenuation 7.85 vs 7.86, echo leak 0.0742 vs 0.0786). The cost column moves a little
(0.0996 vs 0.0894 cores, 99.9 vs 90.2 ms/audio-s): `onnxruntime` single-threaded is a touch heavier than
LiteRT on this CPU, still well under the two real-time incumbents' neighbours and far under NKF-AEC. The
live stage is what the app runs, so the app *is* the h03 winner.

### Live stage on smoke (52 clips, seed 0) — the fast arm check

| mode | erle_fst_median_db | nearend_atten_dt_db | echo_leak_word_rate | user_recall_nst | cpu_core_share | echo_ms_per_audio_s | echo_core_share |
|---|---|---|---|---|---|---|---|
| dtln (h04 live, ONNX) | 16.15 [11.76, 20.34] | 8.04 [4.64, 11.30] | 0.0754 [0.0476, 0.1071] | 0.9455 [0.9000, 0.9818] | 0.0937 [0.0915, 0.0961] | 94.4 | 0.344 |
| dtln (h03, LiteRT) | 16.14 [11.87, 20.31] | 8.07 [4.68, 11.33] | 0.0794 [0.0516, 0.1111] | 0.9455 [0.9000, 0.9818] | 0.0771 [0.0749, 0.0794] | 77.6 | 0.322 |

Latency (step 4, this CPU, onnxruntime single-thread, the live setting): per-block p50 0.55 ms / p99
1.01 ms idle, p50 1.09 ms / p99 2.00 ms under full-machine load — the stream advances 8.00 ms per block,
so the stage never falls behind. `tests/test_dtln_stage.py::test_per_block_latency_stays_under_the_frame_shift`
asserts p99 < the frame shift.
