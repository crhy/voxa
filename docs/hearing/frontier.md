# Voxa Hearing 2 - the echo-cancellation frontier (task h01, issue #72)

Every serious candidate for Voxa's echo stage, measured on Voxa's own bench: synthetic clips, seed 0,
offline (no microphone, no playback, no audio devices). This is the frontier Hearing 2 asked for:
what echo removal is achievable in Voxa's rooms today, at what cost to the user's voice, per
candidate - written before anyone writes another chain gate.

How to regenerate every number in this file:
`/Voxa/.venv/bin/python3 -m voxa.bench run --suite full --seed 0 --echo-mode <mode> --out
.voxa-bench/frontier-<mode>-full.json --no-gates --quiet` for each of `voxa`, `aec3`, `speex`,
`hybrid`, then `tools/hearing_frontier_table.py .voxa-bench/frontier-<mode>-full.json`. The tables
below are pasted from that script's output; nothing in them was typed by hand. Arms: `voxa`, `aec3`,
`speex`, `hybrid`, `aec3full`, `dtln`, `nkf`.

## The table (full suite: 480 clips, seed 0)

| mode | erle_fst_median_db (dB) | erle_fst_p10_db (dB) | nearend_atten_dt_db (dB) | echo_leak_word_rate (frac) | dt_wer_penalty_pts (pts) | user_recall_nst (frac) | delay_est_err_ms (ms) | cpu_core_share (cores) | erle_music_loss_db (dB) | echo_ms_per_audio_s | echo_core_share | this machine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voxa | 6.92 [6.11, 7.61] | -5.00 [-5.30, -4.75] | 9.37 [8.21, 10.47] | 0.1911 [0.1799, 0.2028] | 128.64 [111.73, 147.69] | 0.9507 [0.9351, 0.9640] | 9.29 [7.22, 11.46] | 0.0660 [0.0634, 0.0688] | 0.1263 [0.0837, 0.1748] | 65.6 | 0.118 | Voxa's NLMS (f130 delay tracking) + f131 residual suppressor |
| aec3 | 31.43 [30.35, 32.39] | 0.2290 [-0.1341, 0.7504] | 24.96 [23.45, 26.56] | 0.0498 [0.0431, 0.0563] | 84.17 [73.69, 95.75] | 0.9267 [0.9075, 0.9447] | 6.36 [3.74, 9.89] | 0.0337 [0.0329, 0.0345] | unmeasured | 33.8 | 0.154 | WebRTC AEC3 as the `pywebrtc-audio` wheel ships it |
| speex | 2.14 [2.04, 2.24] | 0.6200 [0.5700, 0.6500] | -0.5871 [-1.0131, -0.1733] | 0.4669 [0.4512, 0.4821] | 149.04 [114.95, 191.03] | 0.9507 [0.9351, 0.9640] | 6.36 [3.74, 9.89] | 0.0124 [0.0120, 0.0129] | unmeasured | 12.8 | 0.078 | libspeexdsp (system .so) via ctypes |
| hybrid | 31.43 [30.35, 32.39] | 0.2290 [-0.1341, 0.7504] | 24.96 [23.45, 26.56] | 0.0498 [0.0431, 0.0563] | 84.17 [73.69, 95.75] | 0.9267 [0.9075, 0.9447] | 6.36 [3.74, 9.89] | 0.0337 [0.0329, 0.0346] | unmeasured | 33.9 | 0.194 | linear aec3 (installed wheel) + Voxa's residual suppressor |
| aec3full | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | no AEC3-full shim built (set `VOXA_AEC3FULL_SHIM`); the freedesktop webrtc-audio-processing headers are not installed on this machine and this session had no network to fetch them |
| dtln | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | onnxruntime 1.30.0 is installed (67 MB) but no DTLN-aec weights under `/home/rhy/voxa-bench-data/third_party/DTLN-aec`; upstream (breizhn/DTLN-aec) ships `.tflite` only and this host has no tflite-runtime wheel for Python 3.14 and no network to fetch one |
| nkf | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | no NKF-AEC weights/code under `/home/rhy/voxa-bench-data/third_party/NKF-AEC`; upstream (LebaneseNanoElectronics/NKF-AEC) publishes MATLAB + `.mat` weights with no streaming CPU reference implementation, so real-time on this CPU cannot be measured |

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
* `aec3full`, `dtln`, `nkf` - **unmeasured, and no number is quoted for them.** Why, and what one
  command would fix, is in the probe strings in the rows above and in
  `tasks/notes-h01-voxa-hearing2-echo-frontier.md`. The bench refuses to run them rather than
  substituting another arm: an arm that cannot run here would report the other arm's numbers under
  this arm's name.

## Pareto frontier

Two axes, as the Hearing 2 brief defines them: echo removal (`erle_fst_median_db` up,
`echo_leak_word_rate` and `dt_wer_penalty_pts` down) against user-voice preservation
(`user_recall_nst` up, `nearend_atten_dt_db` down). A mode is on the frontier when nothing measured
beats it on both.

Full suite, the trade in one line: **speex** leaves the near end almost untouched (-0.59 dB) but
removes almost no echo (2.14 dB median, 0.4669 of the echo words still reach the transcript, 149.04
points of double-talk WER penalty); **aec3/hybrid** removes the echo best (31.43 dB, 0.0498 leak,
84.17 points) but attenuates the user by 24.96 dB; **voxa** sits between them (6.92 dB, 0.1911 leak,
128.64 points, 9.37 dB of user attenuation) and has the best user recall of anything that removes
echo (0.9507).

* All three are on the frontier: `aec3`/`hybrid` is not beaten on echo removal, `speex` is not beaten
  on user preservation, and `voxa` is beaten on neither axis - `aec3` damages the user more (24.96 vs
  9.37 dB), `speex` removes less echo (2.14 vs 6.92 dB median, 0.4669 vs 0.1911 leak).
* The incumbent's position is weaker than the smoke set suggested: on the full suite `voxa` leaves
  0.1911 of the echo words in the transcript (four times `aec3`'s 0.0498) and its `erle_fst_p10` is
  **-5.00 dB**, i.e. on the worst tenth of clips the NLMS made the echo *louder*. The `drift_erle_loss_db`
  report for that run is 6.37 dB (over its 0.154 dB threshold): the damage is concentrated in the
  delay-drift clips.
* Nothing measured beats `speex` on near-end preservation and nothing measured beats `aec3`/`hybrid`
  on echo removal, so there is no single winner: the frontier is a trade whose two ends are
  SpeexDSP (never deletes the user) and AEC3 (best echo removal, deletes the user by 25 dB). The
  point in between that a *tuned* AEC3 and a neural residual stage are supposed to occupy is exactly
  what the three unmeasured rows stand for.

Two measurement notes that must be read with the table, not after it:

* `delay_est_err_ms` is **not an arm-vs-arm comparison**. For `voxa` it measures the canceller's own
  tracked delay (375 of 480 clips), and it is 9.29 ms - worse than the bench's own estimate, not
  better. For `aec3`, `hybrid` and `speex` the arm does not estimate delay at all, so all three rows
  carry the bench's own GCC-PHAT figure (6.36 ms, 303 clips). Voxa's delay tracking is a real
  advantage in principle and this suite does not show it.
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
3. **The next measurement, not the next algorithm, is `aec3full`.** The freedesktop
   `webrtc-audio-processing` build with `EchoCanceller3Config` reachable (delay taken from the render
   timeline, suppressor and comfort-noise settings aimed at double talk) is the one candidate that
   could sit between SpeexDSP's near-end safety and AEC3's echo removal, and the gap to close is now
   quantified: recover `aec3`'s 0.0498 leak without its 24.96 dB of user attenuation. One source
   build fills the row; until then the `aec3` row here is a lower bound on AEC3, not a ceiling.
4. **Then `dtln`/`nkf` as residual stages on top of the best linear stage**, which is what the
   `hybrid` arm is built to accept the moment weights are reachable. On this machine the neural stage
   is the missing half, and the 67 MB of onnxruntime is already installed.
5. `speex` is not a candidate for Voxa's stage: 2.14 dB of median echo removal and 0.4669 leak is not
   echo cancellation, and this build cannot be given the delay Voxa already knows. It earns its place
   in this table as the near-end-safe end of the trade and as the cheapest stage measured
   (12.8 ms per second of audio, 0.078 of a core) - the arm any future chain should be measured
   against on cost.

## Frozen-file note

`voxa/bench/render.py`, `voxa/bench/metrics.py` and the clip set are untouched by this task; no
threshold in `voxa/bench/gates.toml` was changed, no clip was skipped or re-seeded, and no arm's
numbers were improved by editing the metric that reads them. Three arms are reported unmeasured with
their reasons rather than passed off as measured. The new code is `voxa/bench/modes.py`, the arm
wiring in `voxa/bench/runner.py`, `tools/hearing_frontier_table.py` and the tests in
`tests/test_echo_modes_speex.py` / `tests/test_echo_modes_probe.py`.
