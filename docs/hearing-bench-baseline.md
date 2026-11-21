# Voxa hearing bench: the baseline of today, and the ratchet that keeps it from getting worse

Recorded 2026-11-18 on `strata/f127-voxa-hear-b5-baseline-ratchet` (roadmap **B5**, issue crhy/voxa#72).
This is the "what Voxa's hearing does today, measured honestly" record: `voxa/bench/baseline.json` holds
the numbers, `voxa/bench/gates.toml` holds the ratchet derived from them, and this page is the readable
version of both. Every later hearing task is judged against it: a change that does not move a number here
did not improve her hearing, and a change that moves a number the wrong way fails the gate.

Nothing here is a fix. B5 measures; the fixes are roadmap tasks 1-13.

## How to reproduce it, and what it costs

```
python -m voxa.bench render --suite full --seed 0                        # 25.6 s (sources cached)
python -m voxa.bench run --suite smoke --seed 0 --echo-mode off    --no-gates --out .voxa-bench/f127-smoke-off.json
python -m voxa.bench run --suite smoke --seed 0 --echo-mode voxa  --no-gates --out .voxa-bench/f127-smoke-voxa.json
python -m voxa.bench run --suite smoke --seed 0 --echo-mode webrtc --no-gates --out .voxa-bench/f127-smoke-webrtc.json
python -m voxa.bench run --suite full  --seed 0 --echo-mode off    --no-gates --out .voxa-bench/f127-full-off.json
python -m voxa.bench run --suite full  --seed 0 --echo-mode voxa   --no-gates --out .voxa-bench/f127-full-voxa.json
python -m voxa.bench run --suite full  --seed 0 --echo-mode webrtc --no-gates --out .voxa-bench/f127-full-webrtc.json
python -m voxa.bench baseline record .voxa-bench/f127-*.json --out voxa/bench/baseline.json
```

* smoke = 48 clips = 252.1 s of clip audio; full = 480 clips = 2963.4 s of clip audio.
* Wall times of those runs: smoke 53.0 / 33.1 / 30.6 s, full 372.2 / 379.0 / 323.6 s, so the bench runs at
  **0.11-0.21x real time** (target <= 0.5x, roadmap B2's gate) - comfortably inside the budget; the whole
  baseline (render + 6 runs) took **19.8 minutes** on this machine, which is why it runs in the background.
  A third repeat of `run --suite smoke --echo-mode voxa` (34.1 s wall) is also recorded; `baseline.json`
  keeps that repeat as the `voxa`/smoke entry because it is the worst of the three on the one metric that
  moves between identical runs (`cpu_core_share`) - see the ratchet section.
* Re-rendering the full suite with seed 0 produced **byte-identical** files
  (`md5` over every file of all 480 clips: `9e6fc5550ba6a43c6c9661153abf3183` before and after), and three
  identical `run`s of the same clips differ in exactly one metric, `cpu_core_share` (0.7270 / 0.7300 / 0.7310).
  Everything else is reproducible to the digit.

## The three echo modes as they exist today

| mode | what runs | note |
|---|---|---|
| `off` | capture → VAD → barge-in gate → ASR, no echo stage | the pipeline without any cancellation |
| `voxa` | `EchoCanceller` (NLMS, `voxa/anc.py`) + `ReferenceMask` (`voxa/refmask.py`) | **the runner's default** (`voxa/bench/runner.py:631`), so this is the mode the ratchet is taken from |
| `webrtc` | GStreamer `webrtcechoprobe` route (`voxa/audio.py:_start_webrtc`) | **not available on this machine**: `webrtc_available()` is False (no `gi`, no GStreamer runtime), so the runner records the live app's own fallback (`voxa/audio.py:519 set_echo_mode`) — `run.degraded = "webrtc unavailable (no GStreamer runtime here): running with echo off"`. Its numbers are therefore the `off` numbers with the degradation note, **not** a WebRTC measurement. Roadmap task 2 (AEC3 in-process) is what will produce a real WebRTC-column entry. |

In `voxa` mode the pipeline produced **0 suspects, 0 barge-ins and 0 own-voice rejections** over all 48
smoke clips (`run.suspects`, `run.barge_ins`, `run.own_voice_rejections`), against 47 suspects and 17
barge-ins in `off` mode. That single contrast is the finding of this baseline: with Voxa's own canceller
and mask in the path, the barge-in gate never even opens.

## Every metric, every mode

| metric | direction | echo off | echo voxa (default) | echo webrtc |
|---|---|---|---|---|
| `align_err_p90_ms` | <= | null | null | null |
| `backchannel_resume_rate` | >= | 0.5000 [0.0000, 1.0000] (n=2) | 1.0000 [1.0000, 1.0000] (n=2) | 0.5000 [0.0000, 1.0000] (n=2) |
| `cpu_core_share` | <= | 0.9820 [0.9792, 0.9846] (n=48) | 0.7310 [0.6840, 0.7752] (n=48) | 0.9769 [0.9736, 0.9800] (n=48) |
| `delay_est_err_ms` | <= | null | 5.8011 [0.0114, 17.3580] (n=11) | null |
| `drift_erle_loss_db` | <= | 0.0000 [0.0000, 0.0000] (n=264) | 0.0800 [0.0160, 0.1536] (n=264) | 0.0000 [0.0000, 0.0000] (n=264) |
| `dt_wer_penalty_pts` | <= | 54.4862 [39.2592, 68.5204] (n=33) | 125.2365 [27.3710, 352.4475] (n=33) | 54.4862 [39.2592, 68.5204] (n=33) |
| `duck_latency_p50_ms` | <= | 1212.3697 [538.0369, 1667.2974] (n=11) | null | 1212.3697 [538.0369, 1667.2974] (n=11) |
| `duck_latency_p90_ms` | <= | 1681.1150 [1301.6903, 2040.0000] (n=11) | null | 1681.1150 [1301.6903, 2040.0000] (n=11) |
| `echo_leak_word_rate` | <= | 0.3957 [0.3348, 0.4609] (n=230) | 0.1652 [0.1174, 0.2130] (n=230) | 0.3957 [0.3348, 0.4609] (n=230) |
| `ep50_ms` | <= | 828.0322 [767.2424, 880.0000] (n=30) | 913.8869 [855.5428, 1021.4147] (n=28) | 828.0322 [767.2424, 880.0000] (n=30) |
| `ep90_ms` | <= | 983.5861 [911.3631, 1020.3970] (n=30) | 1159.4533 [1081.4575, 1318.9205] (n=28) | 983.5861 [911.3631, 1020.3970] (n=30) |
| `erle_fst_median_db` | >= | 0.0000 [0.0000, 0.0000] (n=372) | 0.0000 [0.0000, 0.0000] (n=372) | 0.0000 [0.0000, 0.0000] (n=372) |
| `erle_fst_p10_db` | >= | 0.0000 [0.0000, 0.0000] (n=372) | -0.0380 [-0.0800, -0.0100] (n=372) | 0.0000 [0.0000, 0.0000] (n=372) |
| `erle_music_loss_db` | >= | null | null | null |
| `false_bargein_per_hour` | <= | 132.7780 [0.0000, 296.1552] (n=42) | 0.0000 [0.0000, 0.0000] (n=42) | 132.7780 [0.0000, 296.1552] (n=42) |
| `first_word_trunc_bargein` | <= | 1.0000 [1.0000, 1.0000] (n=14) | null | 1.0000 [1.0000, 1.0000] (n=14) |
| `first_word_trunc_cold` | <= | 1.0000 [1.0000, 1.0000] (n=3) | 1.0000 [1.0000, 1.0000] (n=3) | 1.0000 [1.0000, 1.0000] (n=3) |
| `fst_transcript_rate` | <= | 0.0000 [0.0000, 0.0000] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) |
| `interrupt_stop_rate` | <= | 0.2222 [0.0000, 0.5556] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) | 0.2222 [0.0000, 0.5556] (n=9) |
| `missed_bargein_rate` | <= | 0.8148 [0.6296, 0.9630] (n=27) | 1.0000 [1.0000, 1.0000] (n=27) | 0.8148 [0.6296, 0.9630] (n=27) |
| `nearend_atten_dt_db` | <= | 0.0000 [0.0000, 0.0000] (n=31) | 8.6370 [5.9698, 11.7765] (n=31) | 0.0000 [0.0000, 0.0000] (n=31) |
| `pair_accuracy` | >= | 0.3333 [0.0000, 1.0000] (n=3) | 0.3333 [0.0000, 1.0000] (n=3) | 0.3333 [0.0000, 1.0000] (n=3) |
| `premature_cutoff_rate` | <= | 0.0667 [0.0000, 0.1667] (n=30) | 0.0714 [0.0000, 0.1786] (n=28) | 0.0667 [0.0000, 0.1667] (n=30) |
| `protected_word_keep_rate` | >= | 0.5455 [0.4091, 0.6818] (n=44) | 0.2273 [0.1136, 0.3636] (n=44) | 0.5455 [0.4091, 0.6818] (n=44) |
| `reconverge_s` | <= | null | null | null |
| `self_command_rate` | <= | 0.0000 [0.0000, 0.0000] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) |
| `self_pause_per_hour` | <= | 484.5847 [0.0000, 891.3341] (n=9) | 0.0000 [0.0000, 0.0000] (n=9) | 484.5847 [0.0000, 891.3341] (n=9) |
| `stop_latency_p50_ms` | <= | 1212.3697 [538.0369, 1667.2974] (n=11) | null | 1212.3697 [538.0369, 1667.2974] (n=11) |
| `stop_latency_p95_ms` | <= | 1860.5575 [1491.4027, 2040.0000] (n=11) | null | 1860.5575 [1491.4027, 2040.0000] (n=11) |
| `transport_drop_count` | <= | null | null | null |
| `turn_retention_rate` | >= | 0.0000 [0.0000, 0.0000] (n=17) | null | 0.0000 [0.0000, 0.0000] (n=17) |
| `user_recall_nst` | >= | 0.9455 [0.9000, 0.9818] (n=110) | 0.6091 [0.5091, 0.6909] (n=110) | 0.9455 [0.9000, 0.9818] (n=110) |
| `user_word_deletion_rate` | <= | 0.2953 [0.2513, 0.3394] (n=386) | 0.3161 [0.2720, 0.3601] (n=386) | 0.2953 [0.2513, 0.3394] (n=386) |

| metric | direction | echo off | echo voxa (default) | echo webrtc |
|---|---|---|---|---|
| `align_err_p90_ms` | <= | null | null | null |
| `backchannel_resume_rate` | >= | 0.6500 [0.4487, 0.8500] (n=20) | 1.0000 [1.0000, 1.0000] (n=20) | 0.6500 [0.4487, 0.8500] (n=20) |
| `cpu_core_share` | <= | 0.9778 [0.9769, 0.9787] (n=480) | 0.6483 [0.6317, 0.6638] (n=480) | 0.9762 [0.9752, 0.9773] (n=480) |
| `delay_est_err_ms` | <= | null | 6.3616 [3.7355, 9.8891] (n=303) | null |
| `drift_erle_loss_db` | <= | 0.0000 [0.0000, 0.0000] (n=3268) | 0.0265 [-0.0402, 0.0903] (n=3268) | 0.0000 [0.0000, 0.0000] (n=3268) |
| `dt_wer_penalty_pts` | <= | 67.3118 [62.1806, 72.2585] (n=330) | 116.4305 [19.6646, 265.5194] (n=330) | 67.3118 [62.1806, 72.2585] (n=330) |
| `duck_latency_p50_ms` | <= | 1504.1014 [1276.2369, 1694.4849] (n=76) | null | 1504.1014 [1276.2369, 1694.4849] (n=76) |
| `duck_latency_p90_ms` | <= | 2335.1854 [1782.2921, 3666.5074] (n=76) | null | 2335.1854 [1782.2921, 3666.5074] (n=76) |
| `echo_leak_word_rate` | <= | 0.5473 [0.5321, 0.5623] (n=4014) | 0.1839 [0.1719, 0.1956] (n=4014) | 0.5473 [0.5321, 0.5623] (n=4014) |
| `ep50_ms` | <= | 848.9710 [820.4874, 870.8674] (n=231) | 913.3687 [883.7391, 952.2194] (n=208) | 848.9710 [820.4874, 870.8674] (n=231) |
| `ep90_ms` | <= | 1028.8506 [987.0828, 1053.4906] (n=231) | 1214.0567 [1147.2543, 1282.0665] (n=208) | 1028.8506 [987.0828, 1053.4906] (n=231) |
| `erle_fst_median_db` | >= | 0.0000 [0.0000, 0.0000] (n=3720) | 0.0000 [0.0000, 0.0000] (n=3720) | 0.0000 [0.0000, 0.0000] (n=3720) |
| `erle_fst_p10_db` | >= | 0.0000 [0.0000, 0.0000] (n=3720) | -0.0310 [-0.0600, -0.0200] (n=3720) | 0.0000 [0.0000, 0.0000] (n=3720) |
| `erle_music_loss_db` | >= | null | null | null |
| `false_bargein_per_hour` | <= | 122.0787 [93.1084, 149.8391] (n=420) | 0.0000 [0.0000, 0.0000] (n=420) | 122.0787 [93.1084, 149.8391] (n=420) |
| `first_word_trunc_bargein` | <= | 0.9489 [0.9124, 0.9781] (n=137) | null | 0.9489 [0.9124, 0.9781] (n=137) |
| `first_word_trunc_cold` | <= | 0.9667 [0.9000, 1.0000] (n=30) | 1.0000 [1.0000, 1.0000] (n=30) | 0.9667 [0.9000, 1.0000] (n=30) |
| `fst_transcript_rate` | <= | 0.0000 [0.0000, 0.0000] (n=90) | 0.0333 [0.0000, 0.0667] (n=90) | 0.0000 [0.0000, 0.0000] (n=90) |
| `interrupt_stop_rate` | <= | 0.1778 [0.1000, 0.2556] (n=90) | 0.0000 [0.0000, 0.0000] (n=90) | 0.1778 [0.1000, 0.2556] (n=90) |
| `missed_bargein_rate` | <= | 0.9333 [0.9037, 0.9594] (n=270) | 1.0000 [1.0000, 1.0000] (n=270) | 0.9333 [0.9037, 0.9594] (n=270) |
| `nearend_atten_dt_db` | <= | 0.0000 [0.0000, 0.0000] (n=285) | 5.1932 [4.1952, 6.2421] (n=285) | 0.0000 [0.0000, 0.0000] (n=285) |
| `pair_accuracy` | >= | 0.0000 [0.0000, 0.0000] (n=3) | 0.0000 [0.0000, 0.0000] (n=3) | 0.0000 [0.0000, 0.0000] (n=3) |
| `premature_cutoff_rate` | <= | 0.0260 [0.0087, 0.0476] (n=231) | 0.0433 [0.0192, 0.0721] (n=208) | 0.0260 [0.0087, 0.0476] (n=231) |
| `protected_word_keep_rate` | >= | 0.3600 [0.3138, 0.4123] (n=325) | 0.1723 [0.1323, 0.2154] (n=325) | 0.3600 [0.3138, 0.4123] (n=325) |
| `reconverge_s` | <= | null | 1.2689 [0.8644, 1.6734] (n=9) | null |
| `self_command_rate` | <= | 0.0000 [0.0000, 0.0000] (n=90) | 0.0333 [0.0000, 0.0667] (n=90) | 0.0000 [0.0000, 0.0000] (n=90) |
| `self_pause_per_hour` | <= | 290.7508 [170.2995, 415.0680] (n=90) | 0.0000 [0.0000, 0.0000] (n=90) | 290.7508 [170.2995, 415.0680] (n=90) |
| `stop_latency_p50_ms` | <= | 1504.1014 [1276.2369, 1694.4849] (n=76) | null | 1504.1014 [1276.2369, 1694.4849] (n=76) |
| `stop_latency_p95_ms` | <= | 3528.7783 [2042.3193, 5381.9202] (n=76) | null | 3528.7783 [2042.3193, 5381.9202] (n=76) |
| `transport_drop_count` | <= | null | null | null |
| `turn_retention_rate` | >= | 0.0000 [0.0000, 0.0000] (n=171) | null | 0.0000 [0.0000, 0.0000] (n=171) |
| `user_recall_nst` | >= | 0.9507 [0.9351, 0.9651] (n=832) | 0.6214 [0.5877, 0.6538] (n=832) | 0.9507 [0.9351, 0.9651] (n=832) |
| `user_word_deletion_rate` | <= | 0.4430 [0.4279, 0.4594] (n=3524) | 0.4441 [0.4288, 0.4614] (n=3524) | 0.4430 [0.4279, 0.4594] (n=3524) |

`n=` is the metric's own sample count (frames, words, clips, attempts, depending on the metric); the CI is
the 2.5 %/97.5 % pair of 1000 bootstrap resamples of those samples (`voxa/bench/metrics.py:measure`). A
`null` is a stage that cannot measure it yet, never a zero.

## What goes wrong today, per scenario

From `report.json`'s `by_scenario` block (smoke, echo mode `voxa`; the full suite's ×10 numbers in brackets
where they differ).

| scenario (clips, what it tests) | what happens today |
|---|---|
| `short` (6 clips: the user answers Voxa with one or two words over her reply) | **every word is lost**: `user_word_deletion_rate` 1.0000 (full: 0.9900 over 60 clips), `protected_word_keep_rate` 0.0000, `missed_bargein_rate` 1.0000. A short answer is the single worst case in the whole matrix — the pipeline removes 100 % of it before the recogniser sees a sound. |
| `pair` (6 clips: the SAME words said by Voxa alone, and by the user over Voxa) | `pair_accuracy` 0.3333 (1 of 3 pairs decided right on both sides; full: **0.0000**, 0 of 3), `protected_word_keep_rate` 0.0000, `user_word_deletion_rate` 0.2000 (full 0.8300). The matched pair is what stops "always stop" and "never stop" policies from passing, and today it fails: her own voice is not distinguishable from the user saying the same words. |
| `dt_onset` / `dt_first_syllable` / `dt_middle` / `dt_last_word` / `dt_echo_tail` (12 clips: the user talks over Voxa at five positions) | `missed_bargein_rate` 1.0000 for every position; `user_word_deletion_rate` 0.51 / 0.71 / 0.44 / 0.11 / 0.03 — the earlier the interruption, the more is lost; `first_word_trunc_bargein` is `null` in `voxa` mode because no interruption is ever accepted, and 1.0000 in `off` mode (full 0.9489) — when an interruption IS accepted, the first word is gone anyway. |
| `cold` (3 clips: the user starts within 200 ms of the end of Voxa's reply) | `first_word_trunc_cold` 1.0000 (full 1.0000): the first word is truncated on every cold start; `user_word_deletion_rate` 0.4000 (full 0.2444). |
| `fst` (6 clips: Voxa alone, including words that must never become a command) | `self_command_rate` 0.0000 in smoke but **0.0333 in the full suite** (3 of 90 Voxa-only clips created a user turn), `interrupt_stop_rate` 0.0000 in `voxa` mode vs 0.2222 in `off` mode (2 of 9 replies stopped by her own voice), `echo_leak_word_rate` 0.1580. |
| `nst` (6 clips: the user speaks with NOTHING playing) | `user_recall_nst` 0.6091 (full 0.6214): **39 % of the user's words are lost when there is no echo to cancel at all** — the mask and the saturating canceller are eating her, which is the design rule "never discard user audio" being broken today. |
| `quiet` (3 clips: the user is quieter than the echo) | `user_word_deletion_rate` 0.4889 (full 0.7389), `echo_leak_word_rate` 0.2381: quiet speech is the second worst case, and it gets worse with more samples. |
| `pause` (3 clips: the user pauses mid-sentence) | `premature_cutoff_rate` 0.3333 (full 0.0909), `ep50_ms` 1111 ms: a mid-sentence pause ends her turn a third of the time. |
| `path_change` (3 clips: the echo path/delay changes mid-clip) | `missed_bargein_rate` 1.0000, `user_word_deletion_rate` 0.5000, and `reconverge_s` `null` in smoke (`run.reconverge_unrecovered` = 11 of 11 delay re-estimates never bring ERLE back to 15 dB inside the clip) but 1.2689 s [0.8644, 1.6734] over 9 recovered changes in the full suite. |

The three headline numbers for the programme's goal ("hears EVERY word, NOTHING of her own voice, NEVER
interrupts herself"): today she hears **61 %** of the user's words with nothing playing and **0 %** of a
short answer over her reply, leaks **16.5 %** of her own words into what she hands on, and never interrupts
herself in the default mode only because the gate never opens (in `off` mode she stops herself 3 times in
22.3 s of her own playback).

## The ratchet: `voxa/bench/gates.toml`

Rule (roadmap B5: "set each gate to the stricter of the proposed target and *baseline minus nothing*, so
no later task can regress any metric"): **threshold = the baseline's own 95 % CI bound on the bad side**,
taken from the **smoke** suite in the runner's **default** echo mode (`voxa`). A later task passes while
its value stays inside the spread the baseline already had; a value outside it is a regression, not noise.
Every threshold line in `gates.toml` carries the long-term target as a comment, so the ratchet can never
hide how far the metric still has to go.

| metric | baseline (smoke, echo voxa) | ratchet threshold | long-term target |
|---|---|---|---|
| `align_err_p90_ms` | null | no threshold | <= 2 ms (task 1, once the TTS reference tap exists) |
| `backchannel_resume_rate` | 1.0000 [1.0000, 1.0000] (n=2) | >= 1 | >= 0.95 (task 8: >= 95 % resume on backchannels) |
| `cpu_core_share` | 0.7310 [0.6840, 0.7752] (n=48) | <= 0.775161 | <= 0.1 core (task 2: <= 10 % of one core); task 12 allows <= 0.25 for an experiment |
| `delay_est_err_ms` | 5.8011 [0.0114, 17.3580] (n=11) | <= 17.357955 | <= 1 ms std per take (B4 real-corpus gate; task 3 re-estimates every utterance) |
| `drift_erle_loss_db` | 0.0800 [0.0160, 0.1536] (n=264) | <= 0.153642 | <= 3 dB (task 3: ERLE at +/-100 ppm within 3 dB of no-drift) |
| `dt_wer_penalty_pts` | 125.2365 [27.3710, 352.4475] (n=33) | <= 352.447512 | <= 3 points (task 4 DT WER penalty gate) |
| `duck_latency_p50_ms` | null | no threshold | <= 300 ms (task 8); handoff 8.4 median <= 200 ms |
| `duck_latency_p90_ms` | null | no threshold | <= 600 ms (task 8) |
| `echo_leak_word_rate` | 0.1652 [0.1174, 0.2130] (n=230) | <= 0.213043 | <= 0.001 (task 4: echo-leak words <= 0.1 %), halved again by task 7 |
| `ep50_ms` | 913.8869 [855.5428, 1021.4147] (n=28) | <= 1021.414664 | <= 700 ms (task 9) |
| `ep90_ms` | 1159.4533 [1081.4575, 1318.9205] (n=28) | <= 1318.920546 | <= 1200 ms (task 9) |
| `erle_fst_median_db` | 0.0000 [0.0000, 0.0000] (n=372) | >= 0 | >= 25 dB converged (task 2); task 3 holds it under +/-100 ppm drift, task 10 under a device switch |
| `erle_fst_p10_db` | -0.0380 [-0.0800, -0.0100] (n=372) | >= -0.08 | >= 15 dB (task 2 P10 gate) |
| `erle_music_loss_db` | null | no threshold | ERLE with concurrent music within 1 dB of ERLE without it (task 1) |
| `false_bargein_per_hour` | 0.0000 [0.0000, 0.0000] (n=42) | <= 0 | 0 on smoke, <= 0.5/h on full (task 5); handoff 8.4: zero false playback stops in 10,000 trials |
| `first_word_trunc_bargein` | null | no threshold | <= 0.03 (task 6: <= 3 % barge-in) |
| `first_word_trunc_cold` | 1.0000 [1.0000, 1.0000] (n=3) | <= 1 | <= 0.01 (task 6: first-word truncation <= 1 % cold) |
| `fst_transcript_rate` | 0.0000 [0.0000, 0.0000] (n=9) | <= 0 | <= 0.01 (task 4: FST transcript rate <= 1 %) |
| `interrupt_stop_rate` | 0.0000 [0.0000, 0.0000] (n=9) | <= 0 | 0 (handoff 8.4: zero false playback stops in 10,000 Voxa-only trials) |
| `missed_bargein_rate` | 1.0000 [1.0000, 1.0000] (n=27) | <= 1 | <= 0.02 at SER >= -5 dB (task 5); handoff 8.4: >= 99 % interruption recall |
| `nearend_atten_dt_db` | 8.6370 [5.9698, 11.7765] (n=31) | <= 11.77651 | <= 3 dB (task 2 near-end attenuation gate) |
| `pair_accuracy` | 0.3333 [0.0000, 1.0000] (n=3) | >= 0 | 1.0 (handoff 8.4: keep-and-yield matched pairs both correct) |
| `premature_cutoff_rate` | 0.0714 [0.0000, 0.1786] (n=28) | <= 0.178571 | <= 0.01 (task 9; handoff 8.4: below 1 % is reasonable to test) |
| `protected_word_keep_rate` | 0.2273 [0.1136, 0.3636] (n=44) | >= 0.113636 | 1.0 (task 7: protected words 100 % kept) |
| `reconverge_s` | null | no threshold | <= 2 s after a 50 ms delay step (task 3) and after a device switch (task 10) |
| `self_command_rate` | 0.0000 [0.0000, 0.0000] (n=9) | <= 0 | 0 (handoff 8.4: zero self-generated commands in 10,000 trials) |
| `self_pause_per_hour` | 0.0000 [0.0000, 0.0000] (n=9) | <= 0 | 0 (handoff 8.4: zero events in 100 hours, a ~0.03/h one-sided 95 % Poisson bound) |
| `stop_latency_p50_ms` | null | no threshold | <= 200 ms (handoff 8.4 initial stop-latency target) |
| `stop_latency_p95_ms` | null | no threshold | <= 350 ms (handoff 8.4 p95 stop-latency target) |
| `transport_drop_count` | null | no threshold | 0 (handoff 8.4: zero transport loss or duplication in accepted user spans) |
| `turn_retention_rate` | null | no threshold | 1.0 with zero transport loss/duplication (handoff 8.4: promoted exactly once) |
| `user_recall_nst` | 0.6091 [0.5091, 0.6909] (n=110) | >= 0.509091 | >= 0.99 (handoff 8.4: >= 99 % user-onset recall in the normal-use range) |
| `user_word_deletion_rate` | 0.3161 [0.2720, 0.3601] (n=386) | <= 0.360104 | 0 - the goal is hears EVERY word; task 7 holds the DT recall drop <= 0.5 point |

Metrics with **no threshold**, and why (each is stated in the file):

* permanently unmeasurable at this stage — `align_err_p90_ms` (the live reference-to-echo probe is task 1),
  `erle_music_loss_db` (the frozen B1 clip set has no music playback), `transport_drop_count` (the offline
  runner has no queue; the live appsinks are task A4);
* null **in the default mode only**, because that mode produces no samples — `duck_latency_p50_ms`,
  `duck_latency_p90_ms`, `stop_latency_p50_ms`, `stop_latency_p95_ms`, `first_word_trunc_bargein`,
  `turn_retention_rate` (0 barge-ins, 0 cessations, 0 accepted spans) and `reconverge_s` (11 unrecovered
  changes in smoke). Their `off`-mode baseline values are recorded in `baseline.json` and quoted in the
  comments, so the target is visible even while the ratchet line is absent.

Two deliberate choices worth stating:

* **`cpu_core_share` is baselined on the WORST of three identical runs**, not the luckiest one: its
  threshold 0.775161 is the CI high of the entry `baseline.json` keeps (0.7310 [0.6840, 0.7752]), while the
  other two runs measured 0.7270 [0.6812, 0.7716] and 0.7300 [0.6855, 0.7725]. It is the only metric whose
  value moves between identical runs: it divides ASR seconds by wall-clock seconds on a shared machine, and
  the bootstrap CI covers clip-resampling variance, not machine-load variance. A later task must beat the
  worst baseline run, and `tests/test_bench_baseline.py` fails if any threshold in `gates.toml` is ever made
  looser than the baseline's own CI bound (tightening one is the move the task that improves the metric makes).
* **The ratchet is the smoke ratchet, so a full-suite run can legitimately fail it.** The full suite found
  `self_command_rate` 0.0333 and `fst_transcript_rate` 0.0333 where smoke found 0.0000 (3 of 90 Voxa-only
  clips, not 0 of 9), and `pair_accuracy` 0.0000 where smoke found 0.3333. Those are not regressions, they
  are the same baseline seen with more clips: `python -m voxa.bench baseline check REPORT` compares a run with
  **its own suite and mode** in `baseline.json`, and that is the command to use for a full-suite run.
  `self_command_rate` being non-zero on the full suite is itself a finding for task 11 (speaker anti-enrolment)
  and task A1.

## Proof the ratchet bites

* `python -m voxa.bench run --suite smoke --seed 0 --out .voxa-bench/report.json` in the default mode
  **passes**: 23 thresholds, every one `ok`, exit 0.
* One deliberate break of the canceller — `voxa/anc.py`, the double-talk guard
  `if self._speaking_left > 0 or reference_quiet:` changed to `if reference_quiet:` (the filter adapts on
  the user's voice instead of protecting it) — makes the same command **exit 1**:
  `gate: drift_erle_loss_db<=0.153642: drift_erle_loss_db=1.20306`, and the same report moves
  `erle_fst_p10_db` −0.038 → 0.000, `echo_leak_word_rate` 0.1652 → 0.1391 and
  `nearend_atten_dt_db` 8.6370 → 8.6380. `python -m voxa.bench baseline check` on that report also exits 1
  with `drift_erle_loss_db: REGRESSION 1.2031 beyond baseline CI high 0.1536`. The sabotage was reverted
  (`git checkout -- voxa/anc.py`) and the clean run passes again.
* Honest reading of that result: the break moved **one** threshold. The ratchet is tight where the CI is
  tight (`drift_erle_loss_db`, `ep50_ms`, `echo_leak_word_rate`, `user_word_deletion_rate`,
  `protected_word_keep_rate`, `nearend_atten_dt_db`), and weak where the CI is wide on a handful of samples
  (`dt_wer_penalty_pts` bound 352.4 from 33 samples, `pair_accuracy` bound 0.0 from 3 pairs,
  `backchannel_resume_rate` bound 1.0 from 2 clips). Those lines can only tighten when a task adds samples
  or the full suite does; they are recorded here so nobody mistakes them for strength.

## Limits of this baseline (read before quoting a number)

1. **Synthetic only.** Every clip is B1's rendered set: Voxa's own Edge-TTS voice (`en-US-AriaNeural`) as
   the reference, `en-US-GuyNeural` plus LibriSpeech as the user, synthetic RIRs, injected delay/drift.
   Nothing here is rhY's room, her microphone or her speaker — that is roadmap **B4**, see the TODO below.
2. **The `webrtc` column is not a WebRTC result** (see the mode table). A real AEC3/WebRTC baseline needs
   task 2.
3. **CIs are clip-resampling CIs**, not run-to-run CIs, except where stated (`cpu_core_share`).
4. **Small-sample metrics** (`pair_accuracy` n=3, `backchannel_resume_rate` n=2, `first_word_trunc_cold`
   n=3 in smoke) have CIs that span the whole range; the full suite's ×10 is the honest tightening.
5. **Three metrics are null for structural reasons**, listed above; a later task that makes one measurable
   must say which stage made it measurable.
6. `fst_transcript_rate` 0.0000 is a **floor, not a success**: the metric counts FST clips whose cleaned
   stream yields a transcript with words to check, and in smoke none do — while `echo_leak_word_rate` says
   16.5 % of the words in that stream are her own. Task 4's target (≤ 1 %) is measured against the
   *leak* metric, not this one.
7. **The `voxa` mode's zero barge-ins have a bench cause, not only a product one** (found by f138, left
   unfixed on purpose): `voxa/bench/runner.py:_clean_frame` clips `ReferenceMask.process`'s output to -1..1
   and then scales it by 32767, while that stage returns samples in the scale it was given. Every non-silent
   frame of echo mode `voxa` therefore lands at ±32767, the barge-in level is pinned at 1.0 for her voice
   AND for the user's, and the gate can never open — hence erle 0 dB, `suspects` 0 and the NULL
   `turn_retention_rate` / `first_word_trunc_bargein` / `duck_latency_*` / `stop_latency_*` columns above.
   Re-scaling it would move every number in the `voxa` column and invalidate the ratchet in
   `voxa/bench/gates.toml`, so it belongs with the echo tasks that re-record the baseline (tasks 1-4),
   not with a retention task. Milestone A's retention numbers are measured in echo mode `off`, where the
   barge-in path runs and this file has samples for them (n=17 smoke, n=171 full).

## TODO — roadmap B4: the real-room corpus (NOT this task)

B4 records rhY's actual room in the same bench format and is the only part of B1-B5 that needs sound and a
microphone, so it is rhY's to run, not Strata's (header rule 3: never play sound or record microphones here).

* [ ] `record-bench` in `tools/anc_lab.py`: TTS-only takes at the existing VOLUMES (0.2/0.4/0.7/1.0),
      ~100 labelled barge-ins read over playing TTS, and a TV/music-interference take, mic **and** the exact
      reference sent to `paplay` captured at 48 kHz.
* [ ] Gate for B4: ≥ 30 min of TTS-only audio, ≥ 100 labelled barge-in takes, and the cross-correlation
      delay within one take having std ≤ 1 ms (the corpus proves its own labels).
* [ ] Then re-run this page's tables with `--data-root` pointed at the real corpus and add a `real` column
      beside `off`/`voxa`/`webrtc`; `baseline.json` grows a `corpus` key rather than replacing the synthetic
      numbers, so the synthetic ratchet stays valid on any machine.
* [ ] Until B4 lands, every number on this page is a synthetic-corpus number and should be quoted that way.
