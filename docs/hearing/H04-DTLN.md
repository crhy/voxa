# Hearing 2 - h04: DTLN-aec shipped as Voxa's echo stage (issue #72)

H03 measured DTLN-aec as the strongest echo canceller on Voxa's bench (docs/hearing/frontier.md):
ERLE 24.41 dB median for 7.86 dB of user attenuation, beating the incumbent `voxa` (6.92 dB / 9.37 dB)
and `aec3` (31.43 dB / 24.96 dB) on the echo-removal-per-unit-of-user-damage trade. h04 makes it the
**shipping live echo stage** - in the app and in the Flatpak - not just a measured arm.

Offline throughout: the microphone is never opened and no sound is played.

## What changed

1. **One stage, app and bench share it.** `voxa/dtln.py` is a streaming `DtlnCanceller` behind the same
   canceller contract the live capture and every bench arm use (`process(mic, ref)` on float -1..1
   samples, `continuous`, `delay_managed_internally`, `close()`). The bench `--echo-mode dtln` arm is now
   `voxa/dtln.py`'s class (`voxa/bench/modes.py` imports it), so the frontier number *is* the app number.
   The streaming reconstruction mirrors the DTLN-aec repo's own `run_aec.py` (16 kHz, block 512 / shift
   128, stage 1 = frequency mask, stage 2 = time-domain enhancement), and the LSTM state is kept across
   `process` calls, so it runs frame by frame on a live stream and never rebuilds. A test proves the
   output is chunk-size invariant (one chunk at a time == fed in 320-sample pieces).
2. **`dtln` is the default `echo_mode`** (`voxa/config.py`, ECHO_MODES and `Settings.echo_mode`). The old
   stage is kept as the fallback: when onnxruntime or the weights are missing the app logs once and uses
   Voxa's own NLMS canceller (`voxa/audio.py` / `voxa/window.py`), exactly as `aec3` already falls back.
   The picker lists the neural stage first and only when it can actually run here.
3. **Weights shipped.** `voxa/data/dtln/dtln_aec_512_{1,2}.onnx` (MIT, see that folder's `SOURCES.md` and
   `LICENSE.DTLN-aec.txt`), wired into the wheel via `[tool.setuptools.package-data]` and into the
   Flatpak (the manifest already `cp -r voxa`, plus a build-time `test -f` guard so packaging cannot drop
   them); `onnxruntime` added to `requirements.txt` and as a first-class Flatpak python dep
   (`python3-requirements-flatpak.json`).
4. **Far-end reference = the exact samples Voxa plays.** `dtln` runs against Voxa's own render timeline
   (`use_own_reference`), like `voxa`/`aec3`: the neural stage models the echo delay itself
   (`delay_managed_internally`), so Voxa hands it the raw aligned reference and pushes no delay into it.
   The test voice VoxaTest plays is excluded by construction.

## Weight provenance and verification

The ONNX files are an export of the *same* MIT model h03 measured (breizhn/DTLN-aec commit 9d24e12,
DTLN-aec 512), not a retrain. Their input/output names and shapes are byte-for-byte the `.tflite`
interpreter's. Verified against the DTLN-aec repo's own `*_processed.wav` (its `run_aec.py` output):
per-stage single-frame outputs match the `.tflite` interpreter (mask max |diff| 2.4e-7, state 6e-7),
and the whole-file `run_aec.py` reconstruction through ONNX reproduces `*_processed.wav` (double-talk
corr 1.00000, max |diff| 4.8e-4). So every h03 DTLN number carries over.

## Latency (step 4)

Per-block (stage 1 + stage 2) processing time, measured on this 16-core CPU with `onnxruntime`
single-threaded (the live setting): **p50 0.55 ms, p99 1.01 ms** idle; under full-machine load
p50 1.09 ms, p99 2.00 ms, max 2.12 ms. The stream advances 8.00 ms per block (128 samples at 16 kHz),
so the stage finishes a block in ~1/8 of the time it covers even under load and **never falls behind**.
A test (`tests/test_dtln_stage.py::test_per_block_latency_stays_under_the_frame_shift`) asserts p99 <
the frame shift. Cost is the last two frontier columns: the full suite reports the echo stage's own
ms-per-audio-second and core share below.

## Fallback behaviour

`dtln_available()` is `(ok, reason)` and never raises. The live paths (`voxa/audio.py:set_echo_mode`,
`voxa/window.py`) and the picker (`window.py:_echo_mode_choices`) consult it: if onnxruntime cannot be
imported or a weight file is missing under `voxa/data/dtln`, the stage is reported unavailable, the
capture falls back to Voxa's own NLMS canceller (`voxa`), and a single log line says which. The neural
stage never crashes the capture; a runtime error inside a cleaned chunk is caught by the capture path
and drops to the raw microphone exactly as every other canceller.

## Numbers - see docs/hearing/frontier.md, section "h04 shipped"

The full bench and smoke suites were re-run with `--echo-mode dtln` driving the *live* stage (the same
code path the app uses), on seed 0, offline. The live-stage numbers land on h03's measured DTLN row to
within the bootstrap interval, confirming the ONNX port reproduces the LiteRT arm.
