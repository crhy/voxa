# DTLN-aec ONNX weights shipped with Voxa (task h04, issue #72)

The two stage models Voxa's live echo stage `dtln` runs with `onnxruntime`. Same model, same
licence as the DTLN-aec `.tflite` weights h03 measured on the bench (docs/hearing/frontier.md,
`/home/rhy/voxa-bench-data/third_party/SOURCES.md`): DTLN-aec 512 (MIT, breizhn/DTLN-aec, commit
9d24e12). They are an ONNX export of that published model, not a retrain, so every h03 DTLN number
carries over.

* stage 1 (`dtln_aec_512_1.onnx`): the mask LSTM. inputs `input_3` (mic magnitude [1,1,257]),
  `input_5` (LSTM state [1,2,512,2]), `input_4` (far-end magnitude [1,1,257]); outputs `Identity`
  (mask [1,1,257]) and `Identity_1` (new state).
* stage 2 (`dtln_aec_512_2.onnx`): the time-domain enhancement LSTM. inputs `input_6` (masked frame
  [1,1,512]), `input_8` (LSTM state [1,2,512,2]), `input_7` (far-end time frame [1,1,512]); outputs
  `Identity` (enhanced frame [1,1,512]) and `Identity_1` (new state).

The input/output names and shapes are byte-for-byte the same as the `.tflite` interpreter's
(`get_input_details`/`get_output_details`), so the streaming contract is identical to the repo's own
`run_aec.py` (16 kHz, block 512 / shift 128, stage 1 + stage 2).

Provenance of the ONNX export: the MIT DTLN-aec model exported to ONNX (opset 17) by the `unbleed`
project (https://github.com/AndBobsYourUncle/unbleed, `models/dtln_aec_512_{1,2}.onnx`, whose
`licenses/DTLN-aec-MIT.txt` is the same MIT text as `LICENSE.DTLN-aec.txt` here).

Verification that these ONNX files are the same weights h03 measured (`onnxruntime` 1.30, CPU),
against the DTLN-aec repo's own `*_processed.wav` (its `run_aec.py` output): per-stage single-frame
outputs match the `.tflite` interpreter (mask max |diff| 2.4e-7, state 6e-7), and the whole-file
`run_aec.py` reconstruction through ONNX reproduces `*_processed.wav` (double-talk corr 1.00000,
max |diff| 4.8e-4; far-end single-talk corr 0.998; near-end single-talk corr 0.99999, the residual
there is run_aec's own peak-normalisation clip, not the model).

SHA-256:

    30c8bb1b26bc0be5ef75de06b30395d51611448678daab7a420f26501acff50b  dtln_aec_512_1.onnx
    92c719eba0482f3e3e3b2cbe8c1f46fd6d8f4e2b169c787f838c259b5ab9b5fa  dtln_aec_512_2.onnx
    aa95acd8c8a7341bfcdb2823694dcbb27a2a4143e860bb52db1ff0f29c85e5a1  LICENSE.DTLN-aec.txt  (matches the DTLN-aec LICENSE in SOURCES.md)
