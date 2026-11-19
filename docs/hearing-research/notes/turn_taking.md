# Turn-taking, interruption handling and endpointing for Voxa (CPU-only Python, single mic)

Scope: methods and open tools for deciding when the user is talking to the assistant, when they have finished, and whether an overlap is a real interruption. Research date 2026-11-17. Echo cancellation (AEC) is only touched on where it affects barge-in logic.

## 1. Predictive turn-taking models (VAP, TurnGPT) and what runs on CPU

### Takeaway
Voice Activity Projection (VAP) is the main open, research-grade predictive turn-taking model. It is MIT-licensed, works at 50 Hz, and has been shown to run in real time on CPU. It is built for two-channel (user + system) audio, which suits Voxa because Voxa knows exactly when its own TTS is playing. TurnGPT is the text-only ancestor. It is useful as a concept, but in production it has been superseded by small semantic end-of-turn classifiers (section 2).

### Cited Findings
- VAP is a self-supervised objective that predicts future voice activity of both speakers in a window, from which turn-taking events (shift/hold, backchannel) are derived. Papers: Ekstedt & Skantze, Interspeech 2022 (arXiv:2205.09812), and "How Much Does Prosody Help Turn-taking?" SIGDIAL 2022 (arXiv:2209.05161, SIGDIAL 2022 Best Paper Award). — [ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection)
- The repo's stereo model takes a 2-channel waveform (one channel per speaker) and needs no separate VAD at inference. Mono input is accepted (a silent second channel is added). Architecture: a GPT-like transformer with ALiBi attention on top of a pretrained CPC speech encoder, trained with a VAP + per-frame VAD multitask loss. Frame rate 50 Hz. License MIT. Only one pretrained checkpoint is shipped (`VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`). Training code depends on a private dataset repo and on Switchboard/Fisher/Candor. — [GitHub](https://github.com/ErikEkstedt/VoiceActivityProjection)
- Inoue et al. (IWSDS 2024, arXiv:2401.04868) show a VAP system mapping stereo dialogue audio to future voice activity that "can run in real time on CPU", and they study how input context length affects performance and transformer processing time. — [arXiv 2401.04868](https://arxiv.org/abs/2401.04868v1)
- Multilingual VAP (LREC-COLING 2024): one model trained on English, Mandarin and Japanese performs on par with the monolingual models. — [arXiv 2403.06487](https://arxiv.org/html/2403.06487v3)
- A Dec 2025 follow-up extends VAP to continuous, multilingual backchannel prediction (Japanese, English, Chinese) and reports integration into real-time software with CPU-only inference. — [arXiv 2512.14085](https://arxiv.org/html/2512.14085)
- TurnGPT (Ekstedt & Skantze, Findings of EMNLP 2020, arXiv:2010.10874) is a transformer LM that predicts turn shifts from dialogue text. It beat two prior baselines, and its analyses show it uses dialogue context and pragmatic completeness. The authors also explored projecting (anticipating) turn completions. — [arXiv 2010.10874](https://arxiv.org/abs/2010.10874)

### Inferences
- VAP's two-channel design maps directly onto Voxa. Channel A is the mic (after AEC) and channel B is Voxa's own TTS signal, which Voxa already has digitally. The model can then judge "user starting a turn while system speaks" versus a backchannel, and it naturally knows when Voxa is speaking.
- VAP's public checkpoint is trained on English telephone corpora (Switchboard/Fisher) at 8 kHz-style conditions. Expect a domain gap to a desktop mic. Treat it as an experiment, not a drop-in. Smart Turn (section 2) is the lower-risk first step.
- TurnGPT's idea survives in today's text-based detectors (LiveKit's earlier Qwen2.5-0.5B detector, and the option of asking the main LLM "is the user finished?").

### Gaps
- Parameter count, exact CPU latency per 20 ms frame, and accuracy numbers for the public VAP checkpoint were not stated in the repo. I did not fetch the full IWSDS 2024 paper's timing table.
- TurnGPT's numerical results (balanced accuracy etc.) were not retrieved.
- I did not verify whether any VAP checkpoint is trained on wideband (16 kHz) desktop audio.

## 2. Semantic / smart endpointing open models (Smart Turn, LiveKit turn detector) and combining them with VAD timeouts

### Takeaway
Pipecat's Smart Turn v3.x is the best fit for Voxa: BSD-2-Clause, 8 MB int8 ONNX, Whisper-Tiny encoder plus a linear head, 23 languages, about 10-60 ms per call on CPU, and run only after Silero VAD reports a short silence. Its accuracy on Daily's own test set is 94.7% for English (v3.1, 8 MB). LiveKit's local `v1-mini` audio detector is free to use, but it is under the LiveKit Model License, needs 140-250 MB per instance, and has no published numbers. The standard pattern is: short VAD silence (about 0.2-0.3 s), then the model, then a hard maximum timeout (about 2.5-3 s) as a fallback.

### Cited Findings
- Smart Turn v3: Whisper Tiny base with a linear classifier layer, about 8M parameters, 8 MB int8 CPU version and 32 MB fp32 GPU version, license BSD 2-clause, 23 languages. It is designed to run only after a lightweight VAD (for example Silero) detects silence. It reads the whole user turn audio (truncated to about 8 s, 16 kHz mono, zero-padded at the start). If the user resumes speaking before inference finishes, it reruns on the whole turn. Reported CPU latency is "as little as 10ms on some CPUs, and under 100ms on most cloud instances". Training and test data are open on HuggingFace (`pipecat-ai/smart-turn-data-v3.2-train/-test`). — [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn)
- The Smart Turn v3 launch quotes about 12 ms CPU inference and roughly 100x faster than v2 on an AWS c8g.medium. — [Daily blog: Smart Turn v3](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)
- Smart Turn v3.1 accuracy on `smart-turn-data-v3.1-test`: English 88.3% (v3.0), 94.7% (v3.1 8 MB) and 95.6% (v3.1 32 MB). Spanish 86.7%, 90.1% and 91.0%. The other 21 languages are about the same as v3.0. CPU latency for the 8 MB model is 9 ms (c7a.2xlarge), 20 ms (c8g.2xlarge), 37 ms (c7a.medium) and 57 ms (c8g.medium), plus 7-9 ms preprocessing. The blog recommends `OMP_NUM_THREADS=1` and `OMP_WAIT_POLICY="PASSIVE"` for consistent ONNX runtime performance. — [Daily blog: v3.1](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/)
- An independent comparison (soniqo.audio) measured Smart Turn v3 at 88.2% recall, a 24.3% false-positive rate and a 1,031 ms median decision latency on 5 conversations. The method differs from Daily's, so the numbers are not directly comparable. — [soniqo.audio benchmarks](https://soniqo.audio/blog/voice-agent-benchmarks) (via search summary)
- Pipecat defaults: the stop strategy is `TurnAnalyzerUserTurnStopStrategy` with `LocalSmartTurnAnalyzerV3`. The VAD `stop_secs` is 0.2 s, and the docs say to leave it alone and tune the stop strategy instead. `SpeechTimeoutUserTurnStopStrategy.user_speech_timeout` is 0.6 s, so the shortest wait after a pause is 0.2 + 0.6 = 0.8 s. `user_turn_stop_timeout` is 5.0 s. `wait_for_transcript=True` means the turn closes only once both the analyzer and the transcript are ready. — [Pipecat user turn strategies](https://docs.pipecat.ai/server/utilities/turn-management/user-turn-strategies)
- LiveKit audio turn detector: `v1` runs on LiveKit Inference (cloud). `v1-mini` "runs locally on CPU, free to use in any context", uses about 140-250 MB per loaded copy, covers 14 languages, and is under the LiveKit Model License. No accuracy or latency numbers are published; LiveKit points to its eot-bench harness instead. It requires VAD with `min_silence_duration` of at least 0.25 s. Endpointing defaults are `min_delay` 0.5 s and `max_delay` 3.0 s without the audio detector, and 0.3 s and 2.5 s with it. — [LiveKit turn detector docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- LiveKit's older text turn detector (deprecated, to be removed in SDK 2.0) is Qwen2.5-0.5B-Instruct, 396 MB, about 50-160 ms on CPU, under 500 MB RAM. True-positive / true-negative rates: English 99.3% / 87.0%, Hindi 99.4% / 96.3%, Korean 99.3% / 94.5%, Italian 99.3% / 85.1%. Its `mode="dynamic"` adapts the delay within min/max using pause statistics. — [LiveKit turn detector docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- LiveKit endpointing parameters: `endpointing.min_delay` 0.5 s, `endpointing.max_delay` 3.0 s, `endpointing.mode` "fixed" or "dynamic" (dynamic adapts to the user's pause patterns). Preemptive LLM generation is on by default and starts when a final transcript arrives, before the turn is confirmed. `preemptive_tts` is off by default. — [LiveKit tuning docs](https://docs.livekit.io/agents/logic/turns/tuning/)
- OpenAI Realtime `server_vad` defaults: `threshold` 0.5, `prefix_padding_ms` 300, `silence_duration_ms` 500. `semantic_vad` has an `eagerness` setting (low / medium / high / auto, default auto), where "low will wait longer for the user to continue speaking, high will respond more quickly". — [OpenAI Realtime VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad)

### Inferences
- A recommended Voxa endpointing stack, all CPU and permissively licensed:
  1. Silero VAD at 16 kHz in 32 ms frames, with about 0.2-0.3 s of silence counted as a candidate pause.
  2. Smart Turn v3.x (8 MB ONNX, single thread) runs on the whole turn audio.
  3. If the probability of "complete" is high, end the turn now. If it is low, keep listening and re-run Smart Turn on each new pause. Force the end at a hard cap of about 2.5-3 s of silence (LiveKit's max), or longer for dictation.
  4. Optionally, a text check on the faster-whisper partial transcript for things like a trailing "and", "but", "um", or a number being dictated.
- Smart Turn's 20-60 ms on a small CPU is negligible next to faster-whisper. Run it on its own thread with `OMP_NUM_THREADS=1` so it does not compete with ASR.
- Independent numbers (24% false positives) suggest the model alone will still cut people off sometimes. A guard of "minimum 0.5 s silence plus model says done" is safer than the model alone.

### Gaps
- No public accuracy numbers for Smart Turn v3.2 or for LiveKit `v1-mini`. Smart Turn v2's architecture (wav2vec2-based per earlier experiments) and numbers were not retrieved in detail.
- No head-to-head benchmark of Smart Turn vs LiveKit on the same data was found apart from the small soniqo one.

## 3. Telling backchannels and noise (cough, laugh, TV) apart from real interruptions; how frameworks implement barge-in

### Takeaway
Production systems use three layers: (a) a minimum speech duration (LiveKit 0.5 s; Vapi `voiceSeconds` about 0.2 s), (b) a minimum word count from STT (Pipecat `MinWords...` such as 3; Vapi `numWords` 2-3; LiveKit `min_words` default 0), and (c) an acknowledgement-phrase or learned backchannel classifier. They pair these with "false-interruption recovery": pause TTS, and if no real words follow within about 2 s, resume. LiveKit's learned interruption model is the most advanced, but it is cloud-only.

### Cited Findings
- LiveKit interruption parameters (Python defaults): `interruption.enabled` True, `interruption.min_duration` 0.5 s, `interruption.min_words` 0 (requires STT), `interruption.false_interruption_timeout` 2.0 s (the silence window after an interruption before it is classed as false), `interruption.resume_false_interruption` True (interrupted agent speech resumes), and `interruption.mode` "adaptive" if available, otherwise "vad" (which triggers on any detected speech). — [LiveKit tuning docs](https://docs.livekit.io/agents/logic/turns/tuning/)
- LiveKit adaptive interruption model: an audio encoder plus a CNN looking at the first few hundred ms of user speech, using onset sharpness, duration and prosody, trained on hundreds of hours of human conversation with noise mixed in. Results: 86% precision and 100% recall at 500 ms of overlap speech. It rejects 51% of VAD-triggered barge-ins, detects true barge-ins faster than VAD in 64% of cases, runs inference in 30 ms or less, and needs a median of 216 ms of audio to trigger. Weights are not open; it runs on LiveKit Cloud. — [LiveKit blog](https://livekit.com/blog/adaptive-interruption-handling)
- A LiveKit community thread asking for open weights got an unofficial reply suggesting users layer their own classifier on top of VAD mode. No official open-weights roadmap was found. — [LiveKit community](https://community.livekit.io/t/roadmap-for-adaptive-interruption-handling-in-self-hosted-livekit-workers/1172) (via search summary)
- Pipecat: `MinWordsInterruptionStrategy(min_words=3)` is the older API. User speech that does not meet the interruption criteria is discarded, not queued. Interruption strategies only apply while the bot is speaking. — [Pipecat interruption strategies](https://docs.pipecat.ai/server/utilities/interruption-strategies) (via search summary)
- Pipecat 1.x: `MinWordsUserTurnStartStrategy(min_words, use_interim=True)` triggers after 1 word when the bot is silent, and applies `min_words` only while the bot is speaking. The default start strategies are VAD plus transcription. There is also a `WakePhraseUserTurnStartStrategy(phrases, timeout=10.0)` and a Krisp `KrispVivaIPUserTurnStartStrategy(threshold=0.5)`. — [Pipecat user turn strategies](https://docs.pipecat.ai/server/utilities/turn-management/user-turn-strategies)
- Vapi `stopSpeakingPlan` fields:
  - `numWords`: 0 means stop on any sound. 3 means do not stop unless at least 3 words are transcribed. 2-3 is suggested so users can make brief clarifications.
  - `voiceSeconds`: about 0.2 in examples. Higher values reduce false triggers but slow detection.
  - `backoffSeconds`: default 1 s, the wait before resuming after an interruption.
  - `acknowledgementPhrases`: backchannels such as "yeah" that are ignored.
  - `interruptionPhrases`: phrases that clear the pipeline instantly.
  Some detail comes from draft documentation PRs and community posts. — [Vapi voice pipeline configuration](https://docs.vapi.ai/customization/voice-pipeline-configuration); [Ruby SDK StopSpeakingPlan](https://www.rubydoc.info/gems/vapi_server_sdk/Vapi/StopSpeakingPlan)
- Retell exposes `interruption_sensitivity` from 0 to 1 (0.8 is recommended for noisy lines; lower it to be more robust to background speech). Retell's docs say "agent stops talking for no reason" is likely background noise or speech, and recommend denoising first, then lowering sensitivity. Community threads report short "mhm" or "yeah" sounds splitting agent sentences at high sensitivity, and agents replaying or continuing an interrupted utterance as a separate turn. — [Retell interaction configuration](https://docs.retellai.com/agent/interaction-configuration); [Retell background-noise doc](https://docs.retellai.com/build/handle-background-noise.md); [Retell community](https://community.retellai.com/t/agent-speaks-the-remainder-of-its-sentence-as-a-separate-turn-after-barge-in/3580) (via search summary)
- Full-Duplex-Bench v1.5 shows that even commercial systems trade these errors off. GPT-4o Realtime stops fastest (0.18-0.23 s) but responds to side conversation and background speech in over 90% of trials. Amazon Nova Sonic ignores 98% of background speech and backchannels but takes 1.05-2.25 s to stop. — [Full-Duplex-Bench v1.5](https://arxiv.org/html/2507.23159v1)

### Inferences
- A practical Voxa barge-in rule set with no cloud and no new models:
  1. VAD onset while TTS plays starts a "candidate" state (see section 6) and does not yet count as an interruption.
  2. It becomes a confirmed interruption when the ASR partial has 2 or more non-backchannel words (or 1 word from a command list such as "stop", "wait", "no", "Voxa"), or when there is 0.5 s or more of continuous speech that the AEC residual check does not match to TTS.
  3. Backchannel list ("mm-hm", "yeah", "uh-huh", "okay", "right", "I see") means ignore it and keep speaking.
  4. Cough, laugh or TV is handled by min-duration and min-words. Noise produces no words, or produces Whisper's typical non-speech output such as "(coughs)" or "Thank you." when given noise. Filter bracketed or parenthesised tokens and low `no_speech_prob`/`avg_logprob` segments.
- LiveKit's numbers (median 216 ms of audio to classify, 86% precision) suggest a learned onset classifier can decide in about 0.2-0.3 s. A rule-based min-words gate will be slower (about 0.6-1.0 s, since it needs ASR words) but has no licensing or cloud dependency.
- TV or a second voice is the hardest case. Word-count gates do not reject them because they produce real words. Options are speaker verification against an enrolled user voice, or VAP-style models that use the user channel. Full-Duplex-Bench v1.5 shows commercial systems also fail here often.

### Gaps
- No open-weights backchannel-vs-interruption audio classifier equivalent to LiveKit's adaptive model was found, apart from research VAP backchannel work (arXiv 2512.14085).
- Exact published defaults for Vapi `acknowledgementPhrases` were not verified on a live docs page.
- OpenAI Realtime's internal barge-in logic (beyond VAD parameters and truncating audio playback) was not documented in sources reviewed.

## 4. Full-duplex spoken dialogue models: what they show about overlap, and local usability

### Takeaway
End-to-end full-duplex models (dGSLM, Moshi) are fast (about 0.26-0.35 s turn latency) but take over the floor during user pauses about 93-98% of the time and handle backchannels poorly. Systems with an explicit speak/listen state predictor (Freeze-Omni) or commercial cascaded designs interrupt much less. For Voxa the lesson is to keep a cascaded pipeline with an explicit turn-state controller. None of these models are realistic on CPU only.

### Cited Findings
- Moshi (Kyutai, arXiv 2410.00037) is a speech-text foundation model for real-time full-duplex dialogue. It has 160 ms theoretical latency and about 200 ms in practice, against a 230 ms average gap in natural conversation across 10 languages. — [Moshi paper](https://arxiv.org/html/2410.00037v1)
- MinMo (arXiv 2501.06282) reports full-duplex latency of about 600 ms in theory and about 800 ms in practice. — [MinMo](https://arxiv.org/html/2501.06282v1)
- Full-Duplex-Bench v1 (arXiv 2503.04721) results. Pause TOR and backchannel TOR count turn takeovers during user pauses and backchannels, so lower is better:

  | System | Pause TOR, synthetic | Pause TOR, CANDOR | Backchannel TOR | Turn-taking TOR | Turn-taking latency | Interruption TOR | GPT-4o score | Interruption latency |
  |---|---|---|---|---|---|---|---|---|
  | dGSLM | 0.934 | 0.935 | 0.691 | 0.975 | 0.352 s | 0.917 | 0.201 | 2.531 s |
  | Moshi | 0.985 | 0.980 | 1.000 | 0.941 | 0.265 s | 1.000 | 0.765 | 0.257 s |
  | Freeze-Omni | 0.642 | 0.481 | 0.636 | 0.336 | 0.953 s | 0.867 | 3.615 | 1.409 s |
  | Gemini Live | 0.255 | 0.310 | 0.091 | 0.655 | 1.301 s | 0.891 | 3.376 | 1.183 s |

  The authors credit Freeze-Omni's lower takeover rate to its dedicated speak/listen state-prediction module, and suggest that end-to-end systems would benefit from one. — [Full-Duplex-Bench](https://arxiv.org/html/2503.04721v3)
- Full-Duplex-Bench v1.5 (arXiv 2507.23159) tests four overlap types: user interruption, listener backchannel, talking to someone else, and background speech. It reports two outcomes: the share of trials where the system responds and the share where it resumes, plus the time to stop.

  | System | Respond to real interruption | Resume after backchannel | Respond to background speech | Time to stop on interruption |
  |---|---|---|---|---|
  | Freeze-Omni | 0.72 | 0.80 | 0.63 | 1.42 s |
  | Moshi | 0.50 | 0.06 | 0.21 | 1.16 s |
  | GPT-4o | 0.78 | 0.70 | 0.93 | 0.23 s |
  | Gemini | 0.33 | 0.93 | 0.70 | 2.20 s |
  | Nova Sonic | 0.24 | 0.98 | 0.01 | 2.25 s |

  Moshi's backchannel outcome is mostly "Unknown" (92%). The best human-like re-entry is about 1.2 s, roughly 5-6 times slower than human turn-taking at about 200 ms. — [Full-Duplex-Bench v1.5](https://arxiv.org/html/2507.23159v1)
- Freeze-Omni is described as a cascaded design that operates on raw speech input while keeping internal reliance on text. — [Full-Duplex-Bench v1.5](https://arxiv.org/html/2507.23159v1); [arXiv 2509.14515](https://arxiv.org/html/2509.14515v1)

### Inferences
- The benchmarks show the core trade-off Voxa faces. A fast stop (GPT-4o at 0.2 s) comes with many false activations from background speech, while robust rejection (Nova Sonic) comes with a 1-2 s stop latency. A "pause then confirm" design (section 6) gets around this: stop the audio in about 0.2 s, but only commit to abandoning the reply after about 0.5-1 s of confirmation.
- The Freeze-Omni result supports adding an explicit turn-state classifier (Smart Turn / VAP) rather than relying on VAD silence alone.
- Moshi at 7B parameters, and similar models, need a GPU (Kyutai reports about 200 ms on an L4 GPU per secondary summaries). They are not a fit for a CPU-only assistant that also needs a separate LLM and tool use.

### Gaps
- Moshi's license and CPU feasibility were not verified from the primary README in this session. Kyutai has released MLX and Rust ports, but I did not confirm CPU-only real-time numbers.
- SyncLLM, dGSLM code availability, and 2025-2026 models (for example FLM-Audio, arXiv 2509.02521, and TASTE2, arXiv 2609.08956) were seen in search results but not examined.

## 5. Mid-sentence pauses, adaptive silence timeouts, incremental ASR, and pre-roll so no word is lost

### Takeaway
Never cut audio on the VAD's end: keep a ring buffer and give ASR the whole turn (with pre-roll), and let the endpointer, not the VAD, decide the turn end. Use a short VAD silence plus a semantic model plus a long hard cap, ideally adapted to the user's measured pause habits. For partial transcripts on CPU, whisper_streaming's LocalAgreement-2 works with faster-whisper (MIT, about 3.3 s latency in the paper) but is now marked outdated in favour of SimulStreaming. Moonshine v2 streaming models are a much lower-latency CPU option.

### Cited Findings
- Silero VAD defaults: `threshold` 0.5, `neg_threshold` threshold minus 0.15, `min_speech_duration_ms` 250, `min_silence_duration_ms` 100, `speech_pad_ms` 30, window 512 samples at 16 kHz (32 ms). `VADIterator` has the same `threshold`, `min_silence_duration_ms` 100 and `speech_pad_ms` 30. — [silero-vad utils_vad.py](https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py)
- OpenAI Realtime pads 300 ms of audio before detected speech by default (`prefix_padding_ms`). — [OpenAI Realtime VAD](https://developers.openai.com/api/docs/guides/realtime-vad)
- Smart Turn takes the entire turn (up to about 8 s), and reruns when the user resumes speaking before inference completes. This means pauses inside a turn are handled by re-evaluation, not by cutting. — [smart-turn](https://github.com/pipecat-ai/smart-turn)
- LiveKit's `endpointing.mode="dynamic"` adapts the delay within min/max "based on pause patterns". Its text detector in VAD mode uses an effective delay of max(VAD silence, min_delay). — [LiveKit tuning](https://docs.livekit.io/agents/logic/turns/tuning/); [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/)
- whisper_streaming (ÚFAL):
  - LocalAgreement-n confirms a prefix when n consecutive updates agree (the default is 2).
  - The paper reports 3.3 s latency on unsegmented long-form speech.
  - Options include `--min-chunk-size` (the example uses 1 s), buffer trimming by `segment` (default and better) or `sentence`, and `--vac` (Silero-based voice activity controller, recommended).
  - faster-whisper is the recommended backend. License MIT.
  - The README says: "In 2025, WhisperStreaming is becoming outdated, replaced by SimulStreaming" (faster and higher quality, but only a torch backend).
  — [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming)
- Moonshine v2 (arXiv 2602.12241, Feb 2026) uses sliding-window attention in a streaming encoder to avoid full-attention TTFT growing with utterance length. Vendor-reported table:

  | Model | WER (OpenASR) | Parameters | Latency, MacBook Pro | Latency, Linux x86 | Latency, Raspberry Pi 5 |
  |---|---|---|---|---|---|
  | Medium Streaming | 6.65% | 245M | 107 ms | 269 ms | 802 ms |
  | Small Streaming | 7.84% | 123M | 73 ms | 165 ms | 527 ms |
  | Tiny Streaming | 12.00% | 34M | 34 ms | 69 ms | 237 ms |

  — [Moonshine v2 paper](https://arxiv.org/html/2602.12241); [Moonshine docs (mintlify)](https://www.mintlify.com/moonshine-ai/moonshine) (vendor figures)
- Parakeet: Hacker News commenters say Parakeet v2/v3 beat Moonshine on the OpenASR leaderboard with better RTFx. One third-party page claims Parakeet TDT 0.6B finals arrive about 200 ms after speech end, with no stated methodology (unverified). — [HN thread](https://hackernews.vinext.workers.dev/item/47143755); [VoxPilot benchmarks](https://natearcher-ai.github.io/voxpilot/benchmarks.html)

### Inferences
- Pre-roll: keep a 0.5-1.0 s ring buffer of raw mic audio at all times, and prepend at least 300 ms (OpenAI's default) before the VAD onset when building the turn audio. Silero's own `speech_pad_ms=30` is far too small for first-syllable capture of plosives and soft onsets. Also lower `neg_threshold` / hysteresis, so quiet trailing words ("...to Bob") are not dropped.
- Never transcribe per-VAD-segment and concatenate. Instead, accumulate the whole turn from the first onset (with pre-roll) to the confirmed end, and run final faster-whisper on the full audio once. Use streaming partials (LocalAgreement or Moonshine) only for barge-in word counting and early LLM start. This removes the truncation risk where an early cut drops the last word.
- Adaptive timeout recipe:
  - Base hold is about 0.5 s if Smart Turn says "complete" and about 1.5-2.5 s if it says "incomplete".
  - Extend by about 0.5 s if the partial transcript ends in a conjunction, filler or preposition.
  - Allow a "dictation" mode with a hard cap of 3-5 s.
  - Optionally learn a per-user median pause length (LiveKit's "dynamic" idea).
- If the turn is ended early and the user keeps talking within about 1 s, treat it as a continuation. Cancel the in-flight LLM or TTS and merge the audio, which is what Smart Turn's rerun and LiveKit's preemptive generation imply.

### Gaps
- SimulStreaming CPU latency figures were not retrieved.
- No measured CPU streaming latency for NeMo Parakeet (cache-aware FastConformer) was found from a primary source.
- No published study was found quantifying optimal pre-roll length for first-syllable retention with Whisper.

## 6. Ducking or pausing TTS on suspected barge-in, then resuming if false ("pause then confirm")

### Takeaway
This is a mainstream production pattern. LiveKit pauses agent speech on a suspected interruption, waits `false_interruption_timeout` (2.0 s) for real words, and resumes by default (`resume_false_interruption=True`). Vapi has `backoffSeconds` (1 s). Doing this well needs playback that can pause at a sample position and resume from it, rather than restarting or re-synthesising.

### Cited Findings
- LiveKit: `false_interruption_timeout` 2.0 s is the "silence window after an interruption before it's classified as false". `resume_false_interruption` defaults to True, so "interrupted speech resumes after a false interruption". — [LiveKit tuning](https://docs.livekit.io/agents/logic/turns/tuning/)
- LiveKit's adaptive model needs a median of 216 ms of audio to decide, with inference of 30 ms or less, and rejects 51% of VAD barge-ins as non-interruptions. — [LiveKit blog](https://livekit.com/blog/adaptive-interruption-handling)
- Vapi `backoffSeconds` (default 1 s) controls how long the assistant waits before resuming after an interruption. — [Vapi pipeline configuration](https://docs.vapi.ai/customization/voice-pipeline-configuration) (via search summary)
- Failure modes when resume is done badly: Retell users report the agent replaying an utterance from the beginning after a pause, or speaking the remainder as a separate turn. — [Retell community](https://community.retellai.com/t/issue-fragmented-agent-speech-partial-utterance-plays-then-restarts-from-the-beginning/2498.md) (via search summary)
- In Full-Duplex-Bench v1.5, the best systems for "ignore backchannel and continue" (Gemini 0.93, Nova Sonic 0.98 resume) were also the slowest to stop on real interruptions (about 2.2 s), which is the trade-off "pause then confirm" aims to break. — [Full-Duplex-Bench v1.5](https://arxiv.org/html/2507.23159v1)

### Inferences
- Suggested Voxa state machine while TTS is playing:
  1. **Suspect** (VAD onset at 0.15-0.25 s of speech, and AEC residual or energy above a threshold that the TTS echo cannot explain). Duck TTS by 10-15 dB or pause it, keep the playback pointer, and start buffering mic audio with pre-roll.
  2. **Confirm** (within about 0.4-1.0 s). Real interruption if streaming ASR gives 2 or more non-backchannel words, any stop or command word, or about 0.5 s or more of continuous speech. Then hard-stop TTS, cancel the LLM, and keep the captured audio as the start of the user turn.
  3. **Reject** (only backchannels, no words, a cough or laugh, or silence for about 1-2 s). Restore volume or resume from the paused sample, optionally rewinding about 0.3-0.5 s to the last word boundary so the user does not miss words.
- Ducking is preferable to fully pausing when false positives are common (TV, music). The user hears that Voxa noticed them, but a false trigger does not create an audible gap.
- The suspect onset must exclude Voxa's own voice. Gate the VAD on the AEC-cleaned signal, and also on correlation with the known TTS reference. Otherwise ducking itself changes the echo and can cause oscillation.
- Record the timestamp of what was actually played, so the LLM context only contains text the user actually heard. Truncate the assistant message at the interruption point, as OpenAI Realtime-style truncation does.

### Gaps
- No peer-reviewed study was found comparing duck-vs-pause-vs-stop strategies on user satisfaction or false-stop rates.
- Pipecat's handling of false-interruption resume (whether it has an equivalent of LiveKit `resume_false_interruption`) was not confirmed in sources fetched.
