# How big-tech assistants hear the user while the device is talking (Amazon, Google, Apple, Microsoft, Sonos, Samsung, OpenAI)

Scope note: compiled 2026-11-17 for Voxa (Linux desktop assistant, single mic, laptop speakers, faster-whisper + Silero VAD + local TTS). Each finding carries its publication year. "Documented" means stated in the cited source. Everything under "Inferences" is my reasoning, not a published claim.

## 1. Barge-in during TTS/music playback: detection, wake-word vs open-mic, thresholds/gating

### Takeaway
Every vendor that publishes on this puts the barge-in detector after an acoustic echo canceller (AEC) driven by the loudspeaker reference signal. Two further approaches are documented. Amazon gates on AEC statistics: steady ERLE means it is the device's own playback, and an ERLE collapse means double-talk, so the user is speaking. Amazon also feeds the reference into the detector network itself ("implicit AEC"). Smart speakers use wake-word barge-in. Cloud LLM voice APIs (OpenAI Realtime, Gemini Live) use open-mic VAD barge-in that cancels the response by default, and leave echo removal to the client.

### Cited Findings
- **Amazon patent US10586534B1, "Voice-controlled device control using acoustic echo cancellation statistics"** (filed 2017-09-27, granted 2020-03-10; inventors Argyropoulos, Murgia, Gopalan). The loudspeaker reference goes to the AEC, and AEC statistics measured over the period when a wake word is detected classify that wake word as playback or user speech. [Google Patents](https://patents.google.com/patent/US10586534)
  - Steady-state ERLE is typically about 15–25 dB during playback and about 0 dB with no playback.
  - **Steady ERLE plus high reference-to-mic correlation** means the wake word came from playback: the flag is set to "reject" and playback is not paused.
  - **A sharp ERLE drop or a sharp fall in correlation** means double-talk, a user speaking over playback: the flag is set to "accept" and playback is paused or ducked.
  - It also describes hold-time logic on the Boolean flag, a high-ERLE threshold, an ERLE-based bypass so near-end commands are not wrongly rejected, and ML alternatives (decision trees, neural nets). [Google Patents](https://patents.google.com/patent/US10586534)
- **Amazon, "Implicit Acoustic Echo Cancellation for Keyword Spotting and Device-Directed Speech Detection"** (Cornell, Balestri, Sénéchal; arXiv 2021, IEEE SLT 2022). The KWS and DDD networks take the device playback reference as an extra input channel and learn to ignore the playback. This avoids the alternative of disabling KWS during playback, which would block barge-in.
  - Result: a 56% relative reduction in false-reject rate for device-directed speech detection during device playback, on real Alexa device data.
  - For KWS, accuracy is comparable to or better than a neural AEC + KWS cascade at about 10x less compute. [arXiv 2111.10639](https://arxiv.org/abs/2111.10639)
- **Amazon, "Data Augmentation for Robust Keyword Spotting under Playback Interference"** (Raju, Panchapagesan, Liu, Mandal, Strom; arXiv 2018). The wake-word model is trained on audio mixed with music and TV at varying signal-to-interference ratios, to cope with the residual echo left after AEC. Result: about 30–45% relative reduction in false rejects across a range of false-alarm rates under playback. [arXiv 1808.00563](https://arxiv.org/abs/1808.00563)
- **Amazon, natural turn-taking** (Amazon Science blog, 2020-09-24; Natarajan, Mandal, Ström).
  - When a user interrupts Alexa's speech, Alexa stops talking and processes the new request.
  - "Contextual barge-in": Alexa estimates how far it got into its output from the time between TTS start and the interruption. That position goes to the dialogue manager, which uses it to resolve references such as "that one".
  - Handling barge-ins is named as a key challenge, but the post does not say how Alexa's own voice is excluded. [Amazon Science](https://www.amazon.science/blog/change-to-alexa-wake-word-process-adds-natural-turn-taking)
- **Apple, "Optimizing Siri on HomePod in Far-Field Settings"** (Apple ML Research, 2018-12-03). This is the most detailed public description of the playback front end.
  - Echo can be 30–40 dB louder than far-field speech.
  - Multichannel linear AEC (MCEC) cannot remove all of it, because of nonlinear coupling and non-unique solutions with multichannel content. After MCEC, residual echo is still 10–20 dB above the speech at loud volumes.
  - Residual echo suppression: a DNN trained on real HomePod echo recordings predicts a speech-presence mask. That mask drives a multichannel Wiener filter and then a single-channel suppression gain.
  - Blind source separation produces several candidate streams. The "Hey Siri" detector scores each one and the best stream goes to ASR.
  - Results: WER improved about 90% relative in the playback condition, and absolute false rejects improved 29–30% in the competing-talker condition. The whole front end uses less than 15% of one A8 core. [Apple ML Research](https://machinelearning.apple.com/research/optimizing-siri-on-homepod-in-far-field-settings)
- **Sonos** (ICASSP 2021, Bagheri and Giacobello). "Robust STFT-domain multichannel AEC with adaptive decorrelation of the reference signals" targets very low speech-to-music ratios, with a CPU and memory budget comparable to an always-on wake-word system. [Sonos tech blog](https://tech-blog.sonos.com/posts/sonos-at-icassp-2021/)
  - Sonos patent US10621981, "Tone interference cancellation": after a wake word during music, the already-converged AEC can cancel the device's own acknowledgement tone. A second canceller handles this. [uspto.report](https://uspto.report/patent/grant/10621981)
  - Sonos also has a patent on STFT AEC during playback. [Justia US10446165](https://patents.justia.com/patent/10446165)
- **Google** has a waveform-domain Conformer neural AEC optimised for ASR accuracy (arXiv 2022). Its introduction notes that residual echo hinders hotword recognition, and it cites an earlier log-mel model that takes mixture and reference features as input. [arXiv 2205.03481](https://ar5iv.labs.arxiv.org/html/2205.03481)
  - Google patent application US20230038982 stacks reference-audio features with mic features for joint AEC, enhancement and voice separation feeding ASR. The backend may include a hotword model. [Justia](https://patents.justia.com/patent/20230038982)
- **Microsoft** (Teams; DeepVQE, arXiv 2023). DeepVQE does joint AEC, noise suppression and dereverberation, was state of the art on the ICASSP 2023 AEC Challenge, and was tested on Teams. Microsoft names double-talk as the hard case. [arXiv 2306.03177](https://export.arxiv.org/pdf/2306.03177)
  - A related Microsoft paper adds cross-attention to align the reference with the mic signal, replacing a separate delay-estimation block. [arXiv 2208.11308](https://arxiv.org/pdf/2208.11308.pdf)
  - The AEC Challenge datasets (INTERSPEECH 2021, ICASSP 2023) include single-talk and double-talk scenarios. [Teams blog 2025](https://techcommunity.microsoft.com/t5/microsoft-teams-blog/microsoft-teams-leads-in-audio-quality-echo-cancellation-and/ba-p/4416095)
- **OpenAI Realtime API** (open-mic barge-in, docs current 2025–2026). [OpenAI API reference](https://developers.openai.com/docs/api-reference/realtime-sessions); [OpenAI VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad.md)
  - `server_vad` defaults: `threshold` 0.5, `prefix_padding_ms` 300, `silence_duration_ms` 500.
  - `interrupt_response` defaults to true: a VAD speech-start event cancels the ongoing response.
  - `create_response` defaults to true.
  - The optional `idle_timeout_ms` (5000–30000) counts from when the last response's audio finishes playing.
  - `noise_reduction` takes `near_field` or `far_field`. `far_field` is explicitly for laptop and conference-room mics. It runs before VAD and the model.
- **OpenAI truncation after an interruption.** The client sends `conversation.item.truncate` with `audio_end_ms`, the playback position actually heard, so that the server's transcript drops the part the user never heard. A value larger than the audio received returns an error, and an openai-agents-python fix (PR #4122) clamps it.
  - Source quality: these details come from SDK docs and community threads, not the official page I fetched. [hexdocs](https://hexdocs.pm/ex_openai/2.0.0-beta/ExOpenAI.Components.RealtimeClientEventConversationItemTruncate.html); [OpenAI community](https://community.openai.com/t/openai-realtime-how-to-correctly-truncate-a-live-streaming-conversation-on-speech-interruption-twilio-media-streams/1371637); [GitHub PR](https://github.com/openai/openai-agents-python/pull/4122)
- **Gemini Live API** (Google, docs 2025–2026). [AI Studio Live API docs](https://aistudio.google.com/docs/live-api/capabilities)
  - Automatic activity detection is on by default. On an interruption the ongoing generation is cancelled and discarded, and only what was already sent to the client stays in history.
  - The server sends `BidiGenerateContentServerContent` with `interrupted=true` and discards pending function calls. The client must stop playback and flush its queued audio.
  - Settings: `start_of_speech_sensitivity`, `end_of_speech_sensitivity`, `prefix_padding_ms` and `silence_duration_ms`. Recommended silence is 500–800 ms, the server default is "approximately 800ms", and 100–200 ms splits utterances.
  - The page gives no echo guidance.
  - A developer report says gemini-3.1-flash-live-preview never emits `interrupted` when the user is already speaking as the model's turn begins. This is an unresolved forum report, not official. [Google AI forum](https://discuss.ai.google.dev/t/gemini-3-1-flash-live-preview-live-api-server-never-emits-interrupted-barge-in-when-the-user-is-already-speaking-as-the-models-turn-begins/174346)

### Inferences
- Combining Amazon's ERLE/correlation gating with OpenAI's design gives Voxa a cheap two-detector scheme:
  - Run Silero VAD on the AEC output.
  - Accept a barge-in only when the VAD fires and the AEC indicates double-talk: an ERLE drop, or low coherence between the reference and the mic.
  - Use a hold time so that brief residual-echo bursts do not cut Voxa off.
- On laptop speakers, nonlinear distortion is severe. Apple's data says linear AEC alone leaves 10–20 dB of residual echo at volume, so Voxa likely needs a residual-echo suppressor (neural or spectral) or a reference-aware detector (Amazon iAEC style).
- Neither cloud API removes echo for you. OpenAI's `far_field` noise reduction is not documented as an echo canceller, so a desktop client must run its own AEC (for example WebRTC AEC3 or speexdsp) before streaming, or barge-in will self-trigger.

### Gaps
- I found no published numeric barge-in thresholds from Amazon or Google beyond the patent's typical ERLE ranges (15–25 dB playback, about 0 dB idle). No hold-time values are published.
- I found no published Samsung Bixby barge-in or echo-aware KWS paper. Searches returned only Amazon and NCSOFT work.
- I found no Microsoft Cortana-specific barge-in publication. The Microsoft material is Teams and communications AEC.
- I did not verify how the consumer Gemini Live and ChatGPT Advanced Voice apps do on-device AEC, which is presumably the platform AEC on Android and iOS. No primary source found.

## 2. Self-wake / self-trigger suppression (own TTS, media and ads saying the wake word)

### Takeaway
Amazon and Google both use acoustic fingerprinting of known media containing the wake word, checked on the device against a small set and in the cloud against a large set. Amazon additionally detects "media events" live by spotting identical wake-word audio from at least three customers at the same time. Google's patent family also covers inaudible audio watermarks embedded in media. For the device's own playback, Amazon's ERLE gating (Section 1) and reference-aware models handle self-wake.

### Cited Findings
- **Amazon Science blog**, "Why Alexa won't wake up when she hears her name in Amazon's Super Bowl ad" (Mike Rodehorst, 2019-01-31). [Amazon Science](https://amazon.science/blog/why-alexa-wont-wake-up-when-she-hears-her-name-in-amazons-super-bowl-ad)
  - The fingerprint is Haitsma–Kalker (2002) style: log filter-bank energies on a time–frequency grid, a 2×2-block 2-D gradient, and one sign bit per block.
  - Fingerprints are compared by bit-error rate. Unrelated "Alexa" instances give about 50% bit errors, while the same recording gives far fewer.
  - Noisy audio needs more accumulated data before it matches.
  - On-device: each wake-word detection is checked against a small set of known ad instances, a limit set by device CPU.
  - Cloud: every request is checked against a large library, and also against concurrent requests. A match with at least two other customers' requests counts as a media event, and a cache of averaged fingerprints keeps suppressing it afterwards.
- **Press coverage of the same Amazon disclosure (2019)** says fingerprinting stops about 80–90% of devices responding to TV-originated "Alexa", and that it works worse in noisy rooms or for ads Amazon did not produce. Amazon has patents on these methods. [TechCrunch](https://techcrunch.com/2019/01/31/heres-why-amazons-super-bowl-ad-wont-trigger-alexa/); [PopSci](https://www.popsci.com/why-amazon-alexa-can-ignore-television/)
  - The commonly repeated claim that Amazon ads contain a notch around 4–5 kHz is unconfirmed speculation from a Reddit user. [Entrepreneur](https://entrepreneur.com/business-news/how-amazon-keeps-your-tv-from-accidentally-triggering-your/308552)
- **Google patent US10867600B2, "Recorded media hotword trigger suppression"** (Gruenstein, Schalkwyk, Sharifi; priority 2016-11-07, granted 2020-12-15). [Google Patents](https://patents.google.com/patent/US10867600)
  1. The hotworder fires above a confidence threshold.
  2. A server fingerprints the audio and compares it with a database of known media (TV, movies, music, ads).
  3. If the match score passes a threshold, the server tells the device to suppress the action.
  - Variants: fingerprinting on the device; starting ASR early but halting it on a match; letting the user override.
- **The same Google family adds audio watermarking** (US11244674B2, Ricardo Garcia, granted 2022-02-08; also US11257498B2, US11798543B2, EP3430618). Media creators embed an inaudible watermark. The device detects the hotword plus the watermark and decides whether to run ASR on the audio that follows, based on its source. One variant responds only when the device is on the premises of the media owner and the watermark marks the utterance as high priority. [US11244674](https://patents.google.com/patent/US11244674); [US11257498](https://patents.google.com/patent/US11257498)
- **Google US11972766B2**: "Detecting and suppressing commands in media that may trigger another automated assistant". [Google Patents](https://patents.google.com/patent/US11972766)
- **Amazon US10586534B1** (2020) covers the device's own playback: a wake word detected while ERLE is steady and the reference correlates with the mic is rejected as self-playback (details in Section 1). [Google Patents](https://patents.google.com/patent/US10586534)
- **Apple's "Hey Siri" architecture** (2023): a low-power DNN-HMM detector, then a Conformer checker, speaker ID, and false-trigger mitigation over the full utterance. The article describes no mechanism specific to the device's own playback or echo. [Apple ML Research](https://machinelearning.apple.com/research/voice-trigger)

### Inferences
- Voxa has something no TV-ad system has: the exact TTS waveform it is playing, sample-accurate. That allows much stronger self-suppression than fingerprinting:
  - Correlate the mic signal against the TTS reference, which is the patent's correlation statistic.
  - Or run ASR on the mic, compare the transcript with the TTS text currently playing, and reject any match.
  - The ASR-transcript comparison is my suggestion, not a published method.
- Fingerprinting and watermarking matter for Voxa only if other media on the PC (videos, other apps) could say the wake word. On Linux, PulseAudio or PipeWire monitor sources can supply the system-wide playback reference for that, which is the same idea as Amazon's reference channel.

### Gaps
- No public false-accept numbers for Google's watermark or fingerprint systems.
- Amazon's 80–90% figure is about broadcast media, not self-TTS.
- Whether Alexa and Google use TTS-text matching on their own speech is not documented in what I found.

## 3. Echo-aware / reference-aware VAD, KWS and ASR

### Takeaway
The published trend (2021–2023) is to give the detector the loudspeaker reference as a second input instead of trusting a separate AEC. The main examples are Amazon's iAEC for KWS and DDD and Google's reference-stacked joint AEC front end. Both report large false-reject cuts under playback at low compute.

### Cited Findings
- **Amazon iAEC** (SLT 2022): adding the playback reference as an extra input channel to KWS and DDD networks gives a 56% relative false-reject reduction for DDD during playback, and KWS on par with a neural-AEC cascade at about 10x lower compute. [arXiv 2111.10639](https://arxiv.org/abs/2111.10639)
- **NCSOFT iPhonMatchNet** (not big tech, 2023): zero-shot user-defined KWS that takes the mixed signal, the playback signal and the keyword text, designed for barge-in. [arXiv 2309.06096](https://arxiv.org/abs/2309.06096v2)
- **Google**: a log-mel model taking mixture plus reference features and a waveform Conformer neural AEC optimised for ASR (2022). [arXiv 2205.03481](https://ar5iv.labs.arxiv.org/html/2205.03481)
  - Joint AEC/enhancement/separation feeding ASR and hotword, with reference features stacked onto the input (US20230038982). [Justia](https://patents.justia.com/patent/20230038982)
  - The 2021 keyphrase work builds on "Hotword Cleaner" adaptive noise cancellation. [arXiv 2104.13970](https://arxiv.org/pdf/2104.13970v2.pdf)
- **Apple (2018)**: a DNN speech-presence mask trained on real device echo drives echo suppression, and the hotword detector selects among separated streams. Playback-condition WER improved about 90% relative. [Apple ML Research](https://machinelearning.apple.com/research/optimizing-siri-on-homepod-in-far-field-settings)
- **Amazon (2018)**: augmenting KWS training data with music and TV at varying SIR to model residual echo gave a 30–45% relative false-reject reduction. [arXiv 1808.00563](https://arxiv.org/abs/1808.00563)
- **Microsoft DeepVQE (2023)** and the cross-attention alignment AEC (2022): neural AEC that handles reference-to-mic delay internally. [arXiv 2306.03177](https://export.arxiv.org/pdf/2306.03177); [arXiv 2208.11308](https://arxiv.org/pdf/2208.11308.pdf)

### Inferences
- Silero VAD has no reference input. The cheapest Voxa approximations of iAEC are:
  - (a) Gate the VAD by a reference-energy and coherence test.
  - (b) Train or fine-tune a small two-input VAD on (mic, reference) pairs recorded from the laptop itself, with augmentation in the Amazon 2018 style: user speech plus recorded self-TTS at varying SIR.
- Recording real echo on the target laptop is what Apple did (training on real HomePod echo). Laptop speaker nonlinearity makes this especially valuable.

### Gaps
- No Google paper explicitly titled reference-aware hotword or VAD was found.
- No published reference-aware Silero-class VAD from any of these companies.

## 4. Device-directed speech detection (DDSD) and false-accept reduction

### Takeaway
Apple, Amazon and Google run a second-stage classifier after the trigger that looks at the whole utterance and can veto it. The cues are acoustic (prosody, SNR), ASR-lattice confidence, text semantics, and on camera devices gaze or pose. Apple has since moved to LLM-based multimodal DDSD.

### Cited Findings
- **Apple Siri pipeline** (2023): [Apple ML Research](https://machinelearning.apple.com/research/voice-trigger)
  - An always-on DNN-HMM detector with 23 output states and 4-bit weights.
  - A Conformer checker trained with CTC plus a discriminative loss.
  - LSTM speaker ID.
  - Siri Directed Speech Detection (SDSD) for false-trigger mitigation over the full utterance. It has three components:
    - latticeRNN: true requests have fewer competing ASR hypotheses.
    - aFTM: a streaming acoustic-only transformer using prosody and SNR.
    - ODLD: a BERT-like text classifier that also catches non-vocative uses of "Siri".
  - No accuracy numbers are given.
- **Apple LLM-based DDSD**:
  - "A Multi-signal LLM for Device-directed Speech Detection" (NeurIPS 2024 AFM workshop) frames DDSD as text generation over ASR text plus acoustic signals. [Apple ML Research](https://machinelearning.apple.com/research/llm-device-directed-speech-detection)
  - Multimodal, resource-efficient DDSD with foundation models (Wagner, Churchill, Sigtia et al.). [Apple ML Research](https://machinelearning.apple.com/research/data-resource-efficient)
  - Adaptive knowledge distillation for DDSD. [Apple ML Research](https://machinelearning.apple.com/research/adaptive-knowledge)
  - SELMA does voice-trigger detection, ASR and DDSD in one model. [arXiv 2501.19377](https://export.arxiv.org/pdf/2501.19377)
  - Follow-up turns without a wake word use the previous query as context, so "turn it up" after "play music" counts as directed. [PromptLayer summary](https://www.promptlayer.com/research-papers/how-ai-hears-you-in-noisy-conversations) (secondary source)
- **Amazon Follow-Up Mode / natural turn-taking** (2020): Follow-Up Mode separates device-directed from non-directed speech using acoustic cues. Natural turn-taking adds visual body-pose cues through an on-device fusion model, and future work adds syntactic and semantic features. [Amazon Science](https://www.amazon.science/blog/change-to-alexa-wake-word-process-adds-natural-turn-taking)
  - Amazon's iAEC paper shows that DDD false rejects rise sharply during playback and that a reference input cuts them by 56%. [arXiv 2111.10639](https://arxiv.org/abs/2111.10639)
- **Google Look and Talk** (Nest Hub Max, Google Research blog 2022).
  - Phase 1: an eye-gaze model plus Face Match checks that the enrolled user is looking and is the active speaker. The attention requirement relaxes once speech starts.
  - Phase 2: on-device ASR and intent understanding judge whether the query was meant for the assistant.
  - Users must be within about 5 ft, opted in, with Face Match and Voice Match. No audio goes to the server. [Google Research](https://research.google/blog/look-and-talk-natural-conversations-with-google-assistant/); [9to5Google](https://9to5google.com/2022/05/11/nest-hub-look-and-talk/)

### Inferences
- For Voxa, Apple's three-part SDSD maps onto parts it already has or can add cheaply:
  - faster-whisper's token log-probs and no-speech probability stand in for the lattice confidence.
  - A small text classifier or local LLM prompt plays the role of ODLD ("is this addressed to the assistant?").
  - The same check can reject transcripts that match Voxa's own recent TTS text.

### Gaps
- No public DDSD accuracy, EER or false-accept numbers from Apple's 2023 article or Amazon's 2020 blog.
- The paper numbers for Apple's LLM DDSD were not retrieved.

## 5. Endpointing (end-of-query detection, semantic endpointing)

### Takeaway
Google moved to end-to-end endpointers that share the ASR encoder and predict end-of-query (EOQ) rather than plain silence. Amazon uses acoustic + ASR-decoder features, a two-pass arbitrator, and per-user adaptive (bandit) configurations. OpenAI's `semantic_vad` and Gemini's server VAD expose this as eagerness and sensitivity settings with hard maximum timeouts.

### Cited Findings
- **Google, "Unified End-to-End Speech Recognition and Endpointing"** (Bijwadia, Chang, Li, Sainath, Zhang, He; arXiv 2022, IEEE SLT). One model does both VAD and EOQ. A "switch" connection feeds the endpointer either raw frames (VAD mode) or ASR encoder latents (EOQ mode). [arXiv 2211.00786](https://arxiv.org/abs/2211.00786); [patent app US20240029719](https://patents.justia.com/patent/20240029719)
  - Median endpoint latency fell by 120 ms (30.8%) and P90 by 170 ms (23%), with no WER regression.
  - Continuous-recognition WER improved by 10.6% relative.
- **Google, "Towards Fast and Accurate Streaming E2E ASR"** (Li, Chang et al., 2020): the RNN-T predicts an end-of-sentence token, with early and late penalties. Result: 18.7% relative WER reduction and 40 ms lower median latency versus the earlier RNN-T endpointer on Voice Search. [arXiv 2004.11544](https://arxiv.org/pdf/2004.11544.pdf)
  - Related Google work: the E2E Segmenter (Interspeech 2022) [ISCA](https://www.isca-archive.org/interspeech_2022/huang22_interspeech.pdf), and patent US10929754B2 "Unified endpointer using multitask and multidomain learning" [Google Patents](https://patents.google.com/patent/US10929754).
- **Amazon**:
  - ICASSP 2018 end-of-utterance detector: combines acoustic embeddings, ASR-hypothesis embeddings and decoder features into a 0–1 endpoint score. [Amazon Science](https://www.amazon.science/blog/alexa-scientists-address-challenges-of-end-pointing)
  - Two-pass endpointing: a frame-level end-of-speech classifier, then an "EP Arbitrator" that verifies the first-pass endpoint. [Amazon Science PDF](https://cdn.amazon.science/4b/7c/0903caee4ab2a19c5f185b0a5c1c/two-pass-endpoint-detection-for-speech-recognition.pdf)
  - Adaptive endpointing with contextual multi-armed bandits: chooses a "standard" or a "relaxed" configuration per utterance, where "relaxed" suits slow speakers and mid-utterance pauses. [Amazon Science PDF](https://cdn.amazon.science/bb/e4/a2eb7078430095b60d1e8af610f2/adaptive-endpointing-with-deep-contextual-multi-armed-bandits.pdf)
  - "Towards accurate and real-time end-of-speech estimation". [Amazon Science PDF](https://cdn.amazon.science/87/e5/87ed64524f5c9611c97bf42fc7b8/towards-accurate-and-real-time-end-of-speech-estimation.pdf)
  - Amazon patents "Language model speech endpointing" (US10121471) [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10121471) and "Domain specific endpointing" (US10854192B1) [Google Patents](https://patents.google.com/patent/US10854192).
  - The 2020 "user pacing" beta uses fillers ("um", "uh"), lengthened vowels and incomplete utterances to decide whether the user has finished. [Amazon Science](https://www.amazon.science/blog/change-to-alexa-wake-word-process-adds-natural-turn-taking)
- **OpenAI `semantic_vad`**: a model decides whether the user has finished. Eagerness low, medium (the default, "auto") and high have maximum waits of 8 s, 4 s and 2 s. [OpenAI API reference](https://developers.openai.com/docs/api-reference/realtime-sessions)
- **Gemini Live**: server end-of-speech silence defaults to about 800 ms, with 500–800 ms recommended.
  - "Hybrid VAD": the client sends `audioStreamEnd` when its own VAD sees end of speech, forcing immediate finalisation, and server VAD stays on as the fallback.
  - With manual VAD, the client must send pre-speech context and use at least 500 ms of silence. [AI Studio docs](https://aistudio.google.com/docs/live-api/capabilities)
- **Google Assistant (I/O 2022)**: Assistant became more patient with fillers and pauses. [Chrome Unboxed](https://chromeunboxed.com/google-assistant-can-now-be-activated-without-a-hot-word-will-be-patient-with-filler-words-pauses/); [Google blog](https://blog.google/products/assistant/assistant-io-2022/)

### Inferences
- A practical Voxa endpointer:
  - Silero VAD silence of about 500 ms as a candidate endpoint, then a semantic check on the partial transcript ("does this look complete?").
  - Extend the wait up to a cap of about 2–4 s when the transcript is incomplete or ends in a filler. This mirrors OpenAI eagerness and Amazon's two-pass arbitrator.
- Amazon's bandit work suggests adapting the silence threshold per user over time.

### Gaps
- Amazon endpointing papers' accuracy and latency numbers were not retrieved.
- Gemini's start and end sensitivity enum semantics are not quantified in the docs.

## 6. Follow-up mode / continued conversation and interruption in Alexa, Google/Gemini Live, Siri, ChatGPT voice

### Takeaway
The legacy assistants open a short wake-word-free listening window after a response (Alexa Follow-Up and Conversation Mode, Google Continued Conversation and Look and Talk, Siri follow-ups) and rely on DDSD to reject side talk. LLM voice systems (OpenAI Realtime, Gemini Live) keep the mic open the whole time and cancel the response on a VAD speech start. The client must then stop playback, flush its buffer, and (OpenAI) truncate the assistant item to what was actually heard.

### Cited Findings
- **Alexa**:
  - Follow-Up Mode uses acoustic device-directedness. [Amazon Science](https://www.amazon.science/blog/change-to-alexa-wake-word-process-adds-natural-turn-taking)
  - Conversation Mode (2021) is opt-in per session on Echo Show 8 (2nd Gen) and Echo Show 10 (3rd Gen). It uses visual and acoustic cues, ends automatically after a short period without a request, and starts with "Alexa, join the conversation". Alexa pauses when interrupted. [About Amazon](https://aboutamazon.com/news/devices/conversation-mode-helps-interactions-with-alexa-feel-more-natural); [Voicebot](https://voicebot.ai/2021/11/18/new-alexa-conversation-mode-skips-wake-word-repetition/)
  - Contextual barge-in passes the interruption offset into the TTS to the dialogue manager (2020).
- **Google**: Look and Talk (2022) does on-device gaze, face and voice matching followed by on-device intent. "Hey Google" still works, and tapping the screen cancels. [Google Research](https://research.google/blog/look-and-talk-natural-conversations-with-google-assistant/)
- **Gemini Live API**:
  - VAD barge-in cancels and discards generation, sends `interrupted`, and the client flushes audio.
  - "Proactive audio" lets the model decline to respond to irrelevant input. It is always on for Gemini 3.8 Live, configurable on 2.5 Flash Live, and unsupported on 3.1 Flash Live.
  - `turn_coverage` defaults differ by model. [AI Studio docs](https://aistudio.google.com/docs/live-api/capabilities)
- **OpenAI Realtime / Advanced Voice**:
  - `interrupt_response` (default true) cancels on speech start.
  - The client truncates the assistant item at `audio_end_ms`.
  - `semantic_vad` handles turn end. [OpenAI](https://developers.openai.com/docs/api-reference/realtime-sessions)
- **Siri**: follow-up DDSD uses the previous query as context. [Apple ML Research](https://machinelearning.apple.com/research/llm-device-directed-speech-detection)

### Inferences
- "Never interrupt itself" in these APIs depends entirely on the client's echo removal. With a bad AEC, the default `interrupt_response=true` makes the assistant cut itself off on its own voice.
  - The Voxa design should therefore require stronger evidence for barge-in while TTS is playing than when idle: a higher VAD threshold, the double-talk condition, a minimum duration of about 200–300 ms, and optionally a non-empty ASR result that does not match the TTS text.
  - The 200–300 ms figure is an engineering suggestion, not a published number.
- Truncating the transcript to the heard position (OpenAI) and passing the interruption offset to dialogue (Alexa) both mean Voxa should log the playback position at barge-in. The LLM should know which part of its answer the user actually heard.

### Gaps
- No official technical description of how the consumer ChatGPT Advanced Voice or Gemini Live apps suppress their own voice on phones and laptops; this is presumably platform AEC, unverified.
- No Siri barge-in mechanism description was found.
- Samsung Bixby (continued conversation, barge-in) and Microsoft Cortana: no primary technical sources found in this pass.

## Key patent and paper index (for the report writer)
- **Amazon**:
  - US10586534B1 (AEC-statistics wake-word gating, granted 2020)
  - arXiv 2111.10639 (iAEC, SLT 2022)
  - arXiv 1808.00563 (playback-augmented KWS, 2018)
  - Super Bowl fingerprinting blog (2019)
  - Natural turn-taking blog (2020)
  - Endpointing papers (2018+) and patents US10121471 and US10854192B1
- **Google**:
  - US10867600B2 (media hotword fingerprint suppression, priority 2016)
  - US11244674B2, US11257498B2, US11798543B2 and EP3430618 (watermark and fingerprint family)
  - US11972766B2
  - US10929754B2 (unified endpointer)
  - US20240029719 and arXiv 2211.00786 (E2E ASR + endpointing)
  - arXiv 2004.11544
  - arXiv 2205.03481 (neural AEC for ASR)
  - US20230038982
  - Look and Talk (2022)
- **Apple**:
  - HomePod far-field (2018)
  - Voice trigger / SDSD (2023)
  - LLM DDSD (2024)
  - SELMA (2025)
- **Microsoft**: DeepVQE (arXiv 2306.03177) and the cross-attention AEC (arXiv 2208.11308)
- **Sonos**: ICASSP 2021 robust STFT multichannel AEC, US10621981 and US10446165
- **Samsung**: none found
