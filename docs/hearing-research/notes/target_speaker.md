# Target/personalized speaker detection and own-TTS exploitation for Voxa

Scope: techniques that use (a) the enrolled user's voice identity and (b) Voxa's exact knowledge of the TTS audio/text it is playing, so a single-mic Linux assistant (faster-whisper, Silero VAD, local TTS) accepts only the user and rejects residual echo of its own synthetic voice. Research done 2026-11-17, about 25 tool calls; several numbers are only in paper abstracts or model cards and are marked as such.

## Q1. Personal VAD, Personal VAD 2.0, TS-VAD, VoiceFilter / VoiceFilter-Lite, speaker-conditioned enhancement: sizes, CPU cost, open implementations

### Takeaway
Personal VAD (a speaker-conditioned VAD with three classes: non-speech, target speech, non-target speech) is the cheapest fit for Voxa: about 130K parameters in v1, and about 1.0 MB int8 with about 9.6 MFLOPs in v2. Google released no code. The usable open options are FireRedChat's released streaming pVAD (ECAPA-TDNN conditioned, built for barge-in), the unofficial pirxus/personalVAD, and SelfSupervisedPVAD. VoiceFilter-Lite style target-speaker extraction is heavier and the open versions are reported to be unreliable. Personalized enhancement models from the DNS challenge (PercepNet-class) run on a small fraction of a CPU core but have no turnkey Linux package.

### Cited Findings
- **Personal VAD v1 (Ding et al., Odyssey 2020):** a VAD-like network conditioned on either the target speaker embedding or a speaker-verification score. It outputs per-frame probabilities for non-speech, target-speaker speech and non-target-speaker speech. The best setup has only 130K parameters and beats a baseline that combines separately trained standard VAD and speaker recognition networks. Google gives the motivation as cutting ASR compute and battery use by gating ASR on the target user. No official code. — [Google speaker-id project page](https://google.github.io/speaker-id/publications/PersonalVAD); [arXiv 1908.04284](https://arxiv.org/pdf/1908.04284v4)
- **Personal VAD 2.0 (Ding et al., Interspeech 2022):** adds a Conformer backbone (streaming, limited left context), FiLM conditioning on the d-vector, a "speaker pre-net" that feeds frame-wise cosine similarity to the enrolled embedding into FiLM, 8-bit dynamic-range quantization to TFLite, and training in which 20% (p0=0.2) of utterances get a zero embedding so the model also works with no one enrolled. — [arXiv 2204.03793](https://arxiv.org/abs/2204.03793v3); [ISCA archive](https://www.isca-archive.org/interspeech_2022/ding22_interspeech.html)
- **Personal VAD 2.0 numbers (Tables 1 and 2 of the paper):** ASR WER on a "Concat" (conversational, multi-speaker) test set:

  | Model | Concat WER | Size | FLOPs |
  |---|---|---|---|
  | Personal VAD v1 | 41.0% | 5.8 MB | 3.54 M |
  | Conformer + FiLM + cosine | 27.6% | 4.0 MB | 9.58 M |
  | + int8 quantization | 27.2% | 1.0 MB | 9.58 M |

  The final model with enrollment-less training (E5) scores 32.7% Concat and 12.4% Non-Concat WER. The authors say its extra FLOPs over a standard VAD (0.7-1.5 MB) do not significantly affect overall ASR latency, because the VAD needs far less compute than the ASR. They also note that when the target and non-target voices are very similar the model errs more, and tuning the threshold trades deletions for insertions. — [arXiv 2204.03793 PDF](https://arxiv.org/pdf/2204.03793v3)
- **FireRedChat pVAD (Xiaohongshu FireRed Team, arXiv 2509.06502, Sept 2025):** a streaming personalized VAD for full-duplex barge-in control.
  - Architecture: a causal 2D conv-recurrent VAD; an ECAPA-TDNN target embedding concatenated along the channel axis; a GRU; a per-step speaking probability (the paper text suggests about 10 ms steps).
  - Training: about 2,000 h of Mandarin and English, mixed with interfering speakers or MUSAN noise at 0-30 dB SNR.
  - Barge-in results on 1,000 Chinese plus 1,000 English utterances:

    | System | False barge-in rate | T90 latency |
    |---|---|---|
    | FireRedChat | 10.2% | 170 ms |
    | LiveKit | 33.4% | 140 ms |
    | TEN | 78.1% | 90 ms |

  - The paper gives no model size or license, and it does not explain how the system's own TTS echo is kept from triggering pVAD.
  — [arXiv 2509.06502 HTML](https://arxiv.org/html/2509.06502v1)
- **FireRedChat release:** the repo README says pVAD and turn-detector models were released 2025-09-16. Its components are forks of LiveKit agents. A Hugging Face commit in FireRedTeam/FireRedChat-pvad includes SpeechBrain ECAPA files. — [FireRedChat demo/paper page](https://fireredteam.github.io/demos/firered_chat); [HF FireRedChat-pvad commit](https://huggingface.co/FireRedTeam/FireRedChat-pvad/commit/f3712c517180dac84ac449f3dc8db765d665d5c1)
- **Other open Personal VAD code:**
  - `pirxus/personalVAD`: an unofficial bachelor's-thesis implementation, about 81 stars, last updated Sept 2022. — [relatedrepos listing](https://relatedrepos.com/gh/pirxus/personalVAD)
  - `HolgerBovbjerg/SelfSupervisedPVAD`: the code for arXiv 2312.16613, a self-supervised personalized VAD on log-mel features with a d-vector target. — [arXiv 2312.16613](https://arxiv.org/html/2312.16613v1)
- **VoiceFilter-Lite (Wang et al., Interspeech 2020):**
  - Design: single-channel and streaming. It uses mostly LSTMs (no bidirectional or temporal-conv layers, 1D convolutions only) and works on ASR features instead of waveforms.
  - Size and accuracy: the quantized model is 2.2 MB. It gives a 25.1% WER improvement on overlapped speech without hurting clean speech, and it applies stronger suppression adaptively when it detects overlap.
  - It is bypassed when the speaker has not enrolled. No official code.
  — [Google project page](https://google.github.io/speaker-id/publications/VoiceFilter-Lite); [Google Research blog](https://research.google/blog/improving-on-device-speech-recognition-with-voicefilter-lite/); [arXiv 2009.04323](https://arxiv.org/abs/2009.04323v1)
- **Original VoiceFilter, unofficial implementations:**
  - `mindslab-ai` / `maum-ai/voicefilter` (PyTorch) reports median SDR of 1.9 dB before and 10.2 dB after, against the paper's 2.5 and 12.6 dB.
  - Another user reported SDR converging at only 4.5 dB, and the repo itself warns it "can be quite unreliable".
  - Alternatives: Edresson/VoiceSplit, kooBH/VFWS.
  — [maum-ai/voicefilter](https://github.com/maum-ai/voicefilter); [issue #5](https://github.com/maum-ai/voicefilter/issues/5); [VoiceSplit](https://edresson.github.io/VoiceSplit)
- **WeSep (WeNet team, arXiv 2409.15799, 2024):** an open target-speaker-extraction toolkit. It integrates with WeSpeaker and offers several separation backbones (e.g., ConvTasNet) and fusion methods. Its paper says models export to TorchScript or ONNX with C++ deployment samples and pretrained models. I did not verify which checkpoints are published. — [arXiv 2409.15799](https://arxiv.org/html/2409.15799v1)
- **Personalized speech enhancement (DNS challenge personalized tracks):**
  - Personalized PercepNet (Amazon, Interspeech 2021): base PercepNet runs in real time on under 5% of a CPU core, and the personalized version adds a speaker embedding at similar complexity. — [ar5iv 2106.04129](https://ar5iv.labs.arxiv.org/html/2106.04129)
  - TEA-PSE: ECAPA-TDNN plus a dual-stage network, 1st in ICASSP 2022 track 2.
  - TEA-PSE 2.0: needs only 21.9% of the multiply-accumulate operations of TEA-PSE.
  - Pärnamaa & Saabas (2024): drop the separate speaker encoder and allow auto-enrollment. — [arXiv 2406.09928](https://arxiv.org/html/2406.09928v1)
  - Challenge overview: [NPU-Elevoc arXiv 2303.06811](https://arxiv.org/pdf/2303.06811)
- **Target-speaker VAD for diarization:** online TS-VAD exists (Wang & Li, TASLP 2024, arXiv 2310.08696), but it targets meeting diarization with several speakers. Personal VAD remains the lightweight real-time variant for a single user. — [arXiv 2310.08696](https://arxiv.org/pdf/2310.08696v1.pdf); [arXiv 2207.05920](https://arxiv.org/pdf/2207.05920)

### Inferences
- For Voxa, a pVAD-style gate (Silero VAD followed by a per-segment speaker-similarity check, or a true frame-level pVAD) is the right cost tier. 1-10 MFLOPs per 10 ms frame is negligible next to faster-whisper.
- Voxa's TTS voice is a fixed, known "non-target speaker". Personal VAD's non-target-speech class is exactly the label echo would get, so a pVAD fine-tuned with Voxa's own TTS voice as the interferer should do better than a generic one.
- Speaker extraction (VoiceFilter-Lite, WeSep, PSE) is only needed if Voxa must transcribe the user correctly while TTS is still playing loudly (double talk). For deciding whether the user is speaking, a gate is enough and much cheaper.
- FireRedChat's pVAD is the most practical open pretrained starting point. Its released size and license need checking in the repo before use.

### Gaps
- No published CPU latency (ms per frame on x86) for any Personal VAD variant. Personal VAD 2.0 gives only FLOPs and MB.
- FireRedChat pVAD's parameter count, license and English-only accuracy were not stated in the paper. I did not open the repo or the HF model card.
- No source evaluates any pVAD against residual echo of the device's own TTS specifically.

## Q2. CPU speaker embeddings, cheap enrollment, rejecting the TTS voice, and anti-spoofing as an "is this my own TTS" detector

### Takeaway
Several Apache/MIT-licensed speaker encoders reach about 0.6-1.2% EER on VoxCeleb1-O at 6-26 MB, run on CPU, and export to ONNX. Enrolling from 5-30 s of the user's speech is the norm. A different-identity TTS voice should be rejected as easily as any other impostor. The hard case is user speech overlapping TTS echo, not a pure TTS segment. Generic anti-spoofing detectors (AASIST) are tiny but generalize poorly (12-43% EER on newer or in-the-wild data), so they are a weak own-TTS detector. Voxa can do better by enrolling its own TTS voice as a negative class, or by watermarking (Q3).

### Cited Findings
- **SpeechBrain ECAPA-TDNN `spkrec-ecapa-voxceleb`:** Apache-2.0, trained on VoxCeleb1+2, AM-softmax loss. EER is 0.80% on VoxCeleb1-test (cleaned) in the current card, and 0.69% in an older card revision. The checkpoint file is about 83 MB. I found no parameter count; ECAPA-TDNN is commonly about 20M, which is unverified here. — [HF model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb); [older revision](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/5a40f2c96080d36cc8d73d0560ec5eabf96d6209/README.md)
- **WeSpeaker ResNet34 (`wespeaker-voxceleb-resnet34-LM`):** 256-dim embeddings from 80-bin fbank input. EER is 0.723% on Vox1-O (0.867% Vox1-E, 1.532% Vox1-H) with large-margin fine-tuning plus AS-Norm. The ONNX file is 26.5 MB. It is officially exported to ONNX with an onnxruntime inference script, and the Python API runs on CPU. pyannote uses it for diarization. — [WeSpeaker GitHub](https://github.com/wenet-e2e/wespeaker/); [pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md); [HF file listing](https://huggingface.co/openspeech/wespeaker-models/tree/main); [Mixpeek listing](https://mixpeek.com/model/pyannote/wespeaker-voxceleb-resnet34-LM)
- **3D-Speaker (Alibaba/ModelScope):** VoxCeleb1-O EER is 0.65% for CAM++ (7.2M params) and 0.61% for ERes2NetV2 (17.8M). An ONNX Runtime release exists, and int8 ERes2NetV2 ONNX community ports are described as usable with sherpa-onnx. — [3D-Speaker-Toolkit arXiv 2403.19971](https://arxiv.org/pdf/2403.19971); [CAM++ arXiv 2303.00332](https://arxiv.org/pdf/2303.00332); [GitHub](https://github.com/modelscope/3D-Speaker)
- **NVIDIA NeMo TitaNet:** TitaNet-S has about 6M parameters and 1.15% EER on VoxCeleb per a benchmark listing; TitaNet-M 0.81%, TitaNet-L 0.68%. The sources disagree on TitaNet-L: the paper body says 25.3M params, while the NGC card says 6.4M params and 1.08% EER. Embeddings are 192-dim, and there is a `verify_speakers` helper. — [arXiv 2110.04410](https://arxiv.org/pdf/2110.04410); [NGC titanet_small](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/titanet_small)
- **Resemblyzer (GE2E d-vector, Apache-2.0):** produces 256-dim embeddings. The README cites about 1000x real time on a GTX 1080 (no CPU figure given), recommends 5-30 s of speech to build a voice profile, and notes it works best on English. It is older and less accurate than the models above (no current EER found). — [Resemblyzer mirror](https://gitee.com/wolverineq/Resemblyzer); [GE2E page](https://google.github.io/speaker-id/publications/GE2E)
- **AASIST anti-spoofing:** about 297K parameters; AASIST-L has 85K. On ASVspoof 2019 LA, AASIST scores 0.83% EER (min t-DCF 0.0275) and AASIST-L 0.99%. The same model scores 12.35% on ASVspoof2021 LA, 17.04% on 2021 DF and 43.01% on In-the-Wild. I found no CPU latency figures. — [AASIST code](https://github.com/entn-at/aasist); [HF benchmark card](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST); [Scalable AASIST arXiv 2507.11777](https://arxiv.org/html/2507.11777v1)
- **Spoofing-aware SV setups** report SV-EER (rejecting other people) and SPF-EER (rejecting synthetic speech) as separate metrics. — [SpoofCeleb arXiv 2409.17285](https://arxiv.org/pdf/2409.17285)

### Inferences
- **Recommended stack:** Silero VAD segment, then a CAM++ or WeSpeaker ResNet34 ONNX embedding (or ECAPA via SpeechBrain), then cosine scoring against:
  1. the user centroid (enrolled from about 3-6 utterances, roughly 20-30 s, ideally at several distances and volumes);
  2. a Voxa-TTS centroid, built for free by embedding Voxa's own TTS output (both clean and re-recorded through the speaker and mic).
  Accept only if user-score minus TTS-score clears a margin. This "anti-enrollment" turns TTS rejection into a two-class closed-set decision, which is much easier than open-set anti-spoofing. This is my design inference; no paper evaluates exactly this.
- Because Voxa generates the TTS itself, it can store the TTS embedding per voice, or even per utterance, and it never needs a generic deepfake detector. AASIST-type detectors would add false rejects for little gain, given their out-of-domain EERs.
- Embeddings over short segments (under 1-1.5 s) and over user+echo mixtures will be noisy. Gate decisions should therefore use segment-level scores with hysteresis, not single 10 ms frames.

### Gaps
- No source measured how reliably these SV models reject a specific local TTS voice (e.g., Piper or Kokoro) replayed through a loudspeaker. This should be measured on Voxa's hardware.
- No CPU latency numbers were found for any of these encoders. They need benchmarking: run CAM++, ResNet34 and ECAPA ONNX on 1-3 s segments.
- Exact parameter count for SpeechBrain ECAPA was not confirmed.

## Q3. Reference-conditioned approaches: echo-conditioned ASR/KWS, known-interference removal, own-TTS watermarking

### Takeaway
The strongest published evidence is that using the playback reference signal beyond the linear AEC brings large gains when the playback is TTS. Google's 1.6M-parameter neural residual echo suppressor cut WER 38-53% beyond linear AEC. NCSOFT's iPhonMatchNet adds reference conditioning to a 656K keyword spotter for +0.13% parameters and cut false activations about 95% on self-playback. Watermarking Voxa's own TTS with AudioSeal (MIT, streaming, per-sample localization) is a novel cheap "is this my own voice" tag. Its survival through loudspeaker, room and mic is not documented.

### Cited Findings
- **Neural residual echo suppression for ASR (Panchapagesan, Shabestary, Narayanan, Google, Interspeech 2023):**
  - Design: a TasNet-style waveform model with 4 causal conformer layers and 1.6M parameters. It runs after a linear AEC and keeps the playback reference as an auxiliary input. Past context is about 310 ms with a 2.5 ms frame shift. It uses an auxiliary ASR loss.
  - Results: 38-53% WER reduction over linear AEC alone. Training with a deliberately weaker linear AEC generalizes 17-30% better.
  - Context: the authors stress that residual echo especially hurts ASR and hotword detection at high volume and when playback is speech (TTS, podcasts). The training echoes included re-recorded internal TTS. The test set covered 1.3 m, 3.3 m and 5.2 m speaker distances.
  - Related prior work they cite: a waveform neural AEC cascaded after linear AEC gave 56-59% WER reduction.
  — [ISCA PDF](https://www.isca-archive.org/interspeech_2023/panchapagesan23_interspeech.pdf)
- **iPhonMatchNet (Lee & Cho, NCSOFT, arXiv 2309.06096, Dec 2023):** a zero-shot user-defined KWS that takes both the mic mix and the playback signal.
  - How it works: a shared encoder and a mask "refiner" do implicit AEC, and a conv mask subnet handles echo delay.
  - Self-playback false-alarm results, as mean absolute error (lower is better):

    | Dataset | PhonMatchNet (no reference) | iPhonMatchNet |
    |---|---|---|
    | Google Speech Commands | 73.97% | 0.15% |
    | LibriPhrase-hard | 85.40% | 3.11% |

  - Size: 655K to 656K parameters. It gets slightly worse with no playback (EER 6.77% to 8.09% on GSC), and an NLMS AEC baseline gave no significant gain in the self-referencing case.
  — [arXiv 2309.06096](https://arxiv.org/html/2309.06096v3)
- **Google joint AEC + enhancement + separation front end:** takes the playback reference as a side input, assumed time-aligned and of equal length. — [arXiv 2111.10639](https://arxiv.org/pdf/2111.10639.pdf)
- **Reference fed to backend classifiers (KWS / device-directed detection):** access to the reference "allows the KWS or DDD classifier to disambiguate between target and interferer" but "could bias the model to learn unintended characteristics of the interferer". Approaches based on the TTS text apply only to TTS playback, not to music or podcasts. — [arXiv 2111.09935](https://arxiv.org/pdf/2111.09935.pdf)
- **AudioSeal (San Roman et al., Meta, ICML 2024):**
  - How it works: a generator/detector pair with sample-level localized detection. The detector outputs a logit at every time step in one pass, so it suits real-time streams, and is up to two orders of magnitude faster than WavMark.
  - Robustness: robust to compression, re-encoding and noise. It can also carry a multi-bit message.
  - Licensing and streaming: MIT for code and weights since April 2024. Version 0.2 adds a causal streaming generator and detector (`audioseal_wm_streaming` / `audioseal_detector_streaming`) with 320-sample frames, Python 3.10+, PyTorch.
  - Not documented: model size and over-the-air (loudspeaker to mic) robustness.
  — [arXiv 2401.17264](https://arxiv.org/pdf/2401.17264); [PMLR](https://proceedings.mlr.press/v235/san-roman24a.html); [GitHub](https://github.com/facebookresearch/audioseal); [project page](https://pierrefdz.github.io/publications/audioseal)

### Inferences
- **Cheapest win for Voxa:** Voxa has the exact TTS waveform, so it should first run a proper linear AEC with that reference (Q3 overlaps another researcher's AEC scope). A pretrained reference-conditioned residual suppressor is the next tier, but no open checkpoint equivalent to Google's RAES turned up.
- **Known-interference subtraction:** with the exact TTS waveform, Voxa could also compute a short-time cross-correlation or coherence between the mic and reference per VAD segment. High coherence with no user-embedding match means echo, which can be dropped. This is a signal-processing inference, not from a cited paper.
- **AudioSeal for own-TTS leak detection:** worth a bench test. Embed a watermark in every TTS chunk, then run the streaming detector on mic audio. Frames flagged as watermarked are echo; unflagged speech energy is a candidate user. The risks are that the speaker, room and mic path may destroy the mark (untested), and that the detector costs PyTorch CPU time.
- Conditioning a small classifier (iPhonMatchNet-style) on the reference is the research-grade route to a "user is barging in" detector. It would need training data made from Voxa's TTS.

### Gaps
- No open-source pretrained reference-conditioned residual echo suppressor or echo-aware ASR was found in this search.
- AudioSeal model size, CPU throughput and over-the-air robustness are unreported. Other inaudible-marker schemes (e.g., WavMark) were not researched in depth.
- No paper found that conditions Whisper-class ASR on the TTS text to suppress self-transcription.

## Q4. Better own-speech text filtering via alignment and word timing

### Takeaway
Whisper's native timestamps are unreliable at word level. CTC forced alignment (torchaudio MMS-FA, used by WhisperX) gives better word timings, and stable-ts can force-align a known transcript. For Voxa, the strongest text filter compares ASR words, with their timestamps, against TTS words at the times they were actually played. TTS engines can report per-word timing directly, which needs no alignment of the TTS side.

### Cited Findings
- **WhisperX (Bain et al., Interspeech 2023):** uses VAD segmentation plus an external phoneme CTC model for word alignment, because Whisper's own timestamps are not reliable enough. Default English alignment models come from torchaudio pipelines, others from Hugging Face. — [arXiv 2303.00747](https://arxiv.org/html/2303.00747v2); [Interspeech PDF](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)
- **Aligner comparison:** a 2024 study on TIMIT and Buckeye found Montreal Forced Aligner more accurate than WhisperX and MMS at word and phone boundaries. — [arXiv 2406.19363](https://ar5iv.labs.arxiv.org/html/2406.19363)
- **torchaudio MMS-FA:** reported as slated for deprecation (verify), with a CTC aligner package suggested as a commercially licensed alternative. — [Trelis article](https://trelis.substack.com/p/align-audio-and-text-for-speech-recognition)
- **stable-ts:** offers forced alignment of a known transcript to audio. — [force-align-wordstamps listing](https://awesome.ecosyste.ms/projects/github.com%2Fcrone-ai%2Fforce-align-wordstamps)

### Inferences
- **A design for Voxa:**
  1. Log a playback timeline of (word, start, end) for each TTS word, from TTS engine timing (Piper and Kokoro expose phoneme durations) or by force-aligning the TTS audio once at synthesis time.
  2. Get faster-whisper word timestamps (`word_timestamps=True`) for mic transcripts.
  3. Drop ASR words that fuzzy-match a TTS word played within about ±300-500 ms (accounting for output latency and room delay).
  4. Keep words that do not match the TTS words at that time.

  This beats sentence-level matching. It handles partial echoes, and it keeps a user who interrupts with words that differ from what Voxa is saying. The design is my inference; I found no paper evaluating it.
- Combine with Q2: keep a word only if it is not time-matched to TTS or its segment matches the user's embedding.

### Gaps
- No paper found that evaluates time-aligned ASR-vs-TTS text matching as a self-echo filter. The window size needs measuring on Voxa's audio path.
- faster-whisper word-timestamp accuracy was not measured in the sources retrieved.

## Q5. Risks: false rejection of the user, multiple users, privacy

### Takeaway
Speaker gating trades echo rejection for false rejections. Emotion, arousal, health (colds), vocal effort, distance and codec or channel mismatch all degrade SV, and enrolling in only one condition makes it worse. Multi-condition enrollment, a margin against the TTS centroid instead of an absolute threshold, and a fallback mode (e.g., accept with the wake word, or with no enrollment like PVAD 2.0's zero-embedding mode) reduce the risk. Voiceprints are biometric data and should stay local.

### Cited Findings
- **Emotion and arousal:** SV degrades on non-neutral speech. Synthetic emotional augmentation cut EER by up to 3.64% relative (Amazon, 2024), and emotion-invariant training gave 19.29% relative EER reduction (2025). Enrollments are usually neutral, reliability falls at extreme arousal, and mixing enrollments across arousal levels helped. — [arXiv 2412.00319](https://arxiv.org/html/2412.00319v1); [arXiv 2505.18498](https://export.arxiv.org/pdf/2505.18498); [Rusko et al. 2018](https://elib.dlr.de/118727/)
- **Mismatch factors and demographic gaps:** the SV literature lists speaking style, short-term health, emotion and vocal effort as factors. The SVeritas benchmark reports robustness gaps across age, gender and language background, plus codec stress. — [UEF paper](https://erepo.uef.fi/bitstream/handle/123456789/7740/1567597761506397310.pdf); [arXiv 2509.17091](https://arxiv.org/pdf/2509.17091v1)
- **Similar voices:** Personal VAD 2.0 errs more when the target and non-target voices are similar, a trade-off tuned by threshold. — [arXiv 2204.03793](https://arxiv.org/pdf/2204.03793v3)
- **Enrollment-less fallback:** Personal VAD 2.0's zero-embedding training keeps standard-VAD-level WER when no one is enrolled (7.0% VS and 10.1% Non-Concat, against 6.9% and 10.1% for the Conformer standard VAD). — [arXiv 2204.03793](https://arxiv.org/pdf/2204.03793v3)
- **Unenrolled speakers:** VoiceFilter-Lite is bypassed when the speaker has not enrolled, so other household members are not served by a personalized gate unless they enroll. — [Google blog](https://research.google/blog/improving-on-device-speech-recognition-with-voicefilter-lite/)
- **Auto-enrollment and privacy:** auto-enrollment that needs no separate embedding model has been proposed partly for privacy. — [arXiv 2406.09928](https://arxiv.org/html/2406.09928v1)

### Inferences
- Gate only while TTS is playing. When Voxa is silent, accept any speech, so colds and distance only matter during barge-in. This sharply limits the cost of false rejections.
- Support several enrolled users (a list of centroids) and keep the TTS negative centroid. Refresh user centroids from high-confidence accepted utterances, with a cap to avoid drift.
- Keep embeddings on disk locally and never upload them. Let the user delete them and re-enroll.

### Gaps
- No quantitative sources found on SV false rejection from colds or illness, or from far-field distance, for these specific models.
- No data on EER as a function of enrollment duration for CAM++, ECAPA or ResNet34.
