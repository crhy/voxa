# State of the art in acoustic echo cancellation (AEC) for Voxa (single-mic Linux voice assistant, CPU only)

Research date: 2026-11-17. About 19 tool calls. Primary sources were preferred: WebRTC source and config, arXiv papers, the Microsoft challenge paper, and PyPI/Hugging Face model cards. Where the evidence is anecdotal or comes from my own background knowledge and was not checked in this session, it is marked as such.

## 1. WebRTC AEC3 internals, performance, and Python access

### Takeaway
AEC3 is a block-based, partitioned frequency-domain linear canceller. It runs two filters, a "refined" (main) and a "coarse" (shadow) one, and crossfades between their outputs. It aligns render and capture with a decimated matched-filter delay estimator and follows the linear stage with a per-band residual-echo suppressor whose gains depend on ERLE estimates. It has an explicit clock-drift mode (`has_clock_drift`) and comfort noise. Since September 2026 there is a maintained pip wheel (`pywebrtc-audio` 0.2.0) that exposes AEC3 directly to Python. PipeWire also ships it as its default echo-cancel backend, so Voxa can use it without writing any DSP.

### Cited Findings
- **Refined vs. coarse filters.** The echo remover picks between the outputs of the two linear filters. It prefers the refined filter by default, but switches to the coarse output when the refined filter looks diverged or performs worse, using power thresholds with a margin of about 10%. It crossfades between the two outputs so the switch makes no audible jump. — [WebRTC echo_remover.cc](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/echo_remover.cc)
- **Divergence fix (2020).** After the refined coefficients are copied into the coarse filter, at least 100 ms must pass before the refined filter may speed up its adaptation because of good coarse-filter performance. A kill switch can turn this off. — [WebRTC commit 992a96f](https://webrtc.googlesource.com/src/+/992a96f68e4cadc69ff85ffe85500010e761e2bd)
- **Default config values** (`api/audio/echo_canceller3_config.h`) — [EchoCanceller3Config](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h):
  - **Filters:**
    - `filter.refined.length_blocks = 13`, `leakage_converged = 0.00005`, `leakage_diverged = 0.05`, `error_floor = 0.001`, `error_ceil = 2.0`.
    - `filter.coarse.length_blocks = 13`, `rate = 0.7`.
    - Initial-phase variants: `refined_initial.length_blocks = 12` (leakage 0.005/0.5) and `coarse_initial.rate = 0.9`.
  - **Delay:** `delay.down_sampling_factor = 4`, `num_filters = 5`, `delay_headroom_samples = 32`, `default_delay = 5`, `hysteresis_limit_blocks = 1`, `delay_estimate_smoothing = 0.7`, `delay_candidate_detection_threshold = 0.2`, `delay_selection_thresholds` initial 5 / converged 20, `use_external_delay_estimator = false`, `detect_pre_echo = true`.
  - **ERLE estimator:** `erle.min = 1.0`, `max_l = 4.0`, `max_h = 1.5`, `onset_detection = true`.
  - **Echo path strength:** `ep_strength.default_len = 0.83`, `echo_can_saturate = true`, `bounded_erl = false`.
  - **Echo removal control:** `echo_removal_control.has_clock_drift = false` (default), `linear_and_stable_echo_path = false`.
  - **Suppressor masks:**
    - `normal_tuning.mask_lf` (enr_transparent 0.3, enr_suppress 0.4, emr_transparent 0.3); `mask_hf` (0.07, 0.1, 0.3).
    - `nearend_tuning.mask_lf` (1.09, 1.1, 0.3); `mask_hf` (0.1, 0.3, 0.3).
    - `max_inc_factor = 2.0`, `max_dec_factor_lf = 0.25`.
    - The suppressor switches between a "normal" (echo-dominated) tuning and a more transparent "nearend" tuning used during double talk.
  - **Comfort noise:** `comfort_noise.noise_floor_dbfs = -96.03`.
  - **Echo audibility:** `low_render_limit = 4*64`, `normal_render_limit = 64`, `floor_power = 2*64`.
- **Clock drift.** Clock drift is "generally difficult to handle for echo cancellers". AEC3 tries to detect drift and, when it finds it, applies the same limiting as when `has_clock_drift` is set. Mozilla filed a bug to set that flag explicitly whenever drift is expected, for example when the input and output devices differ. — [Mozilla bug 1887774](https://bugzilla.mozilla.org/show_bug.cgi?id=1887774)
- **`pywebrtc-audio` 0.2.0** (released 2026-09-03, "4 - Beta", Apache-2.0, maintained by strands-labs / awsarron):
  - Wheels for Linux x86_64/aarch64, macOS and Windows, Python 3.10–3.14.
  - Exposes an `EchoCanceller` (AEC3) class and an `AudioProcessor` pipeline that runs HPF → AEC → NS → AGC2 (and includes an RNN VAD).
  - `EchoCanceller` supports 16, 32 and 48 kHz and works internally in 10 ms frames.
  - `stream_delay_ms` is a delay hint. Leaving it at 0 lets the AEC3 delay estimator find the delay itself.
  - The far-end reference is passed as `far`, with the same length as `near`.
  - — [PyPI pywebrtc-audio](https://pypi.org/project/pywebrtc-audio/); [Strands docs](https://strandsagents.com/docs/labs/pywebrtc-audio/)
- **Older bindings.** `webrtc-audio-processing` (xiongyihui, a SWIG build that needs a compiler) shows VAD/NS examples. The results did not confirm that it exposes AEC3. — [PyPI webrtc-audio-processing](https://pypi.org/project/webrtc-audio-processing); [GitHub mirror listing](https://relatedrepos.com/gh/xiongyihui/python-webrtc-audio-processing)
- **`aec-audio-processing`** also exists on PyPI, but its list of WebRTC components was not confirmed. — [PyPI aec-audio-processing](https://pypi.org/project/aec-audio-processing/)
- **PipeWire.** `libpipewire-module-echo-cancel` creates virtual `echo-cancel-capture` (source) and `echo-cancel-playback` (sink) nodes. Its default backend is `aec/libspa-aec-webrtc`. Debian ships PipeWire 1.4.2 (trixie) through 1.6.8 (unstable). — [PipeWire docs](https://docs.pipewire.org/page_module_echo_cancel.html); [Debian manpage](https://manpages.debian.org/testing/libpipewire-0.3-modules/libpipewire-module-echo-cancel.7.en.html)
- **PipeWire friction.** Loading the module via `pactl` did nothing for one user. Echo cancellation only worked with a hand-written config that routed playback through the echo-cancel sink. — [PipeWire issue 3412](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3412)
- **Explainer article (secondary source).** The residual suppressor estimates leftover echo power per band and applies band-dependent gain, calibrated by the ERLE of the linear filter. — [Fora Soft explainer, 2026-06](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/echo-cancellation-aec-ai-hybrid-webrtc)

### Inferences
- **Block size and filter length.** The `4*64` and `2*64` expressions in the config suggest a 64-sample block, which is 4 ms at 16 kHz. If so, 13 blocks × 64 samples = 832 samples, about 52 ms of echo-tail coverage at 16 kHz. A laptop's direct path plus early reflections fits in that. A long room tail does not, and it is left to the suppressor. This is an inference from the config and was not confirmed in the docs.
- **Why AEC3 suits Voxa.** The dual filter with crossfade and leakage switching on divergence, together with a residual suppressor that falls back to heavy suppression when ERLE is low, is exactly what an assistant that "must never hear itself" needs. When the linear filter fails (nonlinear speaker, drift, Bluetooth), AEC3 suppresses more and lets less echo through.
- **Use `has_clock_drift=True`.** For Voxa's separate capture and playback devices, setting `echo_removal_control.has_clock_drift = true` is the documented knob. Whether `pywebrtc-audio` exposes the full `EchoCanceller3Config` was not verified.
- **Two ways to integrate:**
  - **(a) PipeWire echo-cancel module.** It uses a system-level AEC3 (via webrtc-audio-processing), which removes Voxa's own reference-capture timing problem because PipeWire has both streams on one graph clock. Voxa must then route TTS through the echo-cancel sink, or set it as the default sink.
  - **(b) `pywebrtc-audio` in-process.** Voxa feeds its own TTS buffer as `far`. This is more exact than the monitor-loopback reference, because it carries no reference-capture jitter, but Voxa must timestamp playback.

### Gaps
- No published ERLE number for AEC3 on standard data was found. AEC3 has no "ERLE dB" spec in Google's docs, and no peer-reviewed AEC3 benchmark was found this session.
- I did not confirm from source which WebRTC revision `pywebrtc-audio` wraps, whether its config API is exposed, or which webrtc-audio-processing version (1.x/2.x) PipeWire links in current distros.
- No primary design document was retrieved for the AEC3 render delay controller or matched-filter delay estimator. Only the config fields above are sourced.

## 2. SpeexDSP MDF echo canceller and preprocessor residual suppression vs AEC3

### Takeaway
SpeexDSP MDF is a single multidelay block frequency-domain filter. Pairing it with the preprocessor's residual echo suppression is light, simple and easy to call from C or Python. Practitioners consistently report that it degrades badly when capture and playback clocks drift, while AEC3 copes. On a single device with one clock it can be acceptable.

### Cited Findings
- **Drift handling.** An Ubuntu/PulseAudio bug comment reports the difference "especially pronounced when the input and output sound devices do not share the same clock source". It says Speex is known for not resyncing smoothly when clocks drift, as WebRTC does. — [Launchpad bug 1261666 comment 5](https://bugs.launchpad.net/ubuntu/+source/pulseaudio/+bug/1261666/comments/5)
- **Early Speex developer advice (2005).** With two unsynchronized soundcards, use clock-synced pro cards "or forget about echo cancellation". — [speex-dev 2005](https://lists.xiph.org/pipermail/speex-dev/2005-November/003900.html); [speex-dev 2011 drift thread](https://lists.xiph.org/pipermail/speex-dev/2011-April/008114.html)
- **Mixed practitioner reports.**
  - One developer says SpeexDSP "never produced good results" on hardware where WebRTC did. — [Reddit r/dotnet](https://redlib.hyperreal.coffee/r/dotnet/comments/1k5u4w0/echo_and_noise_cancellation/mp213cs/?context=3)
  - A Raspberry Pi user found Speex AEC "quite good" on a single USB card that shares one clock for capture and playback. — [Mycroft forum](https://community.mycroft.ai/t/fighting-with-pulseaudio-aec/9247)
- **PulseAudio discussion (2020)** compared the speex and webrtc backends of `module-echo-cancel`. — [pulseaudio-discuss 2020-May](https://lists.freedesktop.org/archives/pulseaudio-discuss/2020-May/031749.html)

### Inferences
- **No reason to switch.** Voxa's existing NLMS plus per-band mask is roughly in the same class as Speex MDF plus its preprocessor. Moving to Speex would not fix drift or nonlinearity. AEC3 is the better linear-plus-suppressor choice.
- **Where Speex still fits.** It remains useful as a tiny fallback, or as a second, independent linear stage that feeds a neural postfilter, where the postfilter does the heavy lifting.

### Gaps
- No controlled head-to-head ERLE or AECMOS numbers for Speex MDF vs AEC3 were found. The evidence is anecdotal.
- MDF internals (multidelay partitions, adaptive step from the leakage estimate, two-path "foreground/background" filter in newer SpeexDSP versions) were not verified from Xiph source this session.

## 3. Nonlinear echo from cheap speakers and clipping

### Takeaway
I found no fresh primary sources on Volterra or Hammerstein AEC in this session. The evidence collected points to the industry pattern: keep the linear filter, and remove the nonlinear residue with a suppressor, either AEC3's ERLE-driven suppressor or a neural postfilter, rather than modelling the nonlinearity explicitly.

### Cited Findings
- **AEC3 saturation handling.** AEC3 contains explicit handling for echo saturation (`ep_strength.echo_can_saturate = true`) and a conservative tail response (`use_conservative_tail_frequency_response = true`). Both bias the suppressor toward removing echo the linear filter cannot model. — [EchoCanceller3Config](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h)
- **Hybrid linear AEC plus Bark-scale neural postfilter** (Seidel, Mowlaee, Fingscheidt, arXiv 2404.11621, 2024). The postfilter removes residual echo, including echo the linear stage leaves, together with noise. It matches DeepVQE-S on the ICASSP 2023 blind test set at about 10% of its compute. — [arXiv 2404.11621](https://arxiv.org/abs/2404.11621)

### Inferences
- **Explicit models are unlikely to pay off.** From background knowledge (not verified this session): Volterra filters (2nd/3rd order) and Hammerstein models (memoryless polynomial or sigmoid before a linear FIR) are expensive and fragile in practice. For Voxa, the practical defences are:
  - **(a) Keep TTS output below clipping.** Limit TTS to roughly -3 to -6 dBFS and avoid OS volume settings that drive laptop speakers into compression. This prevents the nonlinearity at its source.
  - **(b) Reference-driven residual suppression.** Rely on a mask-based residual suppressor driven by the reference, as Voxa already does and as AEC3 and neural postfilters do.
- **Known reference avoids misclassification.** Voxa knows its own TTS signal exactly, so a residual-echo mask can be aggressive while TTS plays. Near-end speech is the only remaining thing to protect.

### Gaps
- No 2018–2026 sources with numbers were retrieved for Volterra or Hammerstein AEC, or for neural nonlinear echo modelling such as DNN-based nonlinear reference generation.

## 4. Delay estimation and clock-drift / sample-rate-offset (SRO) compensation

### Takeaway
Uncompensated SRO prevents AEC filters from converging. Even 100 ppm amounts to 360 ms of drift per hour. The robust fix is to estimate SRO, using GCC- or coherence-drift-based estimators, and resample the reference so it stays on the capture clock. Recent work shows this recovers no-SRO performance up to about ±75 ppm. AEC3 instead detects drift and becomes more conservative.

### Cited Findings
- **Size of the drift.** A 100 ppm difference makes two recordings diverge by 360 ms per hour. One patent says AEC works best when clocks agree within a few ppm. — [Stack Overflow via kiwix](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/15727333/how-to-calculate-sound-card-clock-drift); [US7680285B2](https://patents.google.com/patent/US7680285)
- **Separate boards drift.** Separate audio boards have independent clocks. Mismatch can occur even with a shared clock and approximate dividers. — [US8515086](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8515086)
- **USB clocking.** Only synchronous USB audio devices lock their audio clock to the host USB clock. Asynchronous and adaptive devices run on their own or adapt to the stream. — [ASIO4ALL device aggregation](https://asio4all.org/device-aggregation/)
- **Compensation in patents.** One patented method cross-correlates an echo estimate with near-end audio and adjusts a sample-rate-conversion factor. Another argues that sub-sample accuracy is needed for good AEC. — [US7680285B2](https://patents.google.com/patent/US7680285); [US8320554](https://patentimages.storage.googleapis.com/ca/ec/8f/0f57def85ce6ad/US8320554.pdf)
- **"Sample Rate Offset Compensated AEC"** (arXiv 2507.05399, 2025):
  - **Setup:**
    - It simulates SRO of ±150 ppm.
    - It estimates SRO with DWACD (a phase function across consecutive complex coherence functions, plus a GCC peak search refined by golden-section search). The estimator uses two previous 8192-sample segments (0.512 s), so it adds no delay.
    - It resamples the far-end reference and runs a partitioned-block frequency-domain Kalman filter AEC.
  - **Results:**
    - Uncompensated performance drops as SRO grows.
    - Compensated performance matches the oracle and no-SRO systems for uncorrelated playback, especially below about ±75 ppm.
    - Estimating SRO from an independent single-channel AEC error signal converges faster. It falls short in double talk.
  - — [arXiv 2507.05399](https://arxiv.org/html/2507.05399v1)

### Inferences
- **How Voxa gets its reference matters most:**
  - **(a) PipeWire monitor of the same sink, or PipeWire echo-cancel.** PipeWire resamples all nodes onto its graph clock, so drift between the mic and speaker hardware shows up as slow delay creep in the monitor relative to the mic. That creep is what Voxa's delay tracker must follow.
  - **(b) Bluetooth.** Bluetooth adds large (100–250 ms+) and variable latency. This is background knowledge, not sourced here. It is the case where AEC3's delay estimator (5 matched filters on 4x-decimated signals, per the config) and a long delay search window matter most.
- **Practical recipe:**
  - Track delay with GCC-PHAT, or reuse AEC3's estimator, on each TTS utterance.
  - Since Voxa's TTS utterances are short (seconds), re-estimate delay at every utterance onset. Drift within one utterance (e.g. 100 ppm × 5 s = 0.5 ms = 8 samples at 16 kHz) is small but not zero, and a sub-sample-accurate resampler, as in the patents, can absorb it.

### Gaps
- I found no measured ppm distribution for laptop onboard codecs vs USB mics. Typical crystal tolerance (tens of ppm) is background knowledge, not sourced here.

## 5. Double-talk detection and avoiding filter divergence

### Takeaway
Modern systems mostly avoid hard double-talk detectors. Instead they combine:
- **Dual (shadow/main) filters with divergence detection and copy-back,** as in AEC3's refined/coarse filters with leakage switching and the 100 ms rule.
- **Kalman or neural step-size control,** as in NKF, where the gain is computed per bin from estimated noise covariances.
- **A suppressor with separate echo-dominated and near-end tunings.**

### Cited Findings
- **AEC3 safeguards.** AEC3 crossfades to the coarse filter when the refined filter diverges (about 10% power margin). It uses leakage of 0.00005 when converged and 0.05 when diverged, and forbids speed-ups for 100 ms after a copy. — [echo_remover.cc](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/echo_remover.cc); [commit 992a96f](https://webrtc.googlesource.com/src/+/992a96f68e4cadc69ff85ffe85500010e761e2bd); [config](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h)
- **AEC3 near-end tuning.** The suppressor has a separate `nearend_tuning` with much higher transparency thresholds (mask_lf enr_transparent 1.09 vs 0.3), which preserves the user's speech during double talk. — [config](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h)
- **NKF (Tencent, arXiv 2207.11388).** A neural network implicitly models the state- and observation-noise covariances and outputs the Kalman gain. That gives robust adaptation without an explicit DTD, plus faster convergence and re-convergence than model-based methods. — [arXiv 2207.11388](https://arxiv.org/abs/2207.11388v2)

### Inferences
- **Geigel and NCC detectors are older techniques.** From background knowledge (not sourced here): Geigel compares |mic| against max |far| over a window with a threshold of about 0.5 (-6 dB), and NCC / cross-correlation detectors compare normalized correlation. Both are simple and break with nonlinear or loud echo paths. A two-filter scheme plus a suppressor is more robust.
- **Freeze adaptation on the user's voice.** For Voxa, a cheap and reliable extra cue is VAD on the AEC output while TTS plays. If Silero fires on the residual and the residual is uncorrelated with the reference, freeze adaptation (and treat it as a barge-in).

### Gaps
- No 2018–2026 sources with numbers comparing Geigel, NCC and other DTDs were retrieved.

## 6. Neural and hybrid AEC: challenge systems, open models, CPU cost

### Takeaway
Since 2021, the top challenge systems have been hybrids (a linear AEC plus a neural postfilter) or small end-to-end models of roughly 0.6–10M parameters with CPU RTF around 0.2–0.35. DeepVQE (Microsoft, 7.5M params) matches the 2023 winner, and its small version DeepVQE-S (0.59M params, RTF 0.014) is close. The most practical open option for offline use is LocalVQE (2026): a pruned DeepVQE derivative with 1.3M params and Apache-2.0 GGUF weights that runs in C++/GGML at about 3.4 ms per 16 ms hop on one CPU thread. NKF (5.3K params) is a tiny open neural linear stage. DTLN-aec (MIT, TF-lite) is the older open baseline.

### Cited Findings
- **ICASSP 2023 AEC Challenge (4th challenge):**
  - **Rules and data:**
    - Latency limit 20 ms (algorithmic plus buffering).
    - RTF must be ≤ 0.5 on an Intel i5 quad-core at 2.4 GHz, single thread. No future frames.
    - It added a personalized track and a full-band AECMOS.
    - Real recordings come from more than 10,000 devices.
  - **Winners (score is the mean of MOS and WAcc):**

    | Place | System | Hybrid | Params | RTF | Score |
    |---|---|---|---|---|---|
    | 1 | Progressive neural network | yes | 1.5M | 0.28 | 0.854 |
    | 2 | Low-latency deep hierarchical fusion network | no | 32.6M | 0.19 | 0.831 |
    | 3 | Two-step band-split network | yes | 9.6M | 0.35 | 0.829 |
    | 4 | Multi-task sub-band network | yes | 3.8M | 0.20 | 0.823 |
    | 5 | TaylorAECNet | no | 19.2M | 0.22 | 0.803 |

    The winner's personalized-track entry scored 0.852 (1.6M params). Personalization gained only 0.002.
  - **Baseline:** a GRU mask network (2 × 322 GRU), about 1.3M params, 20 ms frames, 10 ms hop.
  - **Published in** IEEE OJSP vol. 5, pp. 675–685.
  - — [arXiv 2309.12553](https://arxiv.org/html/2309.12553v1); [IEEE SPS page](https://signalprocessingsociety.org/publications-resources/data-challenges/icassp-2023-acoustic-echo-cancellation-challenge-icassp-2023)
- **DeepVQE (Indenbom et al., Microsoft, Interspeech 2023, arXiv 2306.03177):**
  - **Architecture:** a residual CNN autoencoder with a GRU bottleneck, plus a cross-attention alignment block that replaces DSP delay alignment. It does joint AEC, noise suppression and dereverberation.
  - **Size and speed:**
    - DeepVQE: 7.5M params, 3.66 ms per frame on an i7-11370H.
    - DeepVQE-S: 0.59M params, 0.14 ms per frame, RTF 0.014.
  - **ICASSP 2023 MOS results:**

    | System | Far-end single-talk echo | Double-talk echo | Double-talk other |
    |---|---|---|---|
    | Baseline | 4.53 | 3.47 | 3.88 |
    | Challenge winner | 4.70 | 4.31 | 3.99 |
    | DeepVQE-S | 4.66 | 4.00 | 4.04 |
    | DeepVQE | 4.69 | 4.29 | 4.15 |

    DeepVQE's final score of 0.854 ties the winner.
  - **Alignment ablation:** the alignment block gave higher ERLE and echo MOS than DSP alignment, and lower WER (31.79 vs 36.27).
  - **Deployment:** tested in Microsoft Teams.
  - **Openness:** no released code or weights were stated in the paper.
  - — [ar5iv 2306.03177](https://ar5iv.labs.arxiv.org/html/2306.03177)
- **LocalVQE (LocalAI, Apache-2.0, v1.1):**
  - **Origin and size:** DeepVQE re-implemented in GGML at 7.5M params, then pruned to 1.29M params (about 5 MB F32).
  - **Format:** 16 kHz, causal, 16 ms algorithmic latency (256-sample hop).
  - **Speed** (Ryzen 9 7900, p50 per hop):

    | Threads | p50 per hop |
    |---|---|
    | 1 | 3.40 ms (≈ RTF 0.21) |
    | 2 | 2.07 ms |
    | 4 | 1.32 ms |

  - **ICASSP 2022 blind test (800 real clips):**

    | Scenario | AECMOS echo | AECMOS degradation | Blind ERLE |
    |---|---|---|---|
    | Double talk | 4.70 | 2.35 | 8.4 dB |
    | Far-end single talk | 2.98 | 4.91 | 44.7 dB |
    | Far-end single talk with movement | 3.40 | — | 45.0 dB |
    | Near-end single talk | 4.99 | 4.05 | — |

  - **Runtime:** C++ with a C shared library (`liblocalvqe.so`, callable from Python via ctypes) and a CLI that takes mic.wav and ref.wav.
  - **Not yet released:** Q4_K and Q8_0 quantized weights.
  - — [Hugging Face LocalVQE](https://huggingface.co/Trondtempia/LocalVQE)
  - **Note on the numbers:** the far-end single-talk AECMOS echo score (2.98) is oddly low next to an ERLE of 44.7 dB. The source does not explain this, so treat it with caution.
- **DTLN-aec (Westhausen & Meyer, ICASSP 2021 challenge):**
  - 32 ms frames, 8 ms shift, FFT 512.
  - Pretrained sizes of about 1.8M, 3.9M and 10.3–10.4M params. The 512-unit model was the challenge submission.
  - TF-lite pretrained models, MIT license, repository `breizhn/DTLN-aec`.
  - The mirrors conflict on its ranking ("third" vs "top five").
  - — [arXiv 2010.14337](https://arxiv.org/abs/2010.14337); [Gitee mirror](https://gitee.com/rsmeng/DTLN-aec)
- **NKF (Tencent GVoice, ICASSP 2023):**
  - 5.3K params, RTF 0.09. Baselines had 2.4M and 1M params.
  - Neural Kalman gain. Code at `github.com/fjiang9/NKF-AEC`.
  - — [arXiv 2207.11388](https://arxiv.org/abs/2207.11388v2)
- **Seidel, Mowlaee, Fingscheidt (2024):** a hybrid linear AEC plus Bark-scale neural postfilter. It is comparable to DeepVQE-S at about 10% of its complexity. — [arXiv 2404.11621](https://arxiv.org/abs/2404.11621)
- **AECMOS is open.** The AECMOS model is available in Microsoft's AEC-Challenge repo, so Voxa can score itself. — [arXiv 2010.14337 / AEC-Challenge reference](https://arxiv.org/abs/2010.14337)

### Inferences
- **Most usable open option:** LocalVQE. It is offline, permissively licensed, about 0.2 RTF on one core, and fed with mic plus reference. It also includes its own alignment, inherited from DeepVQE's cross-attention.
- **Proven hybrid template:** NKF or AEC3 for the linear stage, then a small neural postfilter. This is the 2023 winner's hybrid style and the Seidel et al. design.
- **Do not trust single-talk AECMOS alone.** For "never hear herself", the far-end single-talk numbers (echo AECMOS, ERLE) are the ones that matter. LocalVQE's 44.7 dB blind ERLE is strong, but its AECMOS echo of 2.98 in far-end single talk suggests audible residue on some clips. Voxa should measure its own residual on its own TTS through laptop speakers.

### Gaps
- I did not retrieve the 2021 (ICASSP, Interspeech) and 2022 challenge winner details, or Align-CRUSE numbers, this session.
- DTLN-aec's AECMOS scores were not verified.
- The NKF repo's license and ERLE figures were not checked.
- No 2025–2026 challenge (there may have been none after 2023) was found.

## 7. Recommendation for Voxa: hear none of herself and all of the user on a single-mic laptop, CPU only

### Takeaway
Use a layered stack:
1. **Exact reference.** Feed Voxa's own TTS samples, timestamped, rather than only the sink monitor.
2. **AEC3 as the linear stage plus residual suppressor,** via `pywebrtc-audio` in-process or PipeWire's `libspa-aec-webrtc` module, with drift mode on.
3. **Optional neural postfilter for the leftover nonlinear echo:** LocalVQE (1.3M params, about 3.4 ms per 16 ms hop on one core, Apache-2.0).
4. **Self-voice gating at the ASR/VAD level** during TTS, as a last line of defence.

### Cited Findings
- AEC3 combines dual-filter divergence protection, delay estimation, a drift flag, ERLE-driven suppression with separate near-end tuning, and comfort noise. — [config](https://webrtc.googlesource.com/src/+/main/api/audio/echo_canceller3_config.h); [Mozilla bug 1887774](https://bugzilla.mozilla.org/show_bug.cgi?id=1887774)
- `pywebrtc-audio` gives maintained Python wheels for AEC3 (Sept 2026). — [PyPI](https://pypi.org/project/pywebrtc-audio/)
- PipeWire's echo-cancel module defaults to a WebRTC backend. — [PipeWire docs](https://docs.pipewire.org/page_module_echo_cancel.html)
- LocalVQE has open weights and a C library, with measured single-thread CPU latency and ICASSP 2022 AECMOS/ERLE. — [HF LocalVQE](https://huggingface.co/Trondtempia/LocalVQE)
- Hybrid linear plus neural-postfilter systems match small end-to-end models at much lower compute. — [arXiv 2404.11621](https://arxiv.org/abs/2404.11621); [arXiv 2309.12553](https://arxiv.org/html/2309.12553v1)
- SRO compensation by resampling the reference recovers no-SRO AEC performance below about ±75 ppm. — [arXiv 2507.05399](https://arxiv.org/html/2507.05399v1)

### Inferences
- **Concrete order of work for Voxa:**
  1. Swap Voxa's custom NLMS for AEC3 via `pywebrtc-audio`. Keep the existing ERLE metric so the two can be compared on Voxa's recordings.
  2. Feed AEC3 the exact TTS buffer, or keep the PipeWire monitor if timestamping is hard. Leave `stream_delay_ms` at 0, or give a measured hint for Bluetooth.
  3. Run LocalVQE (or Voxa's existing per-band mask) on the AEC3 output during and just after TTS, about 300 ms for the room tail.
  4. Keep a policy gate: during TTS, accept barge-in only when VAD fires on the post-AEC signal and that signal's energy exceeds the predicted residual echo by a margin. Voxa already has this predictor.
- **Bluetooth:** expect worse results (codec nonlinearity, variable delay). Raise suppression and the gate threshold whenever the output sink is Bluetooth.
- **Measurement:** score with AECMOS from the Microsoft AEC-Challenge repo, plus ASR false-trigger rate on TTS-only recordings. The goal "never hears herself" is best measured as zero ASR words transcribed from TTS-only playback.

### Gaps
- No direct benchmark exists of AEC3 + LocalVQE on laptop speakers. Voxa would have to measure it.
- I did not verify whether `pywebrtc-audio` exposes `has_clock_drift` or the full AEC3 config.
