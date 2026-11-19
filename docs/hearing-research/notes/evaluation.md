# Evaluating Voxa's hearing: echo cancellation, double talk, barge-in, missed words, self-interruption, and an offline test bench

Scope: metrics, standards, datasets and methodology for showing that a single-mic Linux voice assistant (faster-whisper ASR, Silero VAD, local TTS) hears every word the user says, none of its own voice, and never interrupts itself. The notes end in a proposed bench design with numeric gates a local coding model can run.

Research note: the Microsoft AEC Challenge papers (arXiv 2309.12553, 2110.03010) and ETSI TS 103 739 were read in full text. The other sources were read through search snippets or abstract pages only, and are marked as such where it matters. Numbers labelled "proposed" under Inferences are my design choices. No source supplies them.

## Q1. AEC metrics and standards (ERLE, AECMOS, PESQ/DNSMOS, ITU-T/ETSI/TIA double-talk requirements, AEC Challenge scenarios)

### Takeaway
ERLE is the only simple, exact, intrusive AEC metric. It is valid only for far-end single talk in a quiet room, and it correlates weakly with what people hear (PCC 0.31). Microsoft's AEC Challenge splits evaluation into three scenarios: far-end single talk (FST), near-end single talk (NST) and double talk (DT). It scores them with AECMOS (echo DMOS plus "other degradation" DMOS) and adds ASR word accuracy. Telephony standards (ETSI/ITU-T P.340 categories) give hard dB targets for double talk: attenuation range ≤3 dB and echo loss ≥27 dB for "full duplex". Voxa can borrow the scenario split and the dB-style gates, but its end metric must be ASR-level, because Voxa's "listener" is Whisper, not a human.

### Cited Findings
- ERLE definition: `ERLE = 10·log10( E[y²(n)] / E[e²(n)] )`, where y(n) is the microphone signal and e(n) is the residual echo after cancellation. "ERLE is only appropriate when measured in a quiet room with no background noise and only for single talk scenarios (not double talk)". Under those conditions the processed mic signal can serve as the estimate of e(n). — [ICASSP 2023 AEC Challenge, Cutler et al., arXiv 2309.12553](https://arxiv.org/pdf/2309.12553)
- On the challenge's single-talk echo data, ERLE correlates with P.808 subjective scores at PCC 0.31 / SRCC 0.23, and PESQ at PCC 0.67 / SRCC 0.57. Because of this, the challenge does not use ERLE or PESQ on real recordings. — [arXiv 2309.12553, Table 1](https://arxiv.org/pdf/2309.12553)
- Standards landscape as the challenge paper summarizes it:
  - IEEE 1329 defines terminal coupling loss for single talk (TCLwst) and double talk (TCLwdt), both measured in anechoic chambers.
  - TIA-920 uses many of these metrics and defines required criteria.
  - ITU-T G.122 covers AEC stability, and G.131 relates Talker Echo Loudness Rating to one-way delay.
  - G.168 gives a comprehensive set of AEC metrics and criteria.
  - The paper's criticism: "it is not clear how to combine these dozens of metrics into a single metric, or how well these metrics correlate to subjective quality."
  - [arXiv 2309.12553 §II](https://arxiv.org/pdf/2309.12553)
- AEC Challenge real test set, recording scenarios (from over 10,000 real devices/rooms/speakers):
  1. FST with no echo path change
  2. FST with echo path change
  3. NST
  4. DT with no echo path change
  5. DT with echo path change
  6. a sweep for RT60 estimation

  Echo path changes were made by telling users to move the device or themselves. — [arXiv 2309.12553 §IV](https://arxiv.org/pdf/2309.12553)
- Convergence handling:
  - For FST, the far-end clip is played twice, so the AEC is judged on both an unconverged and a converged segment. Scoring uses the second half of each FST clip, "to avoid initial degradations from initialization, convergence periods, and initial delay estimation".
  - For DT, only the final third of the clip is scored.
  - The blind test sets are 800 real recordings of 30-45 s each. They are enriched with hard cases: long or varying loopback-to-mic delay, strong speaker or mic distortion, stationary and non-stationary near-end noise, device-side DSP (built-in AEC or noise reduction), CPU glitches, and far-end gain changes.
  - [arXiv 2309.12553 §V, §VII](https://arxiv.org/pdf/2309.12553)
- Subjective ratings use two questions on a 1-5 degradation scale: "How would you judge the degradation from the echo?" and "How would you judge other degradations (noise, missing audio, distortions, cut-outs)?" "Other" is not scored for FST. — [arXiv 2309.12553 §VII](https://arxiv.org/pdf/2309.12553)
- Challenge score (equation 2): `Score = [ (FE−1)/4 + (NE_SIG−1)/4 + (NE_BAK−1)/4 + (DT_echo−1)/4 + (DT_other−1)/4 + WAcc ] / 6`. FE is FST echo DMOS, NE_SIG/NE_BAK are P.835 SIG/BAK for NST, and WAcc is word accuracy. All six terms are weighted equally on a 0-1 scale. — [arXiv 2309.12553 §IX](https://arxiv.org/pdf/2309.12553)
- State of the art:
  - The 2023 top model (MS-1) scored FST echo MOS 4.65, DT echo MOS 4.68 and DT other MOS 4.26, at 20 ms algorithmic plus buffering latency.
  - Remaining headroom to "excellent": FST echo 0.30, DT echo 0.26, DT other 0.64, NST SIG 0.74, NST BAK 0.64, overall 0.56.
  - Double-talk "other" (near-end speech damage) and near-end quality are the weakest areas.
  - [arXiv 2309.12553 Tables 9-10](https://arxiv.org/pdf/2309.12553)
- AECMOS inputs and outputs:
  - Inputs are three signals (near-end mic, far-end/loopback, AEC output) plus an optional one-hot scenario marker (NST/FST/DT).
  - It was trained on human ratings (training mix: 45.6% NST, 26.7% FST, 27.7% DT).
  - Per-contestant correlation with humans: FST echo DMOS PCC 0.996 / SRCC 0.969, DT echo DMOS PCC 0.898, DT other DMOS PCC 0.927, NST MOS PCC 0.923.
  - The paper states that double talk is the hardest scenario for the model.
  - [AECMOS, Purin et al., ICASSP 2022, arXiv 2110.03010](https://arxiv.org/pdf/2110.03010)
- AECMOS against human ratings on the 2023 challenge entries: FST echo DMOS PCC 0.99, DT echo DMOS 0.99, DT other 0.96, NST SIG 0.85, NST BAK 0.93. — [arXiv 2309.12553 Table 5](https://arxiv.org/pdf/2309.12553)
- Per-clip far-end single-talk PCC: AECMOS 0.847, against roughly 0.54-0.71 for ERLE, PESQ and EQUEST. The PDF's text extraction scrambles the column alignment, so the mapping of each number to ERLE, PESQ or EQUEST is uncertain. — [arXiv 2110.03010 Table 2](https://arxiv.org/pdf/2110.03010)
- Tooling: `pip install speechmos`, then call `aecmos.run({'lpb':…, 'mic':…, 'enh':…}, sr, talk_type)`.
  - `talk_type` is `'st'` (far-end single talk), `'nst'` or `'dt'` and uses the 48 kHz scenario model. `None` selects the 16 kHz scenarioless model, which the docs say correlates about 2% worse.
  - It returns `echo_mos` and `deg_mos`.
  - `dnsmos.run(audio, 16000)` returns `ovrl_mos`, `sig_mos`, `bak_mos` and `p808_mos`.
  - The package is archived; the last release is 0.0.1.1 from July 2023, and it pins older librosa/onnxruntime.
  - [speechmos on PyPI](https://pypi.org/project/speechmos/)
- ETSI TS 103 739 (wideband wireless handset/headset, 2010) gives concrete numbers:
  - TCLw ≥ 55 dB, and ≥ 46 dB at maximum volume. A note says residual echo may be perceived even above 50 dB.
  - Double-talk categories follow ITU-T P.340, using attenuation range in the send direction (A_H,S,dt): Cat 1 "full duplex" ≤ 3 dB, 2a ≤ 6 dB, 2b ≤ 9 dB, 2c ≤ 12 dB, Cat 3 "no duplex" > 12 dB.
  - Cat 1 requires echo loss during double talk (from TELRdt) ≥ 27 dB.
  - The terminal's overall category is the lowest of the three sub-categories.
  - The test signals are P.501 composite source signals with overlapping near/far CSS bursts, and AECs are tested in a realistic room.
  - [ETSI TS 103 739 V1.1.2](https://www.etsi.org/deliver/etsi_TS/103700_103799/103739/01.01.02_60/ts_103739v010102p.pdf)
- An older 3GPP SA4 working document says MOS ≥ 4.0 in double talk needs TELR_DT ≥ 37 dB, with attenuation range ≤ 3 dB in both send and receive. This comes from the search snippet only. — [3GPP S4-000147](https://3gpp.org/FTP/tsg_sa/WG4_CODEC/TSGS4_10/Docs/PDF/s4-000147.pdf)
- ITU-T P.1100 (2019, narrowband in-car hands-free) has double-talk clauses: attenuation range in send and receive, detection of echo components during double talk, and attenuation of sent speech. This is from the table of contents only. — [ITU-T P.1100 TOC](https://www.itu.int/dms_pubrec/itu-t/rec/p/T-REC-P.1100-201901-S!!TOC-HTM-E.htm)
- HEAD acoustics compares double-talk measurement methods (abstract). — [Daga 2014](https://cdn.head-acoustics.com/fileadmin/data/global/Abstracts/Abstract-Daga-2014-Comparison-of-Double-Talk-Measurement-Methods.pdf)

### Inferences
- For Voxa (one mic, its own loudspeaker, and an AEC such as WebRTC AEC3, speexdsp or PipeWire's echo-cancel module), the AEC metrics fall into three tiers:
  - Diagnostic: ERLE on FST segments. Synthetic data can also compute true ERLE, because the bench knows the echo-only signal d(n). Then `ERLE_true = 10·log10(Σd² / Σ(ŝ−s)²)` over far-end-only frames, where ŝ is the AEC output and s is the near-end component (zero in FST).
  - Perceptual sanity check: AECMOS echo_mos and deg_mos per scenario.
  - Decisive: ASR-level metrics (Q2).
- The telephony categories translate to proposed Voxa gates. Near-end attenuation during double talk (AEC output level of user speech, double talk compared with near-end only) ≤ 3 dB matches P.340 Category 1 "full duplex". Residual echo attenuation ≥ 27 dB matches the Cat 1 echo loss.
- The FST/NST/DT split and the "score only the converged part" rule should be copied directly. Voxa should also score the unconverged first seconds separately, because each new TTS utterance after silence is an "initial convergence" event.

### Gaps
- Full texts of ITU-T P.1110, P.1130 and P.340 annex values, ETSI ES 202 396 and TIA-920 were not retrieved, as they are paywalled or PDFs were not fetched. Category 2a-3 echo-loss values in TS 103 739 Table 6.8.3.1 were only partly extracted (Cat 1 ≥ 27 dB is certain).
- PESQ and POLQA (ITU-T P.862/P.863) are intrusive and licensed. I found no source validating them for AEC output on ASR tasks.
- The ICASSP 2022 paper defines how WAcc was computed (which ASR, which clips). It was not retrieved.

## Q2. ASR-level metrics (WER in double talk, word recall of user speech, echo-leak false inserts, first-word truncation)

### Takeaway
No standard defines "echo-leak insertion rate" or "first-word truncation". The AEC Challenge only adds a WAcc term. Endpointing papers from Google measure premature cut-offs as deletion errors. Voxa should define these metrics itself, with exact formulas over a jiwer word alignment, and run them on clips whose user transcript and TTS transcript are both known.

### Cited Findings
- The AEC Challenge added a word accuracy (WAcc) term, equal-weighted in the final score, because AEC output is consumed by ASR as well as by humans. Remaining WAcc headroom was listed as one of the main open areas. — [arXiv 2309.12553](https://arxiv.org/pdf/2309.12553)
- Google's RNN-T endpointer work treats premature end-of-query decisions as a source of deletion errors. LAS rescoring plus MWER training gave an 18.7% relative WER reduction and cut EP50/EP90 by 40 ms and 160 ms on Voice Search. This is from the abstract/snippet. — [arXiv 2004.11544](https://arxiv.org/pdf/2004.11544.pdf)

### Inferences
Proposed definitions. Compute them with `jiwer.process_words(ref, hyp)` alignments after normalisation: lower-case, strip punctuation, number-to-word.
- User WER in double talk: `WER_DT = (S+D+I)/N_user`, scored only on segments where the TTS was playing.
- Word recall of user speech: `R_user = (N_user − S − D)/N_user`, which equals hits/N_user. This is the "heard everything" metric.
- Echo-leak insertion rate (false inserts of the assistant's own words). Align the hypothesis against both the user reference and the TTS script. An inserted word counts as an echo leak if it also matches the TTS text inside a ±1.5 s window of when it was played.
  - `ELR = echo-leak words / TTS words played`
  - Also report leak events per hour of TTS playback.
  - With an empty user reference (pure FST), any non-empty transcript is a leak. Then `FST_transcript_rate = clips with ≥1 word / clips`. That number is also Whisper's "hallucination on residual echo" rate.
- First-word truncation rate: `FWT = utterances whose first reference word is deleted or substituted / utterances`. Report it separately for barge-in onsets (user starts during TTS) and for cold starts (user starts in silence). It catches VAD onset lag, pre-roll buffers that are too short, and AEC double-talk suppression at the onset.
- Last-word truncation / premature cut-off: `PCO = utterances where endpoint time < reference last-word end time, or the last reference word is deleted / utterances`.
- Hallucination check for Whisper: run the ASR on residual-only or noise-only clips. Whisper is known to hallucinate on non-speech, but I found no fetched source quantifying this, so treat it as a test to run, not a cited fact.

### Gaps
- I found no published benchmark reporting echo-leak insertion or first-word truncation rates for assistants. Typical industry values are unknown.

## Q3. Barge-in, wake-word and endpointing metrics, and how industry reports them

### Takeaway
Industry reports wake-word performance as false accepts per hour (FA/h), with false-reject rate at a fixed FA/h operating point. Endpointing is reported as EP50/EP90 latency plus WER or cut-off rate. Barge-in has no standard. The one quantitative academic source (FireRedChat, 2025) reports false barge-in rate and T90 latency. Vendors define false barge-in, missed barge-in and false-stop rates per turn. Voxa should normalise false barge-ins per hour of TTS playback, the wake-word convention, because playback time is the exposure.

### Cited Findings
- Amazon certification-style definitions: false rejection rate = missed wake words / wake words spoken; false accept rate = incorrect wakes during the test period. Picovoice argues FA should be reported per hour, with FRR read at a fixed FA rate such as 1 FA/h. — search summary of [Picovoice wake-word benchmark](https://picovoice.ai/blog/benchmarking-a-wake-word-detection-engine/) and [Amazon Science blog](https://www.amazon.science/blog/alexa-scientists-present-two-new-techniques-that-improve-wake-word-performance)
- Amazon reported a ~16% relative false-reject reduction and a ~37% relative false-alarm reduction with a two-stage on-device wake-word approach. These are relative numbers only. — [Amazon Science](https://www.amazon.science/blog/alexa-scientists-present-two-new-techniques-that-improve-wake-word-performance)
- A Northeastern/Imperial study found smart speakers misactivated about once per hour, in the worst combinations, while TV shows played. This is from secondary coverage. — [Consumer Reports](https://www.consumerreports.org/smart-speakers/yes-your-smart-speaker-is-listening-when-it-should-not)
- Google: endpointer latency is the time from the end of user speech to the system's end-of-query decision, reported as EP50 (median) and EP90 (90th percentile). — [arXiv 2011.10798](https://arxiv.org/pdf/2011.10798)
- Unified ASR+endpointing reduced median endpoint latency by 120 ms and EP90 by 170 ms without WER regression. — [arXiv 2211.00786](https://arxiv.org/pdf/2211.00786)
- An "ep-cutoff" metric, the proportion of user turns where the endpointer fires before the true turn end, is used in a 2025 streaming-endpointer paper. This is from the snippet; I could not confirm the affiliation. — [arXiv 2506.07081](https://arxiv.org/pdf/2506.07081)
- FireRedChat (full-duplex cascaded system):
  - Its turn-taking controller uses streaming personalized VAD to "suppress false barge-ins from noise and non-primary speakers".
  - The paper proposes barge-in, end-of-turn accuracy and end-to-end latency as system metrics (from the abstract).
  - The search snippet reports false barge-in rates of 33.4% for LiveKit, 78.1% for Ten and 10.2% for FireRedChat, with a T90 latency. I did not verify these numbers in the full text.
  - [arXiv 2509.06502](https://arxiv.org/abs/2509.06502)
- Vendor operational definitions, not peer-reviewed:
  - False barge-in rate: share of agent turns cut short by something other than the user ([caller.digital](https://caller.digital/blog/barge-in-turn-taking-benchmark-voice-ai-india-2026)).
  - "False stop rate": share of backchannels or noise events that halt the agent ([Deepgram](https://deepgram.com/blog/elevenlabs-barge-in-interruptions-turn-taking)).
  - Testing both directions is recommended, because false positives cut the agent off and false negatives make users repeat themselves ([Hamming](https://hamming.ai/blog/voice-agent-interruption-handling-runbook)).
- A practitioner blog describes the reverberant-room self-interruption failure: a reflected TTS tail survives the AEC and triggers barge-in or the wake word. The suggested test plays fixed phrases at several volumes, adds human interruptions and TV speech, and measures false interrupts per hour, missed barge-ins, and recovery time after an echo-path change. — [Zimaspace blog](https://shop.zimaspace.com/blogs/tech-ai-hub/local-voice-assistant-interrupts-itself-reverberant-room)

### Inferences
Proposed Voxa definitions:
- False barge-in (self-interruption) rate: `FBI/h = barge-in triggers with no user speech onset within the trigger window / hours of TTS playback`. Count it separately for:
  - (a) pure TTS (FST)
  - (b) TTS plus non-speech noise
  - (c) TTS plus background TV/other talker, if Voxa claims target-speaker robustness
- Missed barge-in rate: `MBI = user barge-in attempts (≥ 2 words, ≥ 0.5 s) that do not stop TTS within 1.0 s / attempts`. Report it per SER bin.
- Barge-in detection latency: `t_stop − t_user_onset`, where t_onset is the reference VAD onset of the clean near-end track and t_stop is when TTS output is muted. Report P50/P90.
- Endpointing: EP50/EP90 = `t_endpoint − t_last_word_end`, and PCO as in Q2.
- Wake word (if used during TTS): FA/h measured over hours of TTS self-playback plus TV audio, and FRR at that operating point.

### Gaps
- I found no peer-reviewed absolute numbers for Alexa/Google false barge-in or self-wake during playback. Amazon publishes relative improvements only in what I found.
- FireRedChat's exact test protocol and numbers need confirmation from the full text.

## Q4. Datasets and simulation ingredients

### Takeaway
The Microsoft AEC Challenge provides the most relevant data: 10,000 synthetic far/near/echo/mic quadruples with full metadata, plus real recordings from over 10,000 devices. Its synthetic recipe is a direct, citable template for the bench: RIR convolution, 80% loudspeaker nonlinearity, SER drawn uniformly from −10 to +10 dB, and noisy speech at SNR 0-40 dB.

### Cited Findings
- AEC Challenge synthetic set:
  - 10,000 scenarios, each with far-end speech, echo, near-end speech and mic signal, covering single talk, double talk, near- and far-end noise and nonlinear distortion.
  - Source speech is LibriVox clips filtered to P.808 MOS 4.3-5 (from the DNS Challenge). The noisy versions mix in AudioSet/Freesound/DEMAND noise at SNR ~U[0, 40] dB.
  - Far end is 10 s of one random speaker out of 1,627. Near end is 3-7 s of a different speaker, zero-padded to 10 s.
  - Echo is far end convolved with a random RIR (RT60 200-1200 ms).
  - In 80% of cases the far end first passes through a nonlinearity: hard clipping, a sigmoid, or learned distortion, with linear-to-nonlinear ratio 0.25.
  - Mixing uses SER ~U[−10, +10] dB, computed on clean speech.
  - 50% of far/near signals come from the noisy set. The first 500 clips have held-out speakers and RIRs for validation.
  - [arXiv 2309.12553 §IV-B](https://arxiv.org/pdf/2309.12553)
- AEC Challenge real data: recordings from over 10,000 audio devices and speakers, covering the scenarios in Q1, with loopback captured alongside the mic by a custom recording tool. Datasets live in the repo (`datasets/…`, including `test_set_icassp2022`, `fullband`, `fullband_mobile`). — [microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge)

### Inferences
These are from general knowledge; I did not fetch sources for them this session, so verify before citing.
- LibriSpeech/LibriTTS (CC-BY 4.0) for user and TTS-like speech with transcripts.
- MUSAN (music, speech, noise) for TV-like interference.
- RIRS_NOISES (OpenSLR SLR28) and BUT ReverbDB for real RIRs.
- pyroomacoustics (`ShoeBox`, image-source method plus ray tracing) for parametric rooms where speaker and mic positions are set close together, like a laptop or desktop.
- Clock drift: resample the loopback by a factor of (1 + ppm·1e−6) with ±50-200 ppm, for separate USB speaker and mic devices. This is a design choice, not from a source.
- Voxa's TTS should itself be the far-end source in most cases. The AEC Challenge uses human speech for the far end, but Voxa's echo is always its own synthetic voice, which matters for ASR echo-leak behaviour.

### Gaps
- Licences and download details for BUT ReverbDB, RIRS_NOISES and MUSAN were not checked this session.
- I found no public dataset of a TTS assistant being barged-in with labelled onsets.

## Q5. Reproducible offline bench design (synthetic + recorded), metric definitions, gates

### Takeaway
Build a deterministic, seed-controlled bench that renders Voxa's own TTS through simulated and measured speaker-to-mic paths. It mixes in labelled user speech at controlled SER/SNR and timing, and injects delay, drift and nonlinearity. It feeds the result through Voxa's real capture pipeline (AEC → VAD → endpointer → faster-whisper → barge-in logic) as a file-driven fake audio device, and emits one JSON of metrics with pass/fail gates. A small real-room corpus recorded with a fixed script calibrates the synthetic results. The design follows the AEC Challenge recipe and scenario split (cited in Q1/Q4). The target numbers below are proposals unless a citation is given.

### Cited Findings
- Recipe parameters to reuse:
  - SER U[−10, +10] dB
  - SNR U[0, 40] dB
  - RT60 0.2-1.2 s
  - 80% nonlinear loudspeaker (clip or sigmoid)
  - FST, NST and DT, each with and without echo-path change
  - score FST on the second half and DT on the last third
  - [arXiv 2309.12553](https://arxiv.org/pdf/2309.12553)
- Full-duplex reference targets: near-end attenuation during double talk ≤ 3 dB, echo loss in double talk ≥ 27 dB (P.340 Cat 1 via ETSI TS 103 739). — [ETSI TS 103 739](https://www.etsi.org/deliver/etsi_TS/103700_103799/103739/01.01.02_60/ts_103739v010102p.pdf)
- AECMOS via `speechmos` with talk_type st/nst/dt. — [PyPI speechmos](https://pypi.org/project/speechmos/)
- Industry reports EP50/EP90 latency. — [arXiv 2011.10798](https://arxiv.org/pdf/2011.10798)

### Inferences
Proposed bench, `voxa/bench/`, seeded and run with one command such as `python -m voxa.bench run --suite smoke|full --seed 0`.

1. **Sources**
   - User speech: LibriSpeech test-clean/test-other utterances with transcripts, plus 50-100 of rhY's own recorded command phrases.
   - Assistant speech: Voxa's actual TTS rendering of about 200 typical responses, with text and word timings saved.
   - Interference: MUSAN music/speech/noise (TV proxy).
2. **Echo paths (three families)**
   - (a) pyroomacoustics ShoeBox rooms with RT60 0.2-0.8 s and speaker-mic distance 0.1-1.0 m.
   - (b) Real RIRs (BUT ReverbDB / RIRS_NOISES).
   - (c) Measured RIRs of rhY's own setup: play an ESS sweep from Voxa's speaker, record at the mic, deconvolve. Store 5-10 positions and volumes.
3. **Loudspeaker and device impairments**
   - Nonlinearity in 80% of clips: hard clip at 0.5-0.9 of full scale, or `tanh(g·x)` with g 1-4.
   - Bulk delay of 0-300 ms, plus a step change of 20-80 ms mid-clip.
   - Clock drift of ±0-200 ppm via resampling.
   - Volume steps during playback.
   - Echo-path change mid-clip (switch RIRs).
4. **Mixing**
   - SER ∈ {−10, −5, 0, +5, +10} dB, computed on the clean near-end and echo components.
   - SNR ∈ {∞, 30, 20, 10} dB.
   - Barge-in onset placed at fixed relative points (0.2/0.5/0.8 of TTS) and at TTS start.
   - Save s, d, n and the mic mix separately, plus a JSON of ground-truth onsets, offsets and words.
5. **Scenarios**
   - FST: TTS only, the self-interruption and echo-leak test. Accumulate ≥ 2 h of playback for per-hour rates.
   - NST: user only.
   - DT: barge-in.
   - DT plus TV.
   - Cold start: user speaks within 200 ms of TTS end, testing the AEC tail and VAD hangover.
   - Echo-path change and drift.
6. **Pipeline under test**
   - Inject the mic and loopback WAVs through the same frame callback Voxa uses live, at real 16 kHz frame timing simulated in faster-than-real-time mode.
   - Log timestamps for VAD on/off, barge-in trigger, TTS mute, endpoint and the ASR transcript.
7. **Recorded corpus**
   - A script prompts rhY (and optionally others) to read 100 phrases, while a second script plays Voxa TTS at three volumes.
   - Record mic plus loopback (PipeWire monitor) at 48 kHz.
   - Hand-label barge-in onsets with a VAD of the near-end-only take, or with a close-talk second mic or headset as reference.
   - Re-run the same corpus on every change. Keep it in a git-LFS or ignored data folder with a checksum manifest.
8. **CI gate.** One `bench_report.json` with each metric, its 95% CI (bootstrap over clips), the threshold and pass/fail. The exit code is non-zero on any fail, so Strata can run it as a numeric gate. Use a smoke suite of about 5 min for each task and the full suite nightly.

Proposed metrics and targets (my proposals; tune after a first baseline run):

| Metric | Formula | Proposed gate |
|---|---|---|
| ERLE (FST, converged, synthetic) | 10·log10(Σd²/Σŝ²) over far-end-only frames, after 2 s | ≥ 25 dB median, ≥ 15 dB P10 |
| Near-end attenuation in DT | 10·log10(Σs²/Σ(proj of ŝ on s)²) in DT minus the same in NST | ≤ 3 dB (P.340 Cat 1, cited) |
| AECMOS echo_mos (FST/DT) | speechmos, talk_type st/dt | ≥ 4.3 (MS-1 achieves ~4.65, cited) |
| AECMOS deg_mos (DT) | speechmos | ≥ 3.8 |
| FST transcript rate | clips with ≥1 ASR word / FST clips | ≤ 1% |
| Echo-leak words | leak words / TTS words played | ≤ 0.1% |
| False barge-in (self-interrupt) | triggers without user onset / h TTS | 0 on smoke; ≤ 0.5/h full (≤ 1/h with TV) |
| Missed barge-in | attempts not stopping TTS ≤ 1.0 s / attempts | ≤ 2% at SER ≥ −5 dB; ≤ 10% at −10 dB |
| Barge-in latency | t_stop − t_onset | P50 ≤ 300 ms, P90 ≤ 600 ms |
| User word recall NST | hits / N_user | ≥ 97% (LibriSpeech clean, small Whisper ~3-5% WER expected) |
| User WER DT vs NST | WER_DT − WER_NST | ≤ +3 points absolute at SER ≥ −5 dB |
| First-word truncation | first ref word D/S / utterances | ≤ 1% cold start; ≤ 3% barge-in |
| Premature cut-off | endpoint < last word end / utterances | ≤ 1% |
| Endpoint latency | t_endpoint − t_last_word_end | EP50 ≤ 700 ms, EP90 ≤ 1200 ms |
| Recovery after path change | time until ERLE ≥ 15 dB again | ≤ 2 s |

- Report every metric per stratum (SER bin, RT60 bin, nonlinear yes/no, drift yes/no), so a regression points to its cause.
- Seed everything (numpy RNG and the order of torch, faster-whisper `temperature=0` beam settings). Store the rendered set's hash so runs are comparable.
- Whisper-specific guards: set `condition_on_previous_text=False` during evaluation, and record `no_speech_prob`, to separate hallucination from echo leak.

### Gaps
- No source validates the specific proposed gate values for a single-mic desktop assistant. Calibrate them against the baseline and the recorded corpus.
- I found no published approach for measuring barge-in latency against an AEC'd single mic without a reference track. The bench relies on synthetic ground truth, plus a close-talk reference mic in the recorded corpus.
