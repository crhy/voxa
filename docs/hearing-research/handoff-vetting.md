# Vetting of handoff-full-duplex.md (Claude, 2026-11-17)

## Voxa code findings (section 5): ALL CONFIRMED on origin/main 480d1d7 (= the handoff's snapshot)
- 5.1 voxa/conversation.py:331-364, 585-596: feed() fills `_barge_buf` (160000 bytes ~5 s) while muted; candidate_text()
  joins it and CLEARS it; mute()/unmute() clear it. voxa/window.py:3152-3250 `_maybe_barge_in`: on INTERRUPT it calls
  speech.stop(), conversation.unmute() (clears again), arm_prompt(). The words that proved the interruption never reach
  the request: the user must repeat themselves. CONFIRMED, highest priority.
- 5.2 voxa/bargein.py:68-94 is_own_voice(): `if len(words) < 3: return True` (comment: "one or two stray words are
  usually her own voice"). "yes", "blue", "that one" over TTS are discarded. CONFIRMED.
- 5.3 voxa/ownspeech.py:43-68: deque(maxlen=3) texts, keep_seconds=6.0, removes matching word runs. CONFIRMED.
- 5.4 voxa/window.py:181-183 BARGE_IN_GRACE_SECONDS=0.4, BARGE_IN_LISTEN_SECONDS=0.9; ASR runs inside
  _maybe_barge_in (GTK tick). CONFIRMED.
- 5.5 voxa/audio.py:470-485 delay estimated ONCE (`_delay_done`). EXTRA finding: voxa/anc.py:115
  `self._ref_hist = np.concatenate([self._ref_hist, ref_b])` grows WITHOUT BOUND (memory + O(n^2) copying over a
  long session). ReferenceRing in audio.py IS bounded.
- 5.6 voxa/audio.py:321-350 probe on a default-sink monitor branch, not the TTS playback branch. CONFIRMED.
- 5.7 voxa/audio.py:62-63 mix_chunk returns raw mic when the reference window is silent (ignores the echo tail);
  appsinks max-buffers=12 drop=True (audio.py:260-263, 361-363). CONFIRMED.

## External references spot-checked
- DiffVQE2 arXiv 2609.31703: exists (TU Braunschweig / GN; beats DeepVQE on AEC Challenge 2023 blind set, authors' claims,
  no code found). HiThink Turn arXiv 2609.34096: exists (intent vs completeness, playback-conditioned).
- WebRTC NeuralResidualEchoEstimator: exists in libwebrtc api/audio + aec3/residual_echo_estimator.cc; takes a TFLite
  model path; NO official public model confirmed (discuss-webrtc thread, May 2026 unanswered).
- WO2025136579A1 (Google): NOT confirmed by search; treat as unverified. Other patents/papers match the first research pass.

## How the two research passes combine
- Agreement: exact timestamped TTS reference, AEC3 as linear stage + residual stage, reference-aware double-talk evidence,
  pre-roll, separate fast yield vs semantic decisions, endpointing beyond silence, ASR-level evaluation, ERLE alone useless.
- The handoff CORRECTS the first pass on: text-overlap own-speech filtering must NOT delete words (evidence only; the
  user may repeat Voxa's words); short utterances are uncertainty, not echo; never reset the canceller per sentence
  (re-estimate delay but keep the filter); provisional pauses/ducks count as self-interruptions; never lower system
  volume; one AEC owner; raw fallback = degraded state, no automatic commands from it; audio work off the GTK thread;
  test-fixture playback (voxatest) must not be cancelled as Voxa's echo.
- The handoff ADDS: Milestone A (stop losing words) BEFORE any AEC change; metrics self_command_rate, turn retention
  (frame ids, no loss/dup), pair accuracy (matched keep/yield pairs), stop latency; the 8.2 test matrix.
