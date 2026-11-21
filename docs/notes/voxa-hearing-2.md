---
name: voxa-hearing-2
description: rhY 2026-11-20 "whatever it takes" to make Voxa's hearing perfect; Hearing 2 program h01-h05 on the fleet, root cause and plan.
metadata:
  type: project
---
rhY 2026-11-20 23:10: "Whatever it takes, more research, inventing new algorithms, we need to fix voxa's hearing and make it
perfect. Do whatever it takes. Overcome any obstacle."

**Root cause (f129/f131/f132 notes):** linear echo stage removes ~3 dB (pywebrtc AEC3 has no config; no gst webrtcdsp);
later stages fight leftover echo; delay estimate drifts ~2 ms. Old chain gates were never checked against what is achievable.

**Program** (brief ~/strata-fleet/research/voxa-hearing-2/_hearing-2.md; crhy/voxa#72 comment): h01 frontier bench (AEC3
full config built from source + shim, SpeexDSP, DTLN-aec, NKF-AEC, hybrids -> docs/hearing/frontier.md), h02 exact
alignment from the render timeline (<=0.1 ms). TO WRITE after h01 results: h03 integrate the winner (Flatpak), h04
known-voice rejection (Voxa's TTS voiceprint + known-words echo veto + enrolled user voiceprint), h05 gates from the
frontier + re-run chain t2-t10 (f133-f137, f146 moved to the END of the queue until then).
Queue: f161 (pause) -> f160 (0.1.9 integration) -> h01 -> h02 -> other repos.

**How to apply:** review h01's frontier.md, then write h03-h05 with gates at/above the frontier; never let tasks open the
mic or play sound (offline bench only); tell rhY before any live audio run. Related: [[voxa-speed-is-top-priority]],
[[strata-fleet]], [[voxa-tell-before-live-audio-tests]].
