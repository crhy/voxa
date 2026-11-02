---
name: voxa-tell-before-live-audio-tests
description: Tell rhY before any lab test that plays sound or records the microphones, so the room can be quietened first
metadata:
  type: feedback
---

Before any Voxa echo-lab run that plays sound through the speakers or records the microphones, give rhY a heads-up first (rhY, same evening: "Live audio is fine, I just need a heads up" — notice, not permission). Offline processing of stored recordings needs no notice.

**Why:** On 2026-10-07 rhY said "Tell me before we do more lab tests. I'll close the door and lower the ambient noise." A Roomba had spoiled one recording that evening (noise floor −41 dBFS instead of about −52), and room noise sets the ceiling of every echo measurement.

**How to apply:** Announce the test, its length and loudness, and give rhY a moment to close the door; wait for a "ready" only when the test needs rhY to do something (e.g. read a script). Check the noise floor (Blue mic below about −48 dBFS) before recording. Local-model tasks must not record on their own: `/Voxa/.voxa-spec/tasks/agent/_header.md` carries a "NO LIVE AUDIO" line for that. Related: [[voxa-0-1-6-handoff]], [[no-desktop-automation]].
