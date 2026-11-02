---
name: voxatest-never-lower-volume
description: VoxaTest runs and Voxa lab runs must never turn rhY's system volume down
metadata:
  type: feedback
---

On 2026-10-10 about 02:30, during a live VoxaTest run, rhY said: "Yeah just don't turn the volume down." A spoken test case ("Set the volume to 40 percent") had lowered the system volume from 70% to 40%.

**Why:** rhY was in the room using the machine; the lower volume also made every later spoken case inaudible to Voxa.

**How to apply:** no spoken test case may change, set or mute the volume (the shipped case files are checked for this in `tests/test_voxatest.py`, and the case generator's prompt forbids it). Note the volume before a live run and compare afterwards; if rhY changed it by hand during the run, leave it alone. Related: [[voxa-tell-before-live-audio-tests]], [[voxa-0-1-6-handoff]].
