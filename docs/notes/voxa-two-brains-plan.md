---
name: voxa-two-brains-plan
description: Voxa's next phase (after the Dr. Mario Wii port ships) is the two-brain design in crhy/voxa#116.
metadata:
  type: project
---

rhY decided on 2026-10-22: finish the Dr. Mario Wii port first, then Voxa's two-brain system, specified in https://github.com/crhy/voxa/issues/116 (posted at his request).

Key points he asked for: little brain (LFM2.5-2.6B NEO-MAX Q6_K, downloaded to /media/rhy/data/Models/LFM2.5-2.6B-TBrilliance/) keeps quick web searches, app opening, dictation; an ABLITERATED big brain, resident for instant switching, takes tougher/more detailed questions and must consider alternate and non-mainstream sources (show disagreement, cite each); the big brain reviews the little brain's answers in the background and live usage data (local, opt-in) feeds concrete improvements via VoxaTest.

**Why:** rhY wants Voxa fast for easy things and much smarter for hard ones without manual switching.

**How to apply:** start Voxa work from #116; big-brain model choice and the GPU split are open and must be decided by measurement. Related: [[voxa-speed-is-top-priority]], [[voxa-spec-lives-in-github-issues]], [[delegate-coding-to-local-qwen]].
