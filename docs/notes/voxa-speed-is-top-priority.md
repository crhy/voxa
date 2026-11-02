---
name: voxa-speed-is-top-priority
description: rhY made response speed and natural conversation Voxa's top priority (GitHub crhy/voxa #69), ahead of new features; goal is the most efficient and most capable voice assistant.
metadata:
  type: project
---

On 2026-10-04 rhY said of the speed and naturalness plan: "Let's move this to our top
priority. We want to be the most efficient voice assistant that exists, and the most
capable." The plan is GitHub issue crhy/voxa #69 (measure every stage, smarter end of
speech, tiered models chosen automatically, streaming speech, natural turn-taking);
capability work is tracked in the roadmap issue #68, with browser control (#40) called
"a key feature for almost everything we want to do".

**Why:** rhY wants Voxa to clearly beat Siri, Alexa and Gemini; in testing it "was a
little bit slow".

**How to apply:** when choosing the next Voxa task, latency work from #69 and browser
control come before new assistant features. Measure before and after any speed change
and report the numbers. Coding goes to the local model per [[delegate-coding-to-local-qwen]].
