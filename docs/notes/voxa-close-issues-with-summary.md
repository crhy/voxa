---
name: voxa-close-issues-with-summary
description: "When a Voxa GitHub issue is resolved, close it with a comment covering what was done, how, and what would make it better"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 712a9b75-b95e-5a26-a250-d807b40dfdb4
  modified: 2026-10-06T23:32:03.278Z
---

When a Voxa issue on `crhy/voxa` is resolved, close it with a comment that says (1) what was done, (2) how it was done, and (3) what needs to be done in future to make it work better.

**Why:** rhY asked for this on 2026-10-06 while Qwen was working through the issue list unattended: "When the issues are resolved, close the issue with a comment on what was done, how it was done, and what needs to be done in the future to make it work better."

**How to apply:** Close an issue only after I have checked the work myself (Qwen's tests passing is not enough; see [[delegate-coding-to-local-qwen]]). State plainly in the comment what was verified and what was not. If only part of an issue is done, comment with the same three points and leave it open. Issues are the spec: [[voxa-spec-lives-in-github-issues]].

**Update 2026-10-07:** rhY said "It looks like lots of issues are open on github that can be closed. Let's clean it up." after I had left built-but-untested issues open with status comments. So: close an issue once the work is built and gated (with the what/how/improve comment), without waiting for rhY's own test; he reopens or files a new one if the test fails. Keep open only what is genuinely unfinished, undecided or unverifiable (my interpretation of the scope).
