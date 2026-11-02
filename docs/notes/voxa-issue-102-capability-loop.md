---
name: voxa-issue-102-capability-loop
description: rhY's standing order (2026-10-10) to work through GitHub issue crhy/voxa #102's 100 assistant capabilities with VoxaTest until each is 90% successful, through Monday 2026-10-12
metadata:
  type: project
---

On 2026-10-10 about 12:10 rhY said: "See Issue #102. Start building test cases for voxatest/voxa, and implement them 1 at a time. Start and stop strata as needed, then continuously run the voxa/voxatest test suite until all of those tasks are 90% successful. When you run out of tokens, start again after the tokens reset. Keep going all weekend and Monday. Provide beautiful graphs about additional capabilities and it's success/failure rate, and update the capabilities on gh continuously."

Issue #102 is "Digital Assistant ToDo List": 100 capabilities in 10 categories (saved as `/Voxa/.voxa-spec/issue102/items.json`).

Later the same day (about 14:00) rhY raised the bar: "Just keep going over and over until all 100 personal assistant tasks get a 30 out of 30 between Voxa and Voxa test. Make recommendations on other improvements we can make after that." My reading: every capability must reach a perfect spoken score (30 of 30), not just 90%; when that is reached, write up recommended further improvements. Live runs can be recorded in OBS (`RECORD=1` before the run script; rhY asked for specific runs to be recorded, twice on 2026-10-10 — my reading is that he asks run by run, so do not record every run unprompted), staged on an empty side of the Compiz cube, with the High ("live") face switched on, and Strata stopped before each run and started again after.

**Why:** rhY wants Voxa to become a real hands-free assistant and wants measured, visible progress.

**How to apply:**
- The working state is `/Voxa/.voxa-spec/issue102/STATE.md` — read it first every session and update it before stopping; it lists what is done and the next step.
- One capability at a time: write spoken cases (tagged with the capability number), implement, run live, fix, repeat until its pass rate is at least 90%.
- Coding goes to the local model where practical ([[delegate-coding-to-local-qwen]]): start and stop Strata only with `/Voxa/.voxa-spec/issue102/strata.sh start|stop` (no windows). Do NOT use `strata-opencode-starter` for this: on 2026-10-10 it left btop/nvtop/server terminals and an error dialog on rhY's screen, which rhY sent a screenshot of. Stop Strata before live runs (GPU and memory).
- Live runs: `/Voxa/.voxa-spec/out/night-20261010/sidebyside.sh` (Voxa left, VoxaTest app right, for rhY's video) — never lower the volume ([[voxatest-never-lower-volume]]); say so before audio starts ([[voxa-tell-before-live-audio-tests]]).
- "Update on gh" is done as one status comment on issue #102 that is edited each cycle (tables plus mermaid charts). My interpretation: this does not authorise pushing commits; all code is still local and uncommitted unless rhY says otherwise ([[voxa-0-1-6-handoff]]).
- Capabilities that need rhY's accounts, money or a phone line (email inbox, purchases, calls) must be marked "needs rhY" rather than faked; spending money and sending messages stay approval-only.
- An hourly session cron re-enters this work, but it only lives while the Claude session is open.
