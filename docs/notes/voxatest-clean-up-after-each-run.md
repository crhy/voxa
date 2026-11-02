---
name: voxatest-clean-up-after-each-run
description: After every live Voxa/VoxaTest run, close what the run opened (VLC, windows, browser tabs) and remove its leftovers
metadata:
  type: feedback
---

On 2026-10-10 about 12:40, as the multi-day issue-102 test loop began, rhY said: "Don't forget to close VLC after vlc tests, close windows that were opened during testing, and to basically clean up things after each test run so the computer doesn't get overwhelmed after 3 days of tests."

**Why:** the runs open calculators, terminals, file-manager windows, browser tabs and VLC on rhY's own desktop; over days of unattended runs that would swamp the machine. (Seven runs on 2026-10-10 left 5 calculators, 12 terminals and 4 folder windows behind.)

**How to apply:** the live-run script (`/Voxa/.voxa-spec/out/night-20261010/sidebyside.sh`) records the open windows before a run and afterwards closes every window that was not there before, quits VLC if the run started it, puts back the notes and reminders files and the hidden-files setting, and deletes screenshots the run took. Web pages opened by test prompts go to a separate test browser (`VOXA_URL_OPENER`), never rhY's own Brave. Never close a window that existed before the run. Check `df` and memory between cycles too. Related: [[voxa-issue-102-capability-loop]], [[voxatest-never-lower-volume]].
