---
name: voxa-spec-lives-in-github-issues
description: Voxa's directives are GitHub issues on crhy/voxa; #7 is the master roadmap (phases A-E) and supersedes #6. Read them, comments included, and work one phase at a time.
metadata:
    pinned: false
---

When rhY says to "follow the directives" for Voxa, they mean the GitHub issues on
`crhy/voxa`: #6 is the master implementation roadmap (32 phases, milestones A-F),
#5 is the detailed UI spec, #3 is the visual mockup image, #4 and #2 cover the 3D
avatar, lip-sync and voices. rhY stressed "the comments matter too": the comments
on #5 hold rhY's own reference code (`voxa/ui/state.py`, the `_build_assistant_shell`
overlay layout, and the CSS), which should be used as the base rather than
reinvented. The comments on #4 list open-source avatar-chat projects; rhY is happy
for their code to be borrowed ("projects to steal code from"), subject to checking
each project's licence.

Read every issue and every comment before starting Voxa work, and follow the
roadmap's own rules: small commits, milestones in order, no toolkit rewrite, don't
regress existing functionality.

Practical note: `gh issue view` failed with a 401 in this environment while
`gh api repos/crhy/voxa/issues/N` and `.../issues/N/comments` worked.

## Issue #7 is now the master roadmap (created 2026-09-19)

rhY said: "Issue #7 is the master roadmap. Continue Phase A only. Inspect the current
implementation and commits first, don't redo completed work, preserve tests, commit
logical milestones separately, and don't begin Phase B until Phase A's acceptance
criteria pass." #7 organizes the work as Phase A (consolidate: real MainWindow on the
new shell, AssistantModel authoritative, ACTIVE/OFFLINE guarantees, version metadata,
release naming, Flatpak repo URL, regression tests), B (robustness), C (agent
foundation), D (avatar), E (Spaced Linux integration).

Working rules that follow from it: do only the phase rhY names; verify the phase's
acceptance criteria explicitly before moving on; inspect `git log` and the code
before planning so finished work is not redone; keep separate logical commits; never
weaken tests. #7 section 18 also asks for intentionally narrow Flatpak permissions
(portals and narrow helpers over broad host access), which tensions with the broad
host access rhY approved earlier for starting local AI servers; raise that
explicitly when the server-management work comes up rather than assuming either way.
