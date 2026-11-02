# Project notes

Working notes, decisions and standing orders for this project, kept by Claude Code while working with rhY. Each file is one fact or decision; `[[name]]` links point to other notes (some live in other crhy repos' `docs/notes/`).

- [voxa-0-1-6-handoff](voxa-0-1-6-handoff.md): State of Voxa 0.1.6 work as of 2026-10-07 ~13:00 — what is built, what rhY is about to test, open decisions, and how the Strata/Qwen task queue is run
- [voxa-0-1-7-release-plan](voxa-0-1-7-release-plan.md): Voxa 0.1.7's main goal (end users can get Ollama models and the High face), the GitHub-hosting decision, and the later torrent update backend
- [voxa-close-issues-with-summary](voxa-close-issues-with-summary.md): When a Voxa GitHub issue is resolved, close it with a comment covering what was done, how, and what would make it better
- [voxa-flatpak-host-access-approved](voxa-flatpak-host-access-approved.md): rhY explicitly approved broad host access for the Voxa flatpak (start/stop local AI servers) and wants Voxa to reach Brave tabs, LibreOffice and everything else via permissioned tools.
- [voxa-issue-102-capability-loop](voxa-issue-102-capability-loop.md): rhY's standing order (2026-10-10) to work through GitHub issue crhy/voxa #102's 100 assistant capabilities with VoxaTest until each is 90% successful, through Monday 2026-10-12
- [voxa-separate-from-voice2text](voxa-separate-from-voice2text.md): Voxa and Voice2Text AI are two separate apps that must look nothing alike; a mixup between them already happened once.
- [voxa-spec-lives-in-github-issues](voxa-spec-lives-in-github-issues.md): Voxa's directives are GitHub issues on crhy/voxa; #7 is the master roadmap (phases A-E) and supersedes #6. Read them, comments included, and work one phase at a time.
- [voxa-speed-is-top-priority](voxa-speed-is-top-priority.md): rhY made response speed and natural conversation Voxa's top priority (GitHub crhy/voxa #69), ahead of new features; goal is the most efficient and most capable voice assistant.
- [voxa-tell-before-live-audio-tests](voxa-tell-before-live-audio-tests.md): Tell rhY before any lab test that plays sound or records the microphones, so the room can be quietened first
- [voxa-two-brains-plan](voxa-two-brains-plan.md): Voxa's next phase (after the Dr. Mario Wii port ships) is the two-brain design in crhy/voxa#116.
- [voxatest-clean-up-after-each-run](voxatest-clean-up-after-each-run.md): After every live Voxa/VoxaTest run, close what the run opened (VLC, windows, browser tabs) and remove its leftovers
- [voxatest-never-lower-volume](voxatest-never-lower-volume.md): VoxaTest runs and Voxa lab runs must never turn rhY's system volume down
