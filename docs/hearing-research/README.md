# Voxa hearing research (2026-11-17/18)

How Voxa can hear everything the user says, nothing of its own voice, and never interrupt itself.

- `voxa-hearing-research.md` - the deep-research report: echo cancellation (WebRTC AEC3, neural AEC), how Alexa,
  Google, Siri, Sonos and OpenAI handle barge-in (papers and patents), turn-taking, target-speaker detection, Linux
  audio plumbing, evaluation, and an 18-task engineering roadmap with numeric gates.
- `handoff-full-duplex.md` - a second, independent research pass (full-duplex architecture, Voxa code findings,
  benchmark matrix, milestones A-D, 33 references).
- `handoff-vetting.md` - which of the handoff's claims were checked against the code and sources, and where the two
  passes disagree.
- `notes/` - the six topic notes behind the report, with every source.

Tracking issue: #72. The implementation runs as the `voxa-hear` task series (bench, then milestones).
