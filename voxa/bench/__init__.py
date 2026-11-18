"""The Voxa hearing bench (roadmap B1-B5 in the research notes).

Everything here runs OFFLINE on files: no microphone, no speakers, no audio devices.
``render`` (B1) writes seeded synthetic clips, later tasks add ``run``, ``agree`` and ``check``.
Clip data lives outside the repository in ``/home/rhy/voxa-bench-data``; only small per-run
output goes in the git-ignored ``.voxa-bench/``.
"""
