"""The hearing bench: a file-driven pipeline that renders clips, runs them through
Voxa's real hearing path and compares what it heard with what the clip said.

Roadmap order (B1-B5): render.py (B1) -> runner.py + run (B2) -> agree.py + check (B2/B3)
-> metrics.py + gates.toml + baseline.json (B3-B5).

Every stage is a file: clip.wav in, events.jsonl out. Nothing here touches a sound
device, and the runner's clock is simulated, so a suite runs faster than real time.
"""
