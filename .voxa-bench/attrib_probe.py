"""Scratch: run the 68 live heard lines through attribution with the FIXED reference."""
import json
import re
import sys

sys.path.insert(0, ".")
from voxa.attribution import SourceEvidence, attribute  # noqa: E402
from voxa.bargein import source_decision  # noqa: E402

rows = json.load(open(".voxa-bench/veto-analysis.json"))

for r in rows:
    heard = r["heard"]
    turn = r["turn"]
    ev = SourceEvidence(reference_active=True, alignment_healthy=True, speaking=True,
                        level_ratio=5.2, vad_prob=None)
    d = source_decision(heard, turn, "", raw=heard, evidence=ev, level_ratio=5.2)
    print(f"{r['t']} {d.source:<9} conf={d.confidence:.2f} | {heard[:60]}")
