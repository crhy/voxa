"""Issue #72, task v08 step 4: the replay gate over the rc4 live log.

The recorded live run (0.8B brain, barge.log 18:20+) is replayed through the real decision
path by ``tools/self_hearing_replay.py``; the fixture and hand labels live in
``tests/fixtures/``. This test runs the tool and demands the two lines the task requires:
no echo line ever interrupts, and no user line is swallowed as her own voice.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_live_replay_never_interrupts_herself() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "self_hearing_replay.py")],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    m = re.search(r"^self-interrupts: (\d+) of ([1-9]\d*) echo lines$", out, re.MULTILINE)
    assert m, out
    assert m.group(1) == "0", out  # required: N = 0
    k = re.search(r"^user lines heard: ([1-9]\d*) of \1$", out, re.MULTILINE)
    assert k, out  # required: K = U (every user line survives as not-her-voice)
