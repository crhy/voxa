#!/usr/bin/env python3
"""Copy Voxa's barge.log while a live run is going (issue #72, task v09).

Voxa's _barge_trace truncates barge.log when it passes 300 KB, so a long run can
lose its own early lines (and the rc4 history) before the end-of-run copy. This
tailer polls the log, appends every new chunk to <out>/barge-live.part-N files,
and records truncations so the full run log can be reassembled afterwards.

Usage: barge_log_tailer.py <barge.log> <out-dir> [poll_seconds]
Stops when the sentinel file <out-dir>/STOP appears.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log = Path(sys.argv[1])
    out = Path(sys.argv[2])
    poll = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    out.mkdir(parents=True, exist_ok=True)
    part = out / "barge-live.part-0"
    offset = part.stat().st_size if part.exists() else 0
    part_index = 0
    while True:
        try:
            size = log.stat().st_size
            if size < offset:  # the app truncated (or rotated) the log
                offset = 0
                part_index += 1
                part = out / f"barge-live.part-{part_index}"
                (out / "barge-live.truncations").open("a").write(
                    f"{time.strftime('%H:%M:%S')} truncated at byte {size}\n")
            if size > offset:
                with log.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                with part.open("ab") as handle:
                    handle.write(chunk)
                offset = size
        except FileNotFoundError:
            pass
        if (out / "STOP").exists():
            return 0
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main())
