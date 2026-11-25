#!/usr/bin/env python3
"""Issue #72, task v11 step 2: prove the reference tap is not silent in the REAL flatpak.

v08's offline test passed only because it ran with ``GST_AUDIOSINK=fakesink``. The installed
bundle plays through a real PulseAudio sink, and there Voxa's own speech did not feed the
reference: the tap read SILENT, ``reference active by definition while speaking: silent tap``
made attribution abstain, and real barge-ins died (the v09 live log has 0 INTERRUPTs).

This tool drives the installed bundle exactly like a user would and reads the app's own
``barge.log`` (``render-tap`` lines written by ``voxa/window.py``). For every reply it makes Voxa
speak it checks whether the reference tap recorded her own audio (a non-zero peak), and prints::

    reference silent while speaking: N of M checks

where a "check" is one reply and N must be 0 (never silent) for the fix to be real.

It only launches Voxa and asks her to read a file; it never touches the user's windows or the
system volume.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP_ID = "io.github.crhy.voxa"
# Where the sandboxed app writes barge.log (its XDG_STATE_HOME is inside the sandbox).
SANDBOX_LOG = Path.home() / ".var/app" / APP_ID / ".local/state/voxa/barge.log"
RENDER_TAP = re.compile(
    r"render-tap .*attached=(?P<attached>[A-Za-z]+) .*samples=(?P<samples>\d+) "
    r"peak=(?P<peak>[0-9.]+) silent=(?P<silent>[A-Za-z]+)"
)
# The three sentences the task asks her to read aloud: long enough to be a real reply.
SENTENCES = (
    "The first sentence is long enough that Voxa will still be speaking when the check begins. "
    "The second sentence keeps her talking so the reference has to stay non-silent across the gap. "
    "The third sentence ends the reply so the render tap can report its peak for the whole turn."
)


def _barge_log_path() -> Path:
    override = os.environ.get("VOXA_BARGE_LOG")
    return Path(override) if override else SANDBOX_LOG


def _app_running() -> bool:
    out = subprocess.run(["flatpak", "ps"], capture_output=True, text=True, check=False).stdout
    return APP_ID in out


def _launch_app() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["flatpak", "run", "--user", APP_ID],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def _make_file() -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, prefix="voxa-live-ref-")
    handle.write(SENTENCES)
    handle.close()
    return Path(handle.name)


def _action(action: str, param: str | None = None) -> bool:
    """Activate one of Voxa's exported app actions.

    ``read-file`` takes a string, so its parameter is passed as a GVariant literal (the app's
    own docs use ``read-file "'/home/me/intro.txt'"``); the listening actions take none.
    """
    argv = ["gapplication", "action", APP_ID, action]
    if param is not None:
        argv.append(f"'{param}'")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=15)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _trigger_read_file(path: Path) -> bool:
    return _action("read-file", str(path))


def _count_checks(after_byte: int, timeout: float) -> tuple[int, int]:
    """Watch one reply in ``barge.log`` and judge her reference honestly: (silent, checks).

    A reply is a CHECK once she started speaking (a ``speak-start`` line). It reads SILENT when
    the live tap could not measure her own voice - either the ``render-tap`` line reports a
    near-zero peak, or attribution fell back to ``reference active by definition while speaking:
    silent tap, abstain`` (the exact reason the v09 log had on every line). If the reply neither
    shows a speak-start nor times out, it is not a check (returns 0, 0) and the caller retries.
    """
    deadline = time.monotonic() + timeout
    path = _barge_log_path()
    spoke = False
    silent = False
    render_seen = False
    while time.monotonic() < deadline:
        if not path.exists():
            time.sleep(0.4)
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(after_byte)
            for line in handle:
                if "speak-start" in line:
                    spoke = True
                if "silent tap, abstain" in line:
                    silent = True
                match = RENDER_TAP.search(line)
                if match:
                    render_seen = True
                    samples = int(match.group("samples"))
                    peak = float(match.group("peak"))
                    if match.group("silent").lower() == "true" or (samples > 0 and peak < 1e-3):
                        silent = True
            after_byte = handle.tell()
        # A reply that finished (a reply-end/render-tap after a speak-start) settles the check.
        if spoke and (render_seen or "reply-end" in open(path, encoding="utf-8", errors="replace"
                                                          ).read()[after_byte:]):
            break
        time.sleep(0.4)
    if not spoke:
        return 0, 0
    return (1 if silent else 0), 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live_reference_check", description=__doc__)
    parser.add_argument("--replies", type=int, default=1,
                        help="how many replies to drive (default 1); every one is a check")
    parser.add_argument("--per-reply-timeout", type=float, default=25.0,
                        help="seconds to wait for each reply's render-tap line")
    parser.add_argument("--keep-app", action="store_true",
                        help="leave Voxa running at the end (default: stop the instance we started)")
    args = parser.parse_args(argv)

    log_path = _barge_log_path()
    started = None
    if not _app_running():
        started = _launch_app()
        # Give the app time to register its action and start its audio pipeline.
        time.sleep(12.0)
    # Put her in Active (issue #130's scripted listening) so she reads the reply back.
    _action("start-listening")

    total = 0
    silent_total = 0
    attempt = 0
    max_attempts = max(1, args.replies) * 3
    try:
        while total < max(1, args.replies) and attempt < max_attempts:
            attempt += 1
            before = log_path.stat().st_size if log_path.exists() else 0
            tmp = _make_file()
            if not _trigger_read_file(tmp):
                print("could not ask Voxa to read the file: is the app running and the "
                      "read-file action exported?", file=sys.stderr)
                return 2
            silent, checks = _count_checks(before, args.per_reply_timeout)
            total += checks
            silent_total += silent
            tmp.unlink(missing_ok=True)
    finally:
        if started is not None and not args.keep_app:
            subprocess.run(["flatpak", "kill", APP_ID], capture_output=True, check=False)
            started.terminate()

    print(f"reference silent while speaking: {silent_total} of {total} checks")
    return 0 if silent_total == 0 and total >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
