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
# Issue #72, task v11b step 10: the two live numbers the task's gate reads.
# A DECISION is one attribution line (it carries source='...'/reasons='...'); it is ALIGNED
# when its reasons do not say the echo path could not qualify the audio. An INTERRUPT line is
# a real barge-in she performed - while she speaks ALONE (nobody else in the room) every one
# of them is a false interrupt of herself.
DECISION = re.compile(r"source='(?P<source>[a-z-]+)'")
UNHEALTHY = "alignment unhealthy"
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


def _make_file(text: str | None = None) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, prefix="voxa-live-ref-")
    handle.write(text if text is not None else SENTENCES)
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


def _count_checks(after_byte: int, timeout: float) -> tuple[int, int, int, int, int]:
    """Watch one reply in ``barge.log`` and judge her reference honestly.

    Returns ``(silent, checks, decisions, aligned, interrupts)``:
    * one CHECK once she started speaking (a ``speak-start`` line);
    * SILENT when the live tap could not measure her own voice - the ``render-tap`` line reports
      a near-zero peak, or attribution fell back to ``reference active by definition while
      speaking: silent tap, abstain``;
    * DECISIONS / ALIGNED for step 10's ``alignment healthy while speaking: H of D`` - every
      attribution line during the reply, and those that are not ``alignment unhealthy``;
    * INTERRUPTS for step 10's ``false interrupts while she speaks alone: K of N`` - every
      ``INTERRUPT`` during the reply, which with nobody else talking is her cutting herself off.
    A reply that neither shows a speak-start nor times out is not a check (0, 0, 0, 0, 0).
    """
    deadline = time.monotonic() + timeout
    path = _barge_log_path()
    spoke = False
    silent = False
    render_seen = False
    decisions = aligned = interrupts = 0
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
                if DECISION.search(line):
                    decisions += 1
                    if UNHEALTHY not in line:
                        aligned += 1
                if re.search(r"\bINTERRUPT\b", line):
                    interrupts += 1
                match = RENDER_TAP.search(line)
                if match:
                    render_seen = True
                    samples = int(match.group("samples"))
                    peak = float(match.group("peak"))
                    if match.group("silent").lower() == "true" or (samples > 0 and peak < 1e-3):
                        silent = True
            after_byte = handle.tell()
        # A reply that finished (a speak-end/render-tap after a speak-start) settles the check.
        if spoke and (render_seen or "reply-end" in open(path, encoding="utf-8", errors="replace"
                                                          ).read()[after_byte:]):
            break
        time.sleep(0.4)
    if not spoke:
        return 0, 0, 0, 0, 0
    return (1 if silent else 0), 1, decisions, aligned, interrupts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live_reference_check", description=__doc__)
    parser.add_argument("--replies", type=int, default=1,
                        help="how many replies to drive (default 1); every one is a check")
    parser.add_argument("--per-reply-timeout", type=float, default=25.0,
                        help="seconds to wait for each reply's render-tap line")
    parser.add_argument("--keep-app", action="store_true",
                        help="leave Voxa running at the end (default: stop the instance we started)")
    parser.add_argument("--voice", default="",
                        help="issue #72 v11b step 11: label this run with the voice/language in use")
    parser.add_argument("--text-file", default=None,
                        help="issue #72 v11b step 11: read THIS file aloud instead of the English "
                             "default (each character speaks its own language)")
    args = parser.parse_args(argv)
    reply_text = None
    if args.text_file:
        reply_text = Path(args.text_file).read_text(encoding="utf-8").strip()

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
    decisions = 0
    aligned = 0
    interrupts = 0
    attempt = 0
    max_attempts = max(1, args.replies) * 3
    try:
        while total < max(1, args.replies) and attempt < max_attempts:
            attempt += 1
            before = log_path.stat().st_size if log_path.exists() else 0
            tmp = _make_file(reply_text)
            if not _trigger_read_file(tmp):
                print("could not ask Voxa to read the file: is the app running and the "
                      "read-file action exported?", file=sys.stderr)
                return 2
            silent, checks, reply_decisions, reply_aligned, reply_interrupts = _count_checks(
                before, args.per_reply_timeout)
            total += checks
            silent_total += silent
            decisions += reply_decisions
            aligned += reply_aligned
            interrupts += reply_interrupts
            tmp.unlink(missing_ok=True)
    finally:
        if started is not None and not args.keep_app:
            subprocess.run(["flatpak", "kill", APP_ID], capture_output=True, check=False)
            started.terminate()

    label = f" [{args.voice}]" if args.voice else ""
    print(f"reference silent while speaking: {silent_total} of {total} checks{label}")
    print(f"alignment healthy while speaking: {aligned} of {decisions}{label}")
    print(f"false interrupts while she speaks alone: {interrupts} of {total}{label}")
    return 0 if (silent_total == 0 and total >= 1 and decisions > 0 and aligned / decisions >= 0.9
                 and interrupts == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
