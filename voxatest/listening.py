"""The live-run listening preflight (issue #130): never run a case into a deaf Voxa.

v09's live run started while Voxa sat at "Press Active to start listening", and every
case came back `not_heard`: Voxa never starts listening by itself (privacy), and until
now nothing could press the button for a test. Voxa now answers

    gapplication action io.github.crhy.voxa start-listening
    gapplication action io.github.crhy.voxa stop-listening

exactly as if Active/OFFLINE had been pressed (voxa/application.py), and records the
real state in listening.json (voxa/listening.py). This module finds the command that
actually reaches this machine's Voxa, sends it, and reads that file back before the
first case and before every case after it.

Which command works, and why (verified on rhY's machine, 2026-11-25):
* Voxa owns her D-Bus name on the session bus while running (flatpak shares the session
  bus), so plain ``gapplication action io.github.crhy.voxa ...`` reaches a running Voxa
  directly - this is the form the Spaced Welcome recordings already use.
* Nothing running: the flatpak form ``flatpak run --command=gapplication io.github.crhy.voxa
  action io.github.crhy.voxa <action>`` is the one that can start her, and the file then
  lives under ~/.var/app/io.github.crhy.voxa/ - the candidates below cover both homes.
* VOXATEST_VOXA_ACTION_CMD overrides the choice entirely (quoted command prefix).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

APP_ID = "io.github.crhy.voxa"
DEFAULT_TIMEOUT_SECONDS = 60.0
RECHECK_TIMEOUT_SECONDS = 15.0


def state_path_candidates() -> list[Path]:
    """Where a listening.json could be, most specific first (Voxa's own homes)."""
    candidates: list[Path] = []
    override = os.environ.get("VOXATEST_LISTENING_STATE")
    if override:
        candidates.append(Path(override).expanduser())
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        candidates.append(Path(xdg_state).expanduser() / "voxa" / "listening.json")
    candidates.append(Path.home() / ".local" / "state" / "voxa" / "listening.json")
    # The Flatpak sandbox keeps XDG_STATE_HOME under ~/.var/app/<app id>.
    candidates.append(Path.home() / ".var" / "app" / APP_ID / ".local" / "state" / "voxa" / "listening.json")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def choose_state_path(candidates: list[Path] | None = None) -> Path:
    """Whichever listening.json was written last is Voxa's (same rule as the action log).

    An explicit VOXATEST_LISTENING_STATE is never second-guessed, even when the file
    is missing - that is how the doctor knows Voxa never reported at all.
    """
    if candidates is None:
        override = os.environ.get("VOXATEST_LISTENING_STATE")
        if override:
            return Path(override).expanduser()
        candidates = state_path_candidates()
    newest: Path | None = None
    newest_mtime = -1.0
    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = candidate, mtime
    return newest if newest is not None else candidates[0]


def _run_simple(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def bus_name_owned(name: str, runner=_run_simple) -> bool:
    """True when a running application already owns its D-Bus name on the session bus."""
    try:
        done = runner(["busctl", "--user", "list"])
    except (OSError, subprocess.SubprocessError):
        return False
    return name in (done.stdout or "").split()


def flatpak_installed(runner=_run_simple) -> bool:
    """True when Voxa is installed as a Flatpak (how crhy ships her)."""
    try:
        done = runner(["flatpak", "list", "--app", "--columns=application"])
    except (OSError, subprocess.SubprocessError):
        return False
    return APP_ID in (done.stdout or "").split()


def action_command(action: str, *, runner=_run_simple) -> list[str] | None:
    """The command that sends this app action here, or None when nothing can."""
    override = os.environ.get("VOXATEST_VOXA_ACTION_CMD")
    if override:
        return [*shlex.split(override), action]
    if bus_name_owned(APP_ID, runner=runner):
        # Running (host or Flatpak): the plain host command reaches her over the session bus.
        return ["gapplication", "action", APP_ID, action]
    if flatpak_installed(runner=runner):
        return ["flatpak", "run", "--command=gapplication", APP_ID, "action", APP_ID, action]
    if shutil.which("gapplication"):
        return ["gapplication", "action", APP_ID, action]
    return None


def read_state(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
        return None
    return data


class ListeningPreflight:
    """Sends the scripted actions and reads listening.json back.

    * ``ensure_ready()``: before the first case - never plays anything if it fails.
    * ``case_ready()``: before every case - re-sends start-listening once if she went
      inactive mid-run, and the run stops with that reason if she stays deaf.
    * ``finish()``: sends stop-listening only when this VoxaTest run started listening.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_SECONDS, runner=_run_simple) -> None:
        self.timeout = timeout
        self.runner = runner
        self.path = choose_state_path()
        self.started_by_us = False
        self.reason = ""
        self.commands_sent: list[list[str]] = []

    def _send(self, action: str) -> bool:
        command = action_command(action, runner=self.runner)
        if command is None:
            self.reason = f"nothing can reach Voxa: no running {APP_ID}, no Flatpak Voxa, no gapplication"
            return False
        try:
            done = self.runner(command)
        except (OSError, subprocess.SubprocessError) as exc:
            self.reason = f"{command[0]} failed: {exc}"
            return False
        self.commands_sent.append(command)
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip().splitlines()
            self.reason = f"{action} was refused: {detail[-1] if detail else 'no reason given'}"
            return False
        return True

    def _wait_active(self, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = read_state(self.path)
            if state is not None and state["active"]:
                return state
            time.sleep(1.0)
        return read_state(self.path)

    def ensure_ready(self) -> bool:
        """Voxa must report active: true in listening.json before the first case."""
        state = read_state(self.path)
        if state is not None and state["active"]:
            self.reason = ""
            return True
        if not self._send("start-listening"):
            return False
        self.started_by_us = True
        state = self._wait_active(self.timeout)
        if state is not None and state["active"]:
            self.reason = ""
            return True
        reason = (state or {}).get("reason") or "no listening.json: Voxa is not running or too old to report"
        self.reason = f"{reason} (waited {self.timeout:.0f} s for active: true)"
        return False

    def case_ready(self) -> bool:
        """Recheck before every case: one re-send, and if she stays deaf the run stops."""
        state = read_state(self.path)
        if state is not None and state["active"]:
            self.reason = ""
            return True
        self.reason = (state or {}).get("reason") or "Voxa went inactive mid-run"
        if not self._send("start-listening"):
            return False
        state = self._wait_active(RECHECK_TIMEOUT_SECONDS)
        if state is not None and state["active"]:
            self.reason = ""
            return True
        return False

    def finish(self) -> None:
        """Stop listening only if this run started it - and only if she really is listening."""
        if not self.started_by_us:
            return
        state = read_state(self.path)
        if state is None or not state["active"]:
            return
        self._send("stop-listening")
        self.started_by_us = False
