"""Accessibility on demand: the desktop a11y bus is on only while Voxa is operating a program.

Voxa's app-control helper (`voxa/agent/uihost_script.py`) speaks AT-SPI over D-Bus, which answers only while
`org.a11y.Status` / `IsEnabled` is true on the session bus. rhY wants that switch off on the desktop, so every
app-control entry point in `voxa/agent/ui.py` wraps its helper call in this scope: read the value that is there
now, turn it on when it is off, do the action, and put the previous value back on the way out - also when the
action raises. The switch is only ever on for the length of one action.

Consent is decided by the caller (see `voxa/agent/policy.py`); `consent=False` means the scope touches nothing at
all. The D-Bus calls go through an injectable bus object, so tests use a fake and never reach a real desktop.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Protocol

from voxa import simulation
from voxa.agent.host import host_command

log = logging.getLogger("voxa.agent.a11y_scope")

SERVICE = "org.a11y.Bus"
PATH = "/org/a11y/bus"
STATUS = "org.a11y.Status"
PROPERTY = "IsEnabled"

# Spoken when Voxa takes control of another program; the user can stop it straight away (see AccessibilityScope.stop).
TAKING_CONTROL = "Voxa is operating this program for you."


class AccessibilityBus(Protocol):
    """The two D-Bus calls the scope needs. `GdbusBus` is the real one; tests pass an object of this shape."""

    def is_enabled(self) -> bool: ...

    def set_enabled(self, value: bool) -> bool:
        """True when the value was applied; False when the bus refused or is missing."""
        ...


class GdbusBus:
    """The real a11y bus, reached with gdbus so the Voxa process needs no PyGObject of its own."""

    def _call(self, method: str, *args: str) -> tuple[bool, str]:
        argv = [
            "gdbus",
            "call",
            "--session",
            "--dest",
            SERVICE,
            "--object-path",
            PATH,
            "--method",
            method,
            *args,
        ]
        if simulation.actions_simulated():
            log.info("simulated action: would run %s", argv)
            return True, ""
        try:
            proc = subprocess.run(host_command(argv), capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            return False, ""
        output = proc.stdout or ""
        # gdbus reports a refused or missing call on stdout rather than in the exit code.
        return proc.returncode == 0 and "Error" not in output and "ERROR" not in output, output

    def is_enabled(self) -> bool:
        ok, output = self._call("org.freedesktop.DBus.Properties.Get", STATUS, PROPERTY)
        return ok and re.search(r"<\s*(?:true|1)\b", output) is not None

    def set_enabled(self, value: bool) -> bool:
        ok, _ = self._call("org.freedesktop.DBus.Properties.Set", STATUS, PROPERTY, "<true>" if value else "<false>")
        return ok


class AccessibilityScope:
    """Use as `with AccessibilityScope(bus) as scope:`; the switch is back as it was when the block ends.

    `scope.stop()` restores it early, which is the instant way to end Voxa's control of the program.
    """

    def __init__(
        self,
        bus: AccessibilityBus | None = None,
        *,
        consent: bool = True,
        notice=None,
        subject: str = "",
    ) -> None:
        self.bus: AccessibilityBus = bus if bus is not None else GdbusBus()
        self.consent = consent
        self.notice = notice
        self.subject = subject
        self.previous: bool | None = None
        self.took_control = False

    def __enter__(self) -> AccessibilityScope:
        if not self.consent:
            return self
        try:
            self.previous = self.bus.is_enabled()
        except Exception:
            log.warning("cannot read the accessibility bus", exc_info=True)
            return self
        if self.previous:
            # The user already left accessibility on: it is theirs to keep, and not ours to switch off.
            return self
        try:
            self.took_control = bool(self.bus.set_enabled(True))
        except Exception:
            log.warning("cannot turn the accessibility bus on", exc_info=True)
            return self
        if self.took_control:
            self._announce()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.restore()
        return False

    def restore(self) -> None:
        """Put the previous value back. Safe to call twice, and from `stop()` as well as on the way out."""
        if not self.took_control:
            return
        self.took_control = False
        try:
            self.bus.set_enabled(bool(self.previous))
        except Exception:
            log.warning("could not restore the accessibility bus", exc_info=True)

    def stop(self) -> None:
        self.restore()

    def _announce(self) -> None:
        message = f"{TAKING_CONTROL} {self.subject}" if self.subject else TAKING_CONTROL
        log.info("%s", message)
        if self.notice is None:
            return
        try:
            self.notice(message)
        except Exception:
            log.warning("the accessibility notice failed", exc_info=True)
