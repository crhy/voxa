"""Press buttons and read text in other programs, through the helper that runs on the host.

Accessibility is off on the desktop by default, so each app-control call here turns the a11y bus on for its own
duration only (`voxa/agent/a11y_scope.py`) and puts it back afterwards. When the accessibility bus cannot reach a
program at all, pressing falls back to Voxa's own screen toolset (`voxa/agent/visual_control.py`).
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import voxa.simulation
from voxa.agent import visual_control
from voxa.agent.a11y_scope import AccessibilityScope
from voxa.agent.host import host_command

log = logging.getLogger("voxa.agent.ui")

SCRIPT = Path(__file__).with_name("uihost_script.py")

say: Callable[[str], None] | None = None
PASSWORD_NOTICE = "Please enter your password."

# Tests set a fake bus here; None means the real one, reached with gdbus.
a11y_bus = None

# Answers that mean "the accessibility bus cannot see this control", so the screen toolset gets a turn. A control
# reported as disabled is the bus's own answer and is never retried by clicking.
_VISUAL_FALLBACK_ERRORS = frozenset({"not found", "program not found", "helper failed", "no output", "bad JSON"})


def _scope(app: str = "", *, announce: bool = False) -> AccessibilityScope:
    """Accessibility on for this one action, restored on the way out whatever the action did."""
    return AccessibilityScope(a11y_bus, notice=_spoken_notice if announce else None, subject=app)


def _spoken_notice(message: str) -> None:
    """Speak the taking-control notice; the scope logs it whether or not there is a speaker."""
    if say is not None:
        say(message)


def call(command: str, *args: str, timeout: float = 15.0, runner=subprocess.run) -> dict:
    """Run the host helper with the given command and return its parsed JSON answer."""
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would run ui helper command %s", command)
        return {"ok": True, "simulated": True}
    try:
        proc = runner(
            host_command(["python3", "-", command, *args]),
            input=SCRIPT.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"ok": False, "error": "helper failed"}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "no output"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "bad JSON"}


def apps(**kw) -> list[dict]:
    with _scope():
        return call("apps", **kw).get("apps") or []


def items(app: str, *roles: str, **kw) -> list[dict]:
    with _scope(app):
        return call("list", app, *roles, **kw).get("items") or []


def press(app: str, label: str, role: str = "", **kw) -> dict:
    """Press a named button: AT-SPI in the a11y scope first, Voxa's own screen toolset second."""
    with _scope(app, announce=True):
        result = call("press", app, label, role, **kw) if role else call("press", app, label, **kw)
    if result.get("ok") or result.get("error") not in _VISUAL_FALLBACK_ERRORS:
        return result
    return visual_control.press_text(app, label, notice=_spoken_notice)


def wait(app: str, label: str, seconds: float, **kw) -> dict:
    with _scope(app):
        return call("wait", app, label, str(seconds), timeout=seconds + 10, **kw)


def text(app: str, **kw) -> list[str]:
    with _scope(app):
        return call("text", app, **kw).get("text") or []


def wait_for_any(
    app: str,
    labels: tuple[str, ...],
    seconds: float,
    poll: float = 3.0,
    sleep=time.sleep,
    clock=time.monotonic,
    **kw,
) -> str:
    """Poll text(app) until one line contains (case-insensitive) one of the labels."""
    state: dict = {}
    deadline = clock() + seconds
    lowered = [label.casefold() for label in labels]
    while True:
        if announce_password_once(state, **kw):
            pass
        for line in text(app, **kw):
            folded = line.casefold()
            for index, needle in enumerate(lowered):
                if needle in folded:
                    return labels[index]
        if clock() >= deadline:
            return ""
        sleep(poll)


def active_window_title(runner=subprocess.run) -> str:
    """Name of the window in front, via xdotool; "" on failure."""
    try:
        proc = runner(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip()


def password_prompt_showing(**kw) -> bool:
    """True when a password/authentication dialogue appears to be up."""
    needles = ("polkit", "authentication-agent", "pkexec", "gksu")
    titles = ("authenticate", "authentication required", "authentication is required")
    for app in apps(**kw):
        name = (app.get("name") or "").casefold()
        windows = app.get("windows") or []
        if any(needle in name for needle in needles) and windows:
            return True
        for window in windows:
            if (window or "").casefold() in titles:
                return True
    return False


def announce_password_once(state: dict, **kw) -> bool:
    """Speak the password notice once per wait while a prompt is up. Never raises."""
    try:
        if state.get("announced"):
            return False
        if not password_prompt_showing(**kw):
            return False
        state["announced"] = True
        if say is not None:
            say(PASSWORD_NOTICE)
        return True
    except Exception:
        return False
