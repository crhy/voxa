from __future__ import annotations

import logging
import os
import shlex
import subprocess

import voxa.simulation

log = logging.getLogger("voxa.agent.host")

IN_FLATPAK = os.path.exists("/.flatpak-info")

# How long spawn() watches for a command that fails straight away.
_QUICK_FAILURE_SECONDS = 0.25


def host_command(command: list[str]) -> list[str]:
    """Wrap a command so it runs on the host when we are inside Flatpak."""
    if IN_FLATPAK:
        return ["flatpak-spawn", "--host", "--directory=/", *command]
    return list(command)


def spawn(command: list[str]) -> bool:
    """Start a host command without waiting for it; False when it could not be started.

    Most callers start a program that keeps running (a player, a browser), so this never
    waits for the command to finish and never reads its output: a pipe would be inherited
    by that program and block here until it closed. Only a command that fails at once is
    reported.
    """
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return True
    opener = os.environ.get("VOXA_URL_OPENER", "").strip()
    if opener and len(command) == 2 and command[0] == "xdg-open" and command[1].startswith(("http://", "https://")):
        # Test runs: web pages go to a browser of their own, so hundreds of spoken test
        # prompts do not pile tabs into the user's browser and can be closed afterwards.
        command = [*shlex.split(opener), command[1]]
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is built from validated URLs/app paths
            host_command(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.warning("host command failed: %s", exc)
        return False
    try:
        returncode = process.wait(timeout=_QUICK_FAILURE_SECONDS)
    except subprocess.TimeoutExpired:
        return True
    if returncode != 0:
        log.warning("host command %s exited with %s", command[0], returncode)
        return False
    return True
