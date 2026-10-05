from __future__ import annotations

import logging
import os
import subprocess

import voxa.simulation

log = logging.getLogger("voxa.agent.host")

IN_FLATPAK = os.path.exists("/.flatpak-info")


def host_command(command: list[str]) -> list[str]:
    """Wrap a command so it runs on the host when we are inside Flatpak."""
    if IN_FLATPAK:
        return ["flatpak-spawn", "--host", "--directory=/", *command]
    return list(command)


def spawn(command: list[str]) -> None:
    """Fire-and-forget a command, or log it when actions are simulated."""
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return
    subprocess.Popen(  # noqa: S603 - argv is built from validated URLs/app paths
        host_command(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
