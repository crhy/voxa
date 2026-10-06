from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

from voxa import apps, simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.applications import open_app

log = logging.getLogger("voxa.agent.tools.windows")

_THE_PREFIX = re.compile(r"^the\s+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Window:
    id: str
    wm_class: str
    title: str


def parse_wmctrl(listing: str) -> list[Window]:
    """Turn `wmctrl -lx` output into Window records.

    Each line is "<id> <desktop> <class> <host> <title...>"; the title may be
    missing entirely or contain spaces.
    """
    windows: list[Window] = []
    for line in listing.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        windows.append(Window(parts[0], parts[2], parts[4] if len(parts) > 4 else ""))
    return windows


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def list_windows() -> list[Window]:
    result = _run(["wmctrl", "-lx"])
    return parse_wmctrl(result.stdout)


def match_windows(name: str, windows: list[Window]) -> list[Window]:
    """Windows whose title or wm_class contains the spoken name."""
    needle = _normalize(name)
    if not needle:
        return []
    matches: list[Window] = []
    for window in windows:
        if "voxa" in window.wm_class.casefold() and "voxa" not in needle:
            continue
        if needle in _normalize(window.title) or needle in _normalize(window.wm_class):
            matches.append(window)
    return matches


def flatpak_app_id(desktop_path: str) -> str | None:
    """The Flatpak app id behind a .desktop file, or None for a non-Flatpak app."""
    if "flatpak/exports/" not in desktop_path:
        return None
    return desktop_path.rsplit("/", 1)[-1].removesuffix(".desktop")


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        capture_output=True,
        text=True,
        timeout=5,
        check=check,
    )


_GENERIC_BROWSER = {"browser", "web browser", "internet browser", "your browser", "voxa's browser", "voxa browser"}


def _close_own_browser(name: str) -> ToolResult | None:
    """"Close the browser" means the browser Voxa opened, never the user's own with all their tabs.

    Returns None when the request is not about a generic "browser" (e.g. "close Brave": the user's choice).
    """
    if name.lower() not in _GENERIC_BROWSER:
        return None
    try:
        from ..browser_session import browser_socket
        from .web import get_session

        session = get_session()
        if browser_socket(session._port) is not None:
            session.close_browser()
            return ToolResult.success("Closing my browser.")
    except Exception as exc:  # noqa: BLE001 - fall through to the honest answer below
        log.debug("could not close the controlled browser: %s", exc)
    return ToolResult.failure("My browser isn't open. Say the browser's name, like “close Brave”, to close yours.")


def close_app(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    own = _close_own_browser(name)
    if own is not None:
        return own
    windows = match_windows(name, list_windows())
    app = apps.match_app(name, apps.list_apps())
    app_id = flatpak_app_id(app.path) if app is not None else None

    if windows:
        for window in windows:
            _run(["wmctrl", "-i", "-c", window.id])
        # A Flatpak app can linger behind a "save changes?" prompt, so kill it too.
        if app_id is not None:
            _run(["flatpak", "kill", app_id], check=False)
        return ToolResult.success(f"Closing {name}.")

    if app_id is not None:
        result = _run(["flatpak", "kill", app_id], check=False)
        if result.returncode == 0:
            return ToolResult.success(f"Closing {app.name}.")

    return ToolResult.failure(f"{name} doesn't seem to be open.")


def close_window(args: dict[str, str]) -> ToolResult:
    _run(["wmctrl", "-c", ":ACTIVE:"])
    return ToolResult.success("Closed.")


def switch_to(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    windows = match_windows(name, list_windows())
    if not windows:
        # "bring up the calculator" should open it when it is not already running.
        return open_app({"name": name})
    _run(["wmctrl", "-i", "-a", windows[0].id])
    return ToolResult.success(f"Switching to {name}.")


def window_tools() -> list[Tool]:
    return [
        Tool(
            name="close_app",
            description="Close an open application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=close_app,
            required=("name",),
        ),
        Tool(
            name="close_window",
            description="Close the window that currently has focus.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=close_window,
            required=(),
        ),
        Tool(
            name="switch_to",
            description="Focus an already open application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=switch_to,
            required=("name",),
        ),
    ]
