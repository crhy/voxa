"""Install and remove Flatpak apps through Spaced Bazaar's buttons."""
from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.request

from voxa import simulation
from voxa.agent import ui
from voxa.agent.appstore import (
    BAZAAR_APP,
    BAZAAR_ID,
    CONFIRM_LABELS,
    INSTALL_TIMEOUT,
    choose,
    clean_query,
    install_label,
    parse_hits,
    spoken_choice,
    valid_app_id,
)
from voxa.agent.filematch import best_match
from voxa.agent.host import host_command, spawn
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.appstore")

SEARCH_URL = "https://flathub.org/api/v2/search"
UNINSTALL_TIMEOUT = 120.0

_sleep = time.sleep
_clock = time.monotonic


def search(query: str) -> str:
    """Default Flathub search: POST the query as JSON, return the response text."""
    request = urllib.request.Request(
        SEARCH_URL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Voxa"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def _run(command: list[str], timeout: float, check: bool = False) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _installed(app_id: str) -> bool:
    return _run(["flatpak", "info", app_id], 15).returncode == 0


def install_app(args: dict[str, str]) -> ToolResult:
    try:
        name = args.get("name", "")
        query, popular = clean_query(name)
        if not query:
            return ToolResult.failure("What should I install?")

        try:
            hits = parse_hits(search(query))
        except Exception:
            log.exception("app store search failed")
            return ToolResult.failure("I could not reach the app store. Check the internet connection.")

        hit = choose(query, hits, popular)
        if hit is None or not valid_app_id(hit.get("app_id", "")):
            spawn(["flatpak", "run", BAZAAR_ID, "--search-for", query])
            return ToolResult.failure(f"I could not find {query} in the app store. Spaced Bazaar is open on the search.")

        app_id = hit["app_id"]
        app_name = (hit.get("name") or "").strip()
        label = install_label(app_name)

        if _installed(app_id):
            spawn(["flatpak", "run", BAZAAR_ID, f"appstream://{app_id}"])
            return ToolResult.success(f"{app_name} is already installed.")

        spawn(["flatpak", "run", BAZAAR_ID, f"appstream://{app_id}"])
        if not ui.wait(BAZAAR_APP, label, 40).get("ok"):
            return ToolResult.failure(f"Spaced Bazaar did not show {app_name}. It is open; you can install it there.")

        if not ui.press(BAZAAR_APP, label).get("ok"):
            return ToolResult.failure(f"I could not press Install for {app_name} in Spaced Bazaar.")

        state: dict = {}
        pressed: set[str] = set()
        lowered = {label.casefold() for label in CONFIRM_LABELS}
        deadline = _clock() + INSTALL_TIMEOUT
        installed = False
        while True:
            if _installed(app_id):
                installed = True
                break
            ui.announce_password_once(state)
            for item in ui.items(BAZAAR_APP):
                item_name = (item.get("name") or "").strip()
                if (
                    item.get("showing")
                    and item.get("enabled")
                    and item_name.casefold() in lowered
                    and item_name.casefold() not in pressed
                ):
                    ui.press(BAZAAR_APP, item_name)
                    pressed.add(item_name.casefold())
            if _clock() >= deadline:
                break
            _sleep(3)

        if installed:
            return ToolResult.success(f"Installed {spoken_choice(hit)}. Say: open {app_name}.")
        return ToolResult.failure(f"{app_name} is not installed yet. Spaced Bazaar is open on its page; check it there.")
    except Exception:
        log.exception("install_app failed")
        return ToolResult.failure("Spaced Bazaar did not finish while I was watching.")


def uninstall_app(args: dict[str, str]) -> ToolResult:
    try:
        name = args.get("name", "")
        listing = _run(["flatpak", "list", "--app", "--columns=application,name"], 15).stdout
        apps = []
        for line in listing.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            app_id = parts[0].strip()
            app_name = parts[1].strip() if len(parts) > 1 else app_id
            if app_id:
                apps.append((app_id, app_name))
        names = [app_name for _, app_name in apps]
        match = best_match(name, names)
        if match is None:
            return ToolResult.failure(f"I could not find an installed app called {name}.")
        app_id, app_name = next((pair for pair in apps if pair[1] == match), (match, match))

        spawn(["flatpak", "run", BAZAAR_ID, f"appstream://{app_id}"])
        if not ui.wait(BAZAAR_APP, "Uninstall Application", 40).get("ok"):
            return ToolResult.failure(f"Spaced Bazaar did not show {app_name}. It is open; you can remove it there.")

        ui.press(BAZAAR_APP, "Uninstall Application")
        ui.wait(BAZAAR_APP, "Remove", 15)
        ui.press(BAZAAR_APP, "Remove", "button")

        state: dict = {}
        deadline = _clock() + UNINSTALL_TIMEOUT
        gone = False
        while True:
            if not _installed(app_id):
                gone = True
                break
            ui.announce_password_once(state)
            if _clock() >= deadline:
                break
            _sleep(2)

        if gone:
            return ToolResult.success(f"Removed {app_name}. Your data for it was kept.")
        return ToolResult.failure(f"I could not remove {app_name}. Spaced Bazaar is open on its page.")
    except Exception:
        log.exception("uninstall_app failed")
        return ToolResult.failure("Spaced Bazaar did not finish while I was watching.")


def appstore_tools() -> list[Tool]:
    return [
        Tool(
            name="install_app",
            description="Find an app on Flathub, open it in Spaced Bazaar, press Install, and check it is installed.",
            parameters={"name": "the app to install"},
            risk=RiskLevel.REVERSIBLE,
            handler=install_app,
            required=("name",),
            timeout_seconds=INSTALL_TIMEOUT,
        ),
        Tool(
            name="uninstall_app",
            description="Remove an installed Flatpak app through Spaced Bazaar's buttons.",
            parameters={"name": "the installed app to remove"},
            risk=RiskLevel.REVERSIBLE,
            handler=uninstall_app,
            required=("name",),
            timeout_seconds=UNINSTALL_TIMEOUT,
        ),
    ]
