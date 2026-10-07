"""Open Spaced Update and really press its buttons, then verify the version."""
from __future__ import annotations

import logging
import subprocess
import time
import urllib.request

from voxa import simulation
from voxa.agent import ui
from voxa.agent.host import host_command, spawn
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.sysupdate import (
    APP,
    APPS_DONE,
    CHECK_BAD,
    CHECK_CLEAN,
    CHECK_FOUND,
    OS_DONE_BAD,
    OS_DONE_GOOD,
    OS_DONE_WARN,
    is_current,
    parse_latest,
    parse_os_release,
    verdict,
)

log = logging.getLogger("voxa.agent.tools.sysupdate")

OS_TIMEOUT = 3600.0
APPS_TIMEOUT = 1800.0
CHECK_TIMEOUT = 300.0

LATEST_URL = "https://api.github.com/repos/crhy/spaced/releases/latest"

_sleep = time.sleep


def fetch(url: str) -> str:
    """Default web fetch: urllib with a Voxa User-Agent, timeout 8."""
    request = urllib.request.Request(url, headers={"User-Agent": "Voxa"})
    return urllib.request.urlopen(request, timeout=8).read().decode()


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


def _read_versions() -> tuple[str, str]:
    """Installed version from /etc/os-release and latest tag from the web ("" on any failure)."""
    installed = parse_os_release(_run(["cat", "/etc/os-release"], 6).stdout)
    try:
        latest = parse_latest(fetch(LATEST_URL))
    except Exception:
        latest = ""
    return installed, latest


def update_system(args: dict[str, str]) -> ToolResult:
    try:
        if _run(["sh", "-c", "command -v spaced-update"], 6).returncode != 0:
            return ToolResult.failure("Spaced Update is not installed on this computer.")

        spawn(["spaced-update"])
        if not ui.wait(APP, "OS Release", 30).get("ok"):
            return ToolResult.failure("Spaced Update did not open.")

        ui.press(APP, "OS Release")
        ui.press(APP, "Check Release")
        _sleep(4)
        if not ui.press(APP, "Update System").get("ok"):
            return ToolResult.failure("I could not press Update System in Spaced Update.")

        os_result = ui.wait_for_any(
            APP, OS_DONE_GOOD + OS_DONE_WARN + OS_DONE_BAD, OS_TIMEOUT, poll=5.0
        )

        apps_result = ""
        if os_result in OS_DONE_GOOD or os_result in OS_DONE_WARN:
            ui.press(APP, "Updates")
            ui.press(APP, "Check for Updates")
            found = ui.wait_for_any(APP, CHECK_CLEAN + CHECK_FOUND + CHECK_BAD, CHECK_TIMEOUT)
            if found in CHECK_CLEAN or found in CHECK_BAD or found == "":
                apps_result = found
            elif found in CHECK_FOUND:
                ui.press(APP, "Select all")
                ui.press(APP, "Install Selected")
                apps_result = ui.wait_for_any(APP, APPS_DONE, APPS_TIMEOUT, poll=5.0)

        installed, latest = _read_versions()
        ok, sentence = verdict(os_result, apps_result, installed, latest)
        if ok:
            return ToolResult.success(sentence)
        return ToolResult.failure(sentence)
    except Exception:
        log.exception("update_system failed")
        return ToolResult.failure("Spaced Update did not finish while I was watching.")


def check_system_version(args: dict[str, str]) -> ToolResult:
    try:
        installed, latest = _read_versions()
        current = is_current(installed, latest)
        if current is True:
            return ToolResult.success(f"This computer is on Spaced Linux {installed}, the latest release.")
        if current is False:
            return ToolResult.success(
                f"This computer is on Spaced Linux {installed}. The latest is {latest}. Say: update Spaced Linux."
            )
        return ToolResult.success(
            f"This computer is on Spaced Linux {installed}. I could not check the latest release online."
        )
    except Exception:
        log.exception("check_system_version failed")
        return ToolResult.failure("I could not read the version from this computer.")


def sysupdate_tools() -> list[Tool]:
    return [
        Tool(
            name="update_system",
            description="Open Spaced Update and press its buttons: the OS release update first, then the app updates.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=update_system,
        ),
        Tool(
            name="check_system_version",
            description="Report the installed and latest Spaced Linux version.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=check_system_version,
        ),
    ]
