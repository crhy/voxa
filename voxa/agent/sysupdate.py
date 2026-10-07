"""Pure logic for the Spaced Update flow: version maths, parsing, and the verdict."""
from __future__ import annotations

import json

APP = "Spaced Update"
OS_DONE_GOOD = ("Updated to Spaced Linux", "Your system packages are current")
OS_DONE_WARN = ("Update finished with pending packages", "Packages updated; release marker unchanged")
OS_DONE_BAD = ("System update failed", "Authentication cancelled")
CHECK_CLEAN = ("You\u2019re up to date", "You're up to date", "Everything is current")
CHECK_FOUND = ("available",)
CHECK_BAD = ("Could not check for updates", "Update check failed")
APPS_DONE = ("Updates installed", "Update failed", "Authentication cancelled")


def version_key(version: str) -> tuple:
    """Sort key for a month.year.patch version: "v10.26.1" -> (26, 10, 1)."""
    text = version.strip().lstrip("vV")
    parts = [int(p) for p in text.split(".") if p.strip().isdigit()]
    if not parts:
        return ()
    month = parts[0]
    year = parts[1] if len(parts) >= 2 else 0
    patch = parts[2] if len(parts) >= 3 else 0
    return (year, month, patch)


def parse_os_release(text: str) -> str:
    """Return VERSION_ID without quotes from os-release text, or ""."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("VERSION_ID"):
            _, _, value = stripped.partition("=")
            return value.strip().strip("\"'")
    return ""


def parse_latest(json_text: str) -> str:
    """Return tag_name without a leading "v" from GitHub JSON, or "" (bad JSON included)."""
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    tag = data.get("tag_name")
    if not isinstance(tag, str):
        return ""
    return tag.strip().lstrip("vV")


def is_current(installed: str, latest: str) -> bool | None:
    """True when installed >= latest; None when either side is unknown."""
    if not installed or not latest:
        return None
    return version_key(installed) >= version_key(latest)


def verdict(os_result: str, apps_result: str, installed: str, latest: str) -> tuple[bool, str]:
    """Decide whether the update truly succeeded, in strict priority order."""
    current = is_current(installed, latest)
    if os_result == "Authentication cancelled":
        return (False, "The update was cancelled at the password prompt. Nothing was changed.")
    if os_result == "System update failed":
        return (False, "The system update failed. Spaced Update is showing the details.")
    if os_result == "":
        return (False, "The system update did not finish while I was watching. Spaced Update is still open.")
    if current is False:
        return (
            False,
            f"The update ran, but this computer is on {installed} and the latest Spaced Linux is {latest}. "
            "Spaced Update shows what is holding it back.",
        )
    if os_result in OS_DONE_WARN:
        return (
            False,
            f"The update finished, but some packages are still pending. This computer is on {installed}. "
            "Spaced Update lists them.",
        )
    if apps_result in ("Update failed", "Authentication cancelled", "") or apps_result in CHECK_BAD:
        suffix = ", the latest release" if current else ""
        return (
            False,
            f"Spaced Linux is on {installed}{suffix}, but the app updates did not complete. "
            "Spaced Update is showing why.",
        )
    if current is True:
        return (True, f"Spaced Linux is up to date: version {installed}, the latest release. Your apps are up to date too.")
    installed_part = "; this computer is on " + installed if installed else ""
    return (True, f"The updates finished. I could not check the latest release number online{installed_part}.")
