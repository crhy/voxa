"""Find and launch the desktop applications listed in the host's app menu."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher

from .simulation import actions_simulated

log = logging.getLogger(__name__)

IN_FLATPAK = os.path.exists("/.flatpak-info")
_OPEN_VERB = re.compile(r"^\s*(?:please\s+)?(?:open|launch|start|run)\s+(?:up\s+)?(?:the\s+)?(?P<name>.+?)\s*[.!?]*\s*$", re.I)
_SKIP_IDS = {"io.github.crhy.voxa.desktop"}

_LIST_SCRIPT = r'''
for d in "$HOME/.local/share/applications" "$HOME/.local/share/flatpak/exports/share/applications" \
         /var/lib/flatpak/exports/share/applications /usr/local/share/applications /usr/share/applications; do
  for f in "$d"/*.desktop; do
    [ -f "$f" ] && { echo "@@@$f"; cat "$f"; }
  done
done
'''


@dataclass(frozen=True, slots=True)
class DesktopApp:
    name: str
    path: str
    keywords: str = ""


def parse_open_command(prompt: str) -> str | None:
    """The app name from "open X" / "launch X", or None when it isn't such a command."""
    match = _OPEN_VERB.match(prompt)
    return match.group("name").strip() if match else None


def parse_desktop_dump(dump: str) -> list[DesktopApp]:
    apps: dict[str, DesktopApp] = {}
    for chunk in dump.split("@@@")[1:]:
        path, _, body = chunk.partition("\n")
        path = path.strip()
        if os.path.basename(path) in _SKIP_IDS:
            continue
        entry: dict[str, str] = {}
        in_entry = False
        for line in body.splitlines():
            if line.startswith("["):
                in_entry = line.strip() == "[Desktop Entry]"
            elif in_entry and "=" in line:
                key, _, value = line.partition("=")
                entry.setdefault(key.strip(), value.strip())
        if entry.get("Type", "Application") != "Application" or not entry.get("Name"):
            continue
        if entry.get("NoDisplay", "").lower() == "true" or entry.get("Hidden", "").lower() == "true":
            continue
        # earlier directories (the user's own) win over system ones with the same file name
        apps.setdefault(os.path.basename(path), DesktopApp(entry["Name"], path, entry.get("Keywords", "")))
    return sorted(apps.values(), key=lambda app: app.name.casefold())


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.casefold()).split())


def match_app(query: str, apps: list[DesktopApp]) -> DesktopApp | None:
    wanted = _normalize(query)
    if not wanted:
        return None
    best: tuple[float, DesktopApp | None] = (0.0, None)
    for app in apps:
        name = _normalize(app.name)
        if name == wanted:
            return app
        score = SequenceMatcher(None, wanted, name).ratio()
        if wanted in name.split() or name in wanted.split():
            score = max(score, 0.8)
        if score > best[0]:
            best = (score, app)
    return best[1] if best[0] >= 0.75 else None


def _host(command: list[str]) -> list[str]:
    return ["flatpak-spawn", "--host", *command] if IN_FLATPAK else command


def list_apps() -> list[DesktopApp]:
    result = subprocess.run(_host(["sh", "-c", _LIST_SCRIPT]), capture_output=True, text=True, timeout=15, check=True)
    return parse_desktop_dump(result.stdout)


def launch(app: DesktopApp) -> None:
    if actions_simulated():
        log.info("simulated action: would launch %s", app.name)
        return
    subprocess.Popen(  # noqa: S603 - fixed argv, path comes from the host's own menu
        _host(["gio", "launch", app.path]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
