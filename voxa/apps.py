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


_SKIP_EXEC = {"flatpak", "env", "sh", "bash", "python", "python3", "gio", "/usr/bin/flatpak"}
_LEADING_WORDS = re.compile(r"^(?:the|my|a)\s+")


@dataclass(frozen=True, slots=True)
class DesktopApp:
    name: str
    path: str
    keywords: str = ""
    aliases: tuple[str, ...] = ()


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
        apps.setdefault(
            os.path.basename(path),
            DesktopApp(entry["Name"], path, entry.get("Keywords", ""), build_aliases(entry["Name"], path, entry)),
        )
    return sorted(apps.values(), key=lambda app: app.name.casefold())


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.casefold()).split())


def build_aliases(name: str, path: str, entry: dict[str, str]) -> tuple[str, ...]:
    """Sound- and spelling-friendly names for an app, derived from its desktop entry."""
    norm = _normalize(name)
    raw: list[str] = []
    if entry.get("GenericName"):
        raw.append(entry["GenericName"])
    stem = os.path.basename(path)
    if stem.endswith(".desktop"):
        stem = stem[: -len(".desktop")]
    parts = [p for p in stem.split(".") if p]
    if parts:
        raw.append(parts[-1])
        if len(parts) >= 2:
            raw.append(parts[-2])
    exec_line = entry.get("Exec", "")
    if exec_line:
        token = exec_line.split()[0]
        base = os.path.basename(token)
        if token not in _SKIP_EXEC and base not in _SKIP_EXEC:
            raw.append(re.sub(r"-\d+(?:\.\d+)*$", "", base))
    if entry.get("StartupWMClass"):
        raw.append(entry["StartupWMClass"])
    words = name.split()
    if len(words) >= 3:
        raw.append("".join(w[0] for w in words))
    if len(words) == 2:
        raw.append(words[1])
    seen: set[str] = set()
    out: list[str] = []
    for alias in raw:
        key = _normalize(alias)
        if len(key) >= 2 and key != norm and key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def phonetic_key(text: str) -> str:
    """A small pure-Python sound key: how the text sounds, not how it is spelled."""
    s = re.sub(r"[^a-z]", "", text.casefold())
    s = s.replace("ph", "f").replace("ck", "k").replace("wr", "r").replace("gh", "")
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k")
    s = s.replace("q", "k").replace("x", "ks").replace("z", "s").replace("v", "f").replace("y", "i")
    collapsed: list[str] = []
    for ch in s:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    if not collapsed:
        return ""
    return collapsed[0] + "".join(ch for ch in collapsed[1:] if ch not in "aeiou")


def match_app(query: str, apps: list[DesktopApp]) -> DesktopApp | None:
    wanted = _LEADING_WORDS.sub("", _normalize(query), count=1)
    if not wanted:
        return None
    qkey = phonetic_key(wanted)
    if len(qkey) < 2:
        return None
    for app in apps:
        if _normalize(app.name) == wanted or wanted in app.aliases:
            return app
    best: tuple[float, DesktopApp] | None = None
    for app in apps:
        name = _normalize(app.name)
        score = SequenceMatcher(None, wanted, name).ratio()
        wwords, nwords = set(wanted.split()), set(name.split())
        if wwords <= nwords or nwords <= wwords:
            score = max(score, 0.8)
        if score >= 0.75 and (best is None or score > best[0] or (score == best[0] and len(app.name) < len(best[1].name))):
            best = (score, app)
    if best is not None:
        return best[1]
    exact = [app for app in apps if qkey == phonetic_key(app.name) or any(qkey == phonetic_key(a) for a in app.aliases)]
    if exact:
        return min(exact, key=lambda app: len(app.name))
    if len(qkey) >= 3:
        best = None
        for app in apps:
            for candidate in (app.name, *app.aliases):
                score = SequenceMatcher(None, qkey, phonetic_key(candidate)).ratio()
                if score >= 0.82 and (best is None or score > best[0] or (score == best[0] and len(app.name) < len(best[1].name))):
                    best = (score, app)
        return best[1] if best else None
    return None


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
