"""Find a JavaScript runtime for yt-dlp, including from inside the Flatpak sandbox."""

from __future__ import annotations

import os
import shutil
import subprocess

from voxa.agent.host import IN_FLATPAK

RUNTIMES = ("deno", "node", "bun")  # yt-dlp's names; preference order

WRAPPED_IN_FLATPAK = {"node": "/app/bin/voxa-host-node"}  # only node is wrapped so far

_cache: dict[tuple[bool, object, object], tuple[str, str] | None] = {}


def _host_runtime(which=shutil.which) -> str | None:
    """Ask the host (through flatpak-spawn) which runtime it has."""
    cmd = [
        "flatpak-spawn",
        "--host",
        # flatpak-spawn changes to the caller's directory on the host; a sandbox-only path makes it fail.
        "--directory=/",
        "sh",
        "-c",
        # A non-login host shell has a short PATH; Node.js is often a per-user install.
        'PATH="$HOME/.local/bin:$HOME/.deno/bin:$HOME/.bun/bin:$HOME/.nvm/current/bin:/usr/local/bin:$PATH"; '
        "command -v deno || command -v node || command -v bun",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = proc.stdout.strip()
    if not found:
        return None
    name = os.path.basename(found)
    return name if name in RUNTIMES else None


def find_runtime(which=shutil.which, in_flatpak: bool | None = None, host_has=None) -> tuple[str, str] | None:
    """Return (runtime name, path) for the first usable JavaScript runtime, or None."""
    flatpak = IN_FLATPAK if in_flatpak is None else in_flatpak
    key = (flatpak, which, host_has)
    if key in _cache:
        return _cache[key]

    if not flatpak:
        for name in RUNTIMES:
            path = which(name)
            if path:
                result = (name, os.path.abspath(path))
                _cache[key] = result
                return result
        _cache[key] = None
        return None

    # Inside Flatpak the sandbox has none of them, so ask the host.
    found = host_has() if host_has is not None else _host_runtime()
    if found in WRAPPED_IN_FLATPAK:
        result = (found, WRAPPED_IN_FLATPAK[found])
        _cache[key] = result
        return result
    _cache[key] = None
    return None


def ytdlp_options() -> dict:
    """yt-dlp options pointing at the JavaScript runtime, or {} when none was found."""
    found = find_runtime()
    if found is None:
        return {}
    name, path = found
    return {"js_runtimes": {name: {"path": path}}}


def missing_runtime_message() -> str:
    return (
        "Playing YouTube directly needs Node.js or Deno installed. "
        "Install one, or I'll use the browser instead."
    )
