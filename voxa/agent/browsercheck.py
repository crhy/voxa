"""Detect whether the host computer has a real web browser.

On a fresh Spaced install there is no browser, so `xdg-open https://...` hands the
page to the text editor (Pluma) and shows raw HTML. Voxa asks the host what would
open an https link and only treats it as a browser when the handler is actually
one (its name or its .desktop Categories say so).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

# How long a detected answer is trusted before asking the host again.
_CACHE_SECONDS = 60.0

# Where the desktop entry files live (system, user, and the two Flatpak exports).
_DESKTOP_DIRS = (
    "/usr/share/applications",
    os.path.expanduser("~/.local/share/applications"),
    "/var/lib/flatpak/exports/share/applications",
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
)

# Words in a .desktop file name that mean "this is a web browser".
_BROWSER_WORDS = (
    "browser",
    "firefox",
    "chrome",
    "chromium",
    "brave",
    "edge",
    "vivaldi",
    "opera",
    "safari",
    "konqueror",
    "qutebrowser",
    "elysium",
    "zen",
)

# Cached (answer, monotonic time it was computed).
_cache: tuple[bool, float] | None = None


def clear_cache() -> None:
    """Forget the cached answer (used by tests and after an install)."""
    global _cache
    _cache = None


def _handler_names(run: Callable[[list[str]], tuple[int, str]]) -> list[str]:
    """Desktop entry names the host would use for an https link."""
    names: list[str] = []
    for command in (
        ["xdg-mime", "query", "default", "x-scheme-handler/https"],
        ["xdg-settings", "get", "default-web-browser"],
    ):
        returncode, stdout = run(command)
        if returncode != 0:
            continue
        for line in stdout.splitlines():
            name = line.strip()
            if name.endswith(".desktop"):
                names.append(name)
    return names


def _categories(run: Callable[[list[str]], tuple[int, str]], name: str) -> str:
    """Concatenated Categories lines for a desktop entry found under the known dirs."""
    base = name.rsplit("/", 1)[-1]
    paths = [os.path.join(directory, base) for directory in _DESKTOP_DIRS]
    _, stdout = run(["grep", "-h", "^Categories", *paths])
    return stdout


def _suggests_browser(name: str, categories: str) -> bool:
    base = name.rsplit("/", 1)[-1].casefold()
    if any(word in base for word in _BROWSER_WORDS):
        return True
    return "webbrowser" in categories.casefold()


def _detect(run: Callable[[list[str]], tuple[int, str]]) -> bool:
    for name in _handler_names(run):
        if _suggests_browser(name, _categories(run, name)):
            return True
    return False


def has_web_browser(run: Callable[[list[str]], tuple[int, str]]) -> bool:
    """True when the host's default https handler is a real web browser.

    `run(command)` returns `(returncode, stdout)`. The answer is cached for 60 s.
    """
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[1] < _CACHE_SECONDS:
        return _cache[0]
    answer = _detect(run)
    _cache = (answer, now)
    return answer
