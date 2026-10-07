from __future__ import annotations

import logging
import re
import time

from .apps import list_apps

_COMMAND_VERBS = (
    "open",
    "close",
    "play",
    "pause",
    "type",
    "search",
    "switch to",
    "start dictating",
    "stop dictating",
)

MAX_NAME_CHARS = 40
CACHE_SECONDS = 600.0

_WORDISH = re.compile(r"[A-Za-z][A-Za-z'-]*\Z")

_cache: tuple[float, list[str]] | None = None


def build_hint(
    app_names: list[str],
    wake_word: str,
    extra: list[str] = (),
    max_chars: int = 600,
) -> str:
    """A natural sentence nudging Whisper towards this machine's vocabulary.

    Lists the wake word, the command verbs, the fixed Spaced product names
    (kept whenever they fit within ``max_chars``), then the app names de-duplicated
    (case-insensitively), shortest first, until ``max_chars``; names longer
    than 40 characters are skipped.
    """
    wake = wake_word.strip().capitalize()
    base = f"{wake}. Commands: {', '.join(_COMMAND_VERBS)}."
    products = " Products: Spaced Linux, Spaced Update, Spaced Bazaar."
    if len(base) + len(products) <= max_chars:
        base += products
    seen: set[str] = set()
    names: list[str] = []
    for name in [*app_names, *extra]:
        name = name.strip()
        if not name or len(name) > MAX_NAME_CHARS:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    names.sort(key=lambda name: (len(name), name.casefold()))
    chosen: list[str] = []
    for name in names:
        trial = f"{base} Apps: {', '.join([*chosen, name])}."
        if len(trial) > max_chars:
            break
        chosen.append(name)
    if chosen:
        return f"{base} Apps: {', '.join(chosen)}."
    return base


def refresh_app_names() -> list[str]:
    """App names (plus word-like aliases) from :func:`voxa.apps.list_apps`.

    Cached for 10 minutes so the hint can be rebuilt cheaply; returns an
    empty list on any error.
    """
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < CACHE_SECONDS:
        return list(_cache[1])
    try:
        apps = list_apps()
    except Exception as exc:  # noqa: BLE001 - vocabulary boundary, never fatal
        logging.debug("refresh_app_names failed: %s", exc)
        return []
    names: list[str] = []
    seen: set[str] = set()
    for app in apps:
        wordish = [app.name.strip()]
        wordish += [alias.strip() for alias in app.aliases if _WORDISH.match(alias.strip())]
        for candidate in wordish:
            if not candidate or len(candidate) > MAX_NAME_CHARS:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(candidate)
    _cache = (now, names)
    return list(names)
