"""Pure helpers for finding and choosing a Flatpak app on Flathub."""
from __future__ import annotations

import json
import re

BAZAAR_ID = "io.github.crhy.SpacedBazaar"
BAZAAR_APP = "bazaar"
CONFIRM_LABELS = ("Install", "Continue", "Confirm", "Yes", "OK")
INSTALL_TIMEOUT = 1800.0

_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
_POPULARITY = re.compile(r"(?:most popular|most downloaded|highest rated|best|top)")
_SUFFIX = re.compile(
    r"\s+(?:app|application|program|from flathub|from the app store|from spaced bazaar|for me|please)$"
)
_APP_ID = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")


def clean_query(spoken: str) -> tuple[str, bool]:
    """Return (query, popular) from a spoken app request."""
    text = spoken.strip().lower()
    text = _ARTICLE.sub("", text)
    popular = False
    if _POPULARITY.search(text):
        popular = True
        text = _POPULARITY.sub("", text)
    text = _SUFFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, popular


def parse_hits(json_text: str) -> list[dict]:
    """Keep desktop-application hits; a missing type counts. [] on bad JSON."""
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    hits = data.get("hits") or []
    if not isinstance(hits, list):
        return []
    return [
        hit
        for hit in hits
        if isinstance(hit, dict) and hit.get("type", "desktop-application") == "desktop-application"
    ]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def choose(query: str, hits: list[dict], popular: bool) -> dict | None:
    """Pick the best hit for the query, or None when there are no hits."""
    if not hits:
        return None
    wanted = _key(query)
    for hit in hits:
        if _key(hit.get("name", "")) == wanted:
            return hit
        app_id = hit.get("app_id", "")
        if app_id and _key(app_id.split(".")[-1]) == wanted:
            return hit
    if popular:
        return max(hits[:6], key=lambda hit: hit.get("installs_last_month", 0))
    first = hits[0]
    base = first.get("installs_last_month", 0)
    for hit in hits[:3]:
        installs = hit.get("installs_last_month", 0)
        if hit is not first and installs >= 5 * base and installs > base:
            return hit
    return first


def valid_app_id(app_id: str) -> bool:
    """True only for a dotted id of safe characters (no spaces, no shell junk)."""
    return bool(_APP_ID.fullmatch(app_id or ""))


def spoken_choice(hit: dict) -> str:
    """A name plus its summary, or just the name when there is no summary."""
    name = (hit.get("name") or "").strip()
    summary = (hit.get("summary") or "").strip()
    if summary:
        return f"{name}, {summary}"
    return name


def install_label(name: str) -> str:
    """The start of the install button's name on the app's page."""
    return f"Install {name}"
