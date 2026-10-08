"""Internet radio: a small catalogue of stations, found by name or by genre."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SOMA_URL = "https://somafm.com/channels.json"
SOMA_STREAM = "https://ice1.somafm.com/{id}-128-mp3"


@dataclass(frozen=True, slots=True)
class Station:
    name: str
    genre: str
    description: str
    stream: str
    listeners: int = 0


# Used when SomaFM's channel list cannot be fetched; the two independent stations are always present.
EXTRA = (
    Station("Bob's Ska Radio", "ska reggae rocksteady punk", "Ska, rocksteady and reggae around the clock.",
            "http://51.255.235.165:5528/stream"),
    Station("Cowpoke Radio", "country western cowboy", "Classic country and western.",
            "http://pavo.prostreaming.net:8034/stream"),
)
_SOMA_BUILTIN = (
    ("groovesalad", "Groove Salad", "ambient electronic chill", "A nicely chilled plate of ambient and downtempo beats."),
    ("dronezone", "Drone Zone", "ambient space", "Atmospheric textures with minimal beats."),
    ("indiepop", "Indie Pop Rocks", "indie pop rock", "New and classic indie pop tracks."),
    ("spacestation", "Space Station Soma", "electronic ambient space", "Tune in, turn on, space out."),
    ("secretagent", "Secret Agent", "lounge jazz soundtrack", "The soundtrack for your stylish, mysterious life."),
    ("lush", "Lush", "electronic vocal chill", "Sensuous and mellow vocals, mostly female, with an electronic influence."),
    ("deepspaceone", "Deep Space One", "ambient space electronic", "Deep ambient electronic and space music."),
    ("beatblender", "Beat Blender", "electronic house", "A late night blend of deep-house and downtempo chill."),
    ("bootliquor", "Boot Liquor", "americana country roots", "Americana roots music for cowhands, cowpokes and cowtippers."),
    ("u80s", "Underground 80s", "eighties synthpop new wave", "Early 80s UK synthpop and a bit of new wave."),
)
BUILTIN = tuple(Station(t, g, d, SOMA_STREAM.format(id=i)) for i, t, g, d in _SOMA_BUILTIN) + EXTRA


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower().replace("'", ""))).strip()


def parse_soma(json_text: str, top: int = 10) -> list[Station]:
    """The ``top`` most listened SomaFM channels from its channel list; [] on anything unexpected."""
    try:
        channels = json.loads(json_text)["channels"]
        stations = [
            Station(c["title"], c.get("genre", "").replace("|", " "), c.get("description", ""),
                    SOMA_STREAM.format(id=c["id"]), int(c.get("listeners") or 0))
            for c in channels
        ]
    except (ValueError, KeyError, TypeError):
        return []
    return sorted(stations, key=lambda s: -s.listeners)[:top]


def catalogue(fetch=None) -> list[Station]:
    """SomaFM's current top ten (when reachable) plus the fixed extras; the built-in list otherwise."""
    if fetch is not None:
        try:
            soma = parse_soma(fetch(SOMA_URL))
        except Exception:  # noqa: BLE001 - no network: the built-in list is the answer
            soma = []
        if soma:
            return soma + list(EXTRA)
    return list(BUILTIN)


def clean_request(spoken: str) -> str:
    """"some ska radio" -> "ska"; "the groove salad station" -> "groove salad"."""
    text = norm(spoken)
    text = re.sub(r"^(?:the|some|a|me|my)\s+", "", text)
    text = re.sub(r"\s*(?:internet\s+)?(?:radio(?:\s+station)?|station|channel|fm)$", "", text)
    return re.sub(r"^(?:some|the)\s+", "", text).strip()


def find(spoken: str, stations: list[Station]) -> Station | None:
    """By name first (exact, then all words), then by a genre word. None when nothing fits."""
    wanted = clean_request(spoken)
    if not wanted:
        return stations[0] if stations else None
    for station in stations:
        if norm(station.name) == wanted or clean_request(station.name) == wanted:
            return station
    words = wanted.split()
    for station in stations:
        name = norm(station.name)
        if all(word in name for word in words):
            return station
    for station in stations:
        genre = norm(station.genre)
        if all(word in genre.split() or word in genre for word in words):
            return station
    return None


def is_station(spoken: str, stations: tuple[Station, ...] = BUILTIN) -> bool:
    """True when the words name one of the built-in stations exactly (so "play groove salad" is radio)."""
    wanted = clean_request(spoken)
    return any(wanted and wanted in (norm(s.name), clean_request(s.name)) for s in stations)


def spoken_list(stations: list[Station]) -> str:
    names = [s.name for s in stations]
    if not names:
        return "I have no radio stations."
    listed = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    return f"I have {len(names)} stations: {listed}."


def describe(station: Station) -> str:
    genre = ", ".join(station.genre.split()[:3])
    return f"This is {station.name}: {genre}. {station.description}".strip()
