"""Internet radio: a small catalogue of stations, found by name or by genre."""

from __future__ import annotations

import difflib
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
    ("illstreet", "Illinois Street Lounge", "lounge exotica vintage",
     "Classic bachelor pad, playful exotica and vintage music of tomorrow."),
    ("bossa", "Bossa Beyond", "bossa nova brazilian jazz", "Silky-smooth, laid-back Brazilian-style rhythms."),
    ("christmas", "Christmas Lounge", "christmas holiday lounge",
     "Chilled holiday grooves and classic winter lounge tracks."),
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


def squash(text: str) -> str:
    """norm() with spaces removed and a trailing radio/station/somafm dropped."""
    return re.sub(r"\s*(?:radio|station|somafm|soma)$", "", norm(text)).replace(" ", "")


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
    wanted_key = squash(wanted)
    for station in stations:
        if wanted_key and wanted_key == squash(station.name):
            return station
    best, best_ratio, runner_up = None, 0.0, 0.0
    for station in stations:
        ratio = difflib.SequenceMatcher(None, wanted_key, squash(station.name)).ratio()
        if ratio > best_ratio:
            runner_up, best_ratio, best = best_ratio, ratio, station
        elif ratio > runner_up:
            runner_up = ratio
    if best_ratio >= 0.8 and best_ratio - runner_up > 0.05:
        return best
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


ASK_RADIO = "ask-radio"  # ToolResult.detail prefix: the numbered station list follows, one per line


def numbered(stations: list[Station]) -> list[str]:
    """The stations as a numbered list: ["1. Groove Salad", "2. ...", ...]."""
    return [f"{index}. {station.name}" for index, station in enumerate(stations, 1)]


_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}


def pick(spoken: str, stations: list[Station]) -> Station | None:
    """A number as digits or words ("3", "three", "number three", "the third one") picks a
    station; otherwise fall back to find by name or genre."""
    text = norm(spoken)
    number = None
    digits = re.search(r"\b(\d+)\b", text)
    if digits:
        number = int(digits.group(1))
    else:
        for word, value in _ORDINALS.items():
            if re.search(rf"\b{word}\b", text):
                number = value
                break
        if number is None:
            for word, value in _WORDS.items():
                if re.search(rf"\b{word}\b", text):
                    number = value
                    break
    if number is not None and 1 <= number <= len(stations):
        return stations[number - 1]
    return find(spoken, stations)
