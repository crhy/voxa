from __future__ import annotations

import difflib
import re

FOLDER_KEYS = {
    "home": "",
    "home directory": "",
    "home folder": "",
    "desktop": "DESKTOP",
    "documents": "DOCUMENTS",
    "downloads": "DOWNLOAD",
    "download": "DOWNLOAD",
    "music": "MUSIC",
    "pictures": "PICTURES",
    "photos": "PICTURES",
    "videos": "VIDEOS",
    "video": "VIDEOS",
    "movies": "VIDEOS",
}

STANDARD_NAMES = {
    "DESKTOP": "Desktop",
    "DOCUMENTS": "Documents",
    "DOWNLOAD": "Downloads",
    "MUSIC": "Music",
    "PICTURES": "Pictures",
    "VIDEOS": "Videos",
}


def resolve_folder(key: str, home: str, reported: str) -> str:
    if key == "":
        return home
    if not reported or reported.rstrip("/") == home.rstrip("/"):
        return f"{home}/{STANDARD_NAMES[key]}"
    return reported


def folder_key(spoken: str) -> str | None:
    s = spoken.strip().lower()
    for prefix in ("my ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    for suffix in (" folder", " directory"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = s.strip()
    if s in FOLDER_KEYS:
        return FOLDER_KEYS[s]
    s = re.sub(r"\s+(folder|directory)$", "", s).strip()
    return FOLDER_KEYS.get(s)


def normalise(name: str) -> str:
    s = name.lower()
    dot = s.rfind(".")
    if dot > 0:
        s = s[:dot]
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def best_match(spoken: str, names: list[str]) -> str | None:
    if not names:
        return None
    spoken_clean = spoken.strip().lower()
    spoken_clean = re.sub(r"\bdot\b", ".", spoken_clean)
    spoken_clean = re.sub(r"\s*\.\s*", ".", spoken_clean)
    spoken_norm = normalise(spoken_clean)
    if not spoken_norm:
        return None
    exact = [n for n in names if normalise(n) == spoken_norm]
    if exact:
        dotted = [n for n in exact if n.lower() == spoken_clean]
        if dotted:
            return dotted[0]
        return min(exact, key=len)
    words = spoken_norm.split()
    contains = [n for n in names if all(w in normalise(n) for w in words)]
    if contains:
        return min(contains, key=lambda n: len(normalise(n)))
    best: str | None = None
    best_ratio = 0.0
    for n in names:
        ratio = difflib.SequenceMatcher(None, spoken_norm, normalise(n)).ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_ratio = ratio
            best = n
    return best


def size_bytes(amount: str, unit: str) -> int:
    n = int(amount)
    if unit.strip().lower().startswith("g"):
        return n * 1024**3
    return n * 1024**2


def spoken_list(names: list[str], limit: int = 3) -> str:
    if not names:
        return ""
    head = names[:limit]
    if len(head) == 1:
        text = head[0]
    else:
        text = ", ".join(head[:-1]) + " and " + head[-1]
    extra = len(names) - len(head)
    if extra > 0:
        text += f" and {extra} more"
    return text
