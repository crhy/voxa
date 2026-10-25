from __future__ import annotations

import os

from voxa.agent.filematch import normalise

AUDIO_EXTENSIONS = (".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".wav", ".wma", ".aac")
ALL_MUSIC = frozenset({"music", "some music", "my music", "songs", "my songs", "everything", "all my music"})


def pick_tracks(query: str, music_root: str, files: list[str], limit: int = 500) -> list[str]:
    root = music_root.rstrip("/")
    audio = [f for f in files if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)]
    if not audio:
        return []
    if normalise(query) in ALL_MUSIC:
        return sorted(audio)[:limit]
    words = normalise(query).split()
    if not words:
        return []
    best_folder: str | None = None
    best_norm_len: int = 0
    for f in audio:
        rel = os.path.relpath(f, root)
        parts = rel.split(os.sep)[:-1]
        for part in parts:
            n = normalise(part)
            if all(w in n for w in words):
                if best_folder is None or len(n) < best_norm_len:
                    best_folder = part
                    best_norm_len = len(n)
    if best_folder is not None:
        best_norm = normalise(best_folder)
        result = []
        for f in audio:
            rel = os.path.relpath(f, root)
            parts = rel.split(os.sep)[:-1]
            if any(normalise(p) == best_norm for p in parts):
                result.append(f)
        return sorted(result)[:limit]
    matches = [f for f in audio if all(w in normalise(os.path.basename(f)) for w in words)]
    return sorted(matches)[:limit]
