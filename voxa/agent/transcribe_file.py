from __future__ import annotations

import os

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".mp4", ".mkv", ".webm")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def pieces(pcm: bytes, seconds: int = 30) -> list[bytes]:
    size = seconds * SAMPLE_RATE * BYTES_PER_SAMPLE
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]


def _stem(source: str) -> str:
    stem = os.path.basename(source)
    dot = stem.rfind(".")
    if dot > 0:
        stem = stem[:dot]
    return stem


def output_path(source: str) -> str:
    folder = os.path.dirname(source)
    name = _stem(source) + ".txt"
    return os.path.join(folder, name) if folder else name


def tidy(texts: list[str]) -> str:
    return " ".join(" ".join(texts).split())


def spoken_done(source: str, words: int) -> str:
    word = "word" if words == 1 else "words"
    return f"I transcribed {_stem(source)}: {words} {word}, saved as a text file next to it."
