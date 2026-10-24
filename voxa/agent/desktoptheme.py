from __future__ import annotations

import configparser
import re

_DROP = {"spaced", "theme", "the", "desktop"}
_MAP = {"win": "windows", "seven": "7", "eleven": "11"}

# split glued words: a space between a letter and a digit (either order), and
# before "aero" when it is not at the start of a word.
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_BEFORE_AERO = re.compile(r"(?<=.)(?=aero)")


def theme_words(name: str) -> list[str]:
    lowered = name.lower().replace(".", " ")
    words: list[str] = []
    for raw in lowered.split():
        for piece in _BEFORE_AERO.sub(" ", _LETTER_DIGIT.sub(" ", raw)).split():
            if piece in _DROP:
                continue
            words.append(_MAP.get(piece, piece))
    return words


def find_theme(spoken: str, themes: list[dict]) -> dict | None:
    spoken_words = theme_words(spoken)
    if not spoken_words:
        return None
    best: dict | None = None
    best_extra = 0
    for theme in themes:
        name_words = theme_words(theme["name"])
        missing = False
        for word in spoken_words:
            if word not in name_words:
                missing = True
                break
        if missing:
            continue
        extra = len(name_words) - len(set(spoken_words) & set(name_words))
        if best is None or extra < best_extra:
            best = theme
            best_extra = extra
    return best


def metatheme_keys(index_theme_text: str) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(index_theme_text)
    section = "X-GNOME-Metatheme"
    if not parser.has_section(section):
        return {}
    wanted = ("GtkTheme", "MetacityTheme", "IconTheme", "CursorTheme")
    return {key: parser.get(section, key) for key in wanted if parser.has_option(section, key)}
