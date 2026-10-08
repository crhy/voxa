"""The wake word as the user set it, for every hint and tip that tells them what to say."""

from __future__ import annotations

import re

DEFAULT = "Voxa"
_current = DEFAULT
_NAMED = re.compile(r"\bVoxa\b")


def set_wake(word: str) -> None:
    """Remember the wake word (shown capitalised: "computer" -> "Computer")."""
    global _current
    word = (word or "").strip()
    _current = (word[:1].upper() + word[1:]) if word else DEFAULT


def wake() -> str:
    return _current


def apply(text: str) -> str:
    """Replace the default wake word in a hint with the user's own."""
    if not text or _current == DEFAULT:
        return text
    return _NAMED.sub(_current, text)
