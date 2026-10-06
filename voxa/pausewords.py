"""Voice phrases that pause or resume the assistant (pure Python, no GTK)."""

from __future__ import annotations

import re

# The misspellings the app's wake-word matcher already accepts for "voxa".
_WAKE_ALIASES = ("voxa", "vox a", "vaxa", "voxo", "boxa")

_PAUSE_PHRASES = frozenset(
    {
        "pause listening",
        "stop listening",
        "take a break",
        "go to sleep",
        "voxa pause",
        "pause voxa",
    }
)


def _normalize(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def _wake_aliases(wake_word: str) -> tuple[str, ...]:
    """The wake word plus the misspellings the app accepts, longest first."""
    wake = _normalize(wake_word)
    aliases = [wake] if wake else []
    if wake == "voxa":
        aliases += [alias for alias in _WAKE_ALIASES if alias != "voxa"]
    return tuple(sorted(dict.fromkeys(aliases), key=len, reverse=True))


def is_pause_request(text: str, media_playing: bool) -> bool:
    """True when the sentence asks the assistant to pause its listening.

    A bare "pause" counts only when nothing is playing, so it stays a media
    control while music runs. An optional leading wake word is allowed.
    """
    phrase = _normalize(text)
    if phrase in _PAUSE_PHRASES:
        return True
    if phrase == "pause":
        return not media_playing
    for alias in _wake_aliases("voxa"):
        if phrase.startswith(alias + " "):
            rest = phrase[len(alias) + 1 :]
            if rest in _PAUSE_PHRASES:
                return True
            return rest == "pause" and not media_playing
    return False


def resume_request(text: str, wake_word: str) -> tuple[bool, str]:
    """(True, rest) when the sentence begins with the wake word.

    ``rest`` is whatever follows the wake word ("" when it was said alone).
    The misspellings the app already accepts for "voxa" are allowed too.
    """
    phrase = _normalize(text)
    for alias in _wake_aliases(wake_word):
        if phrase == alias:
            return (True, "")
        if phrase.startswith(alias + " "):
            return (True, phrase[len(alias) + 1 :])
    return (False, phrase)
