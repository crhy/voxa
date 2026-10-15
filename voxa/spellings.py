"""Names Whisper mishears, and how the user taught Voxa to spell them.

"rhY Thornton" sounds like "Rye Thornton", so Whisper writes that and every search finds the wrong
person. Saying "Rhy is spelled R-H-Y" after a miss, or "... Ry Thornton, spelled R-H-Y" inside a
request, teaches the spelling: the misheard words become a variant of the name, later requests are
corrected before they are routed, the name leads Whisper's vocabulary hint, and searches quote it.

Stored in ~/.config/voxa/spellings.json as {"Rhy Thornton": ["ry thornton", "rye thornton"]}; the
name can be edited there to any styling (for example "rhY Thornton").
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from pathlib import Path

_LETTERS = r"(?:[A-Za-z](?:[\s.,-]+|$)){2,}"
_STANDALONE = re.compile(rf"(?:it'?s |the name |that'?s |no,? )?([A-Za-z']+) is spelled ({_LETTERS})", re.I)
_INLINE = re.compile(rf",?\s*(?:which is |that'?s |it'?s )?spelled ({_LETTERS})", re.I)


def path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "voxa" / "spellings.json"


def load() -> dict[str, list[str]]:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(name): [str(v).casefold() for v in variants] for name, variants in data.items()
            if isinstance(variants, list)}


def save(names: dict[str, list[str]]) -> None:
    target = path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        logging.warning("could not save spellings: %s", exc)


def letters(spelled: str) -> str:
    """'R-H-Y' or 'R H Y.' -> 'Rhy'."""
    word = "".join(re.findall(r"[A-Za-z]", spelled))
    return word[:1].upper() + word[1:].lower()


def names(store: dict[str, list[str]] | None = None) -> list[str]:
    return list(load() if store is None else store)


def apply(text: str, store: dict[str, list[str]] | None = None) -> str:
    """Replace every learned variant (whole words, any case) with its name."""
    store = load() if store is None else store
    pairs = sorted(((v, name) for name, vs in store.items() for v in vs), key=lambda p: -len(p[0]))
    for variant, name in pairs:
        text = re.sub(rf"\b{re.escape(variant)}\b", name, text, flags=re.I)
    # New mishearings of a full name: a similar first name before the exact surname.
    for name, variants in store.items():
        first, _, rest = name.partition(" ")
        if not rest:
            continue
        heard_firsts = {first.casefold(), *(v.split()[0] for v in variants if " " in v)}

        def fix(match: re.Match, first: str = first, rest: str = rest, heard_firsts=heard_firsts) -> str:
            word = match.group(1).casefold()
            close = max(difflib.SequenceMatcher(None, word, h).ratio() for h in heard_firsts)
            return f"{first} {rest}" if word[:1] == first[:1].casefold() and close >= 0.5 else match.group(0)

        text = re.sub(rf"\b([A-Za-z']+) {re.escape(rest)}\b", fix, text, flags=re.I)
    return text


def quote(query: str, store: dict[str, list[str]] | None = None) -> str:
    """Quote known names in a web search so it looks for that exact person."""
    for name in names(store):
        query = re.sub(rf'(?<!"){re.escape(name)}(?!")', f'"{name}"', query, flags=re.I)
    return query


def _closest(words: list[str], spelled: str) -> int | None:
    """Index of the word the spelling corrects: similar sound, same first letter preferred."""
    best, best_score = None, 0.0
    for i, word in enumerate(words):
        clean = re.sub(r"[^A-Za-z']", "", word).casefold()
        if not clean:
            continue
        score = difflib.SequenceMatcher(None, clean, spelled.casefold()).ratio()
        score += 0.3 if clean[:1] == spelled[:1].casefold() else 0.0
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= 0.6 else None


def _correct(text: str, spelled: str, store: dict[str, list[str]]) -> str | None:
    """Fix the misheard word in text and learn it, with the next word when there is one."""
    words = text.split()
    i = _closest(words, spelled)
    if i is None:
        return None
    heard = re.sub(r"[^A-Za-z']", "", words[i])
    following = re.sub(r"[^A-Za-z'-]", "", words[i + 1]) if i + 1 < len(words) else ""
    # A full name is learned as a pair, so "rye bread" is never touched.
    name = f"{spelled} {following}" if following[:1].isupper() else spelled
    variant = f"{heard} {following}" if following[:1].isupper() else heard
    if variant.casefold() != name.casefold():
        learned = store.setdefault(name, [])
        if variant.casefold() not in learned:
            learned.append(variant.casefold())
    words[i] = words[i].replace(heard, spelled)
    return " ".join(words)


def teach(text: str, previous: str | None) -> tuple[str, str] | None:
    """Handle a spelling lesson.

    Returns (kind, request): kind "inline" means text corrected itself ("... spelled R-H-Y") and
    request is the corrected text to run; kind "previous" means text only taught a spelling and
    request is the corrected previous request to run again. None when text teaches nothing.
    """
    store = load()
    standalone = _STANDALONE.fullmatch(text.strip(" .!?"))
    if standalone:
        spelled = letters(standalone.group(2))
        if not previous:
            return None
        fixed = _correct(previous, spelled, store)
        if fixed is None:
            return None
        save(store)
        return "previous", fixed
    inline = _INLINE.search(text)
    if inline:
        spelled = letters(inline.group(1))
        stripped = (text[: inline.start()] + text[inline.end():]).strip(" ,")
        fixed = _correct(stripped, spelled, store)
        if fixed is None:
            return None
        save(store)
        return "inline", fixed
    return None
