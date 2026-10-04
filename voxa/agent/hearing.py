"""Pure post-processing of what Whisper heard: wake-word stripping and ASR fixes."""

import re

WAKE_VARIANTS: tuple[str, ...] = (
    "voxa", "vox a", "vox", "boxa", "box a", "boxer",
    "voxet", "vaxa", "foxa", "voxer", "vauxa",
)

_WAKE_ALT = r"(?:voxa|vox a|vox|boxa|box a|boxer|voxet|vaxa|foxa|voxer|vauxa)"

PHRASE_FIXES: tuple[tuple[str, str], ...] = (
    (r"\blibra office[, ]+(?:right|write|rite|ray|rate|writer)\b", "LibreOffice Writer"),
    (r"\blibre office[, ]+(?:right|write|rite|ray|rate)\b", "LibreOffice Writer"),
    (r"\blibra office\b", "LibreOffice"),
    (r"\bbreathes[., ]+writer\b", "LibreOffice Writer"),
    (r"\bspace(?:d)?\s+(?:bizarre|bazar|bizaar)\b", "Spaced Bazaar"),
    (r"\bspace update\b", "Spaced Update"),
    (r"\bthe gamp\b|\bgamp\b|\bthe gimp\b", "GIMP"),
    (r"\bstopdictation\b|\bstop dictation\b", "stop dictating"),
    (r"\b(open|close|quit|launch|start)\s+cloud\b", r"\1 Claude"),
)

VERB_FIXES: dict[str, str] = {
    "opened": "open",
    "opens": "open",
    "opening": "open",
    "openly": "open",
    "lupin": "open",
    "hoping": "open",
    "closed": "close",
    "closes": "close",
    "clothes": "close",
    "those": "close",
    "cloze": "close",
    "closing": "close",
    "plays": "play",
    "played": "play",
    "please": "play",
    "launched": "launch",
    "started": "start",
}

_FOLLOWING_VERBS = {"were", "are", "was", "is", "would", "could", "should", "seem", "look"}


def strip_wake(text: str) -> str:
    """Remove one leading (or trailing) wake variant, with optional hey/ok/okay
    before it and punctuation after it. Returns "" for a bare wake word."""
    text = text.strip()
    m = re.match(rf"(?i)^(?:hey|ok|okay)[,\s]+{_WAKE_ALT}[,.!?]*\s+", text)
    if m:
        return text[m.end():]
    m = re.match(rf"(?i)^{_WAKE_ALT}[,.!?]*\s+", text)
    if m:
        return text[m.end():]
    if re.fullmatch(rf"(?i){_WAKE_ALT}[,.!?]*", text):
        return ""
    m = re.search(rf"(?i)\s+{_WAKE_ALT}[,.!?]*$", text)
    if m:
        return text[:m.start()].strip()
    return text


def normalize(text: str) -> str:
    """Strip the wake word, apply phrase fixes, then fix the first-word verb."""
    text = strip_wake(text)
    for pattern, repl in PHRASE_FIXES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    words = text.split()
    if words:
        first = words[0].lower()
        if first in VERB_FIXES:
            if first == "please":
                if len(words) > 1 and words[1].lower() == "some":
                    words[0] = "play"
            elif first == "those":
                following = len(words) - 1
                if (
                    2 <= following <= 5
                    and not text.rstrip().endswith("?")
                    and words[1].lower() not in _FOLLOWING_VERBS
                ):
                    words[0] = "close"
            elif first == "started" and len(words) > 1 and words[1].lower() in ("dictating", "dictation"):
                words[0] = "start"
            else:
                words[0] = VERB_FIXES[first]
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    return text.rstrip(" ,.!?")
