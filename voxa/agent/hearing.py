"""Pure post-processing of what Whisper heard: wake-word stripping and ASR fixes."""

import re
from collections.abc import Callable

# First words Whisper produces for "Voxa". Used ONLY by direct_command below to
# recognise a mis-heard wake word in front of a control command; never to wake
# Voxa (that stays the wake-word matcher's job).
WAKE_SOUNDALIKES: frozenset[str] = frozenset(
    {
        "voxa", "vox", "voxo", "vaxa", "vaxac", "boxa", "boxer", "boxes", "box",
        "fox", "foxa", "xa", "za", "vodka", "vocal", "voka", "bossa", "so",
        "set", "step", "sex",
    }
)

CONTROL_COMMANDS: frozenset[str] = frozenset(
    {
        "stop music", "stop the music", "stop playing", "stop", "pause",
        "pause music", "pause the music", "resume", "play", "next", "next song",
        "skip", "louder", "quieter", "volume up", "volume down", "mute",
    }
)

# Mis-hearings that clearly mean "stop music", mapped from normalised form.
MISHEARD_STOP: frozenset[str] = frozenset(
    {"top music", "its top music", "so its top music", "stop usic", "stopped music"}
)

# Single words that are pure acknowledgement/noise, never a request.
NOISE_WORDS: frozenset[str] = frozenset(
    {"music", "peace", "yeah", "okay", "thanks", "bye", "hmm", "uh", "oh"}
)


def _norm(text: str) -> str:
    """Lower-case, drop apostrophes, drop other punctuation, collapse whitespace."""
    text = re.sub(r"['\u2019]", "", text.casefold())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()

WAKE_VARIANTS: tuple[str, ...] = (
    "voxa", "vox a", "vox", "boxa", "box a", "boxer",
    "voxet", "vaxa", "foxa", "voxer", "vauxa",
)

_WAKE_ALT = r"(?:voxa|vox a|vox|boxa|box a|boxer|voxet|vaxa|foxa|voxer|vauxa)"

PHRASE_FIXES: tuple[tuple[str, str | Callable[..., str]], ...] = (
    (r"\blibra office[, ]+(?:right|write|rite|ray|rate|writer)\b", "LibreOffice Writer"),
    (r"\blibre office[, ]+(?:right|write|rite|ray|rate)\b", "LibreOffice Writer"),
    (r"\blibra office\b", "LibreOffice"),
    (r"\bbreathes[., ]+writer\b", "LibreOffice Writer"),
    (
        r"\b(?:space|spaced|spaces|spayed|spaste|spast|based)[ -]*"
        r"(?:linux|lennox|lenox|linix|linus)\b",
        "Spaced Linux",
    ),
    (
        r"\b(?:space|spaced|spaces|spayed|spaste|spast)[ -]*"
        r"(?:bazaar|bizarre|bazar|bizaar|bizzare|bazzar|bizarro)\b",
        "Spaced Bazaar",
    ),
    (
        r"\b(?:space|spaced|spaces|spayed|spaste|spast)[ -]*"
        r"(?:update|updates|updater|updator)\b",
        "Spaced Update",
    ),
    (r"\b(?:space|spaced|spaces|spayed|spaste|spast)[ -]*welcome\b", "Spaced Welcome"),
    (
        r"\b(?:space|spaces|spayed|spaste|spast)[ -]*(hub|store|app store|window manager|installer)\b",
        lambda m: "Spaced " + m.group(1).title(),
    ),
    (
        r"\b((?:update|upgrade|install|open|launch|start|run|close)\b(?:\s+(?:the|my))?\s+)spaces?\s*[,.!?]*$",
        r"\1Spaced",
    ),
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


def direct_command(text: str) -> str | None:
    """Canonical control command for a short, clearly-meant media request.

    Returns the command when the WHOLE utterance is a control command, when it
    is "<wake-soundalike> <control command>", or when it is a known mis-hearing
    of "stop music". A longer sentence is never matched.
    """
    phrase = _norm(text)
    if not phrase:
        return None
    if phrase in MISHEARD_STOP:
        return "stop music"
    if phrase in CONTROL_COMMANDS:
        return phrase
    words = phrase.split()
    if len(words) >= 2 and words[0] in WAKE_SOUNDALIKES:
        rest = " ".join(words[1:])
        if rest in MISHEARD_STOP:
            return "stop music"
        if rest in CONTROL_COMMANDS:
            return rest
    return None


def is_noise(text: str) -> bool:
    """True for one-word filler, short acknowledgements, or a repeated word.

    Covers (a) a single word of four letters or fewer that is not a control
    command, (b) the listed single acknowledgement words, and (c) the same
    word repeated five or more times in a row (a Whisper hallucination).
    """
    phrase = _norm(text)
    if not phrase:
        return False
    words = phrase.split()
    if len(words) == 1:
        word = words[0]
        if word in NOISE_WORDS:
            return True
        return len(word) <= 4 and phrase not in CONTROL_COMMANDS
    if len(words) >= 5 and len(set(words)) == 1:
        return True
    return False


def should_drop_noise(text: str, followup_active: bool) -> bool:
    """True when noise should be dropped silently.

    Noise is dropped only outside a follow-up window: inside a follow-up, short
    affirmations like "Yes" are genuine replies and must survive.
    """
    return is_noise(text) and not followup_active
