from __future__ import annotations

from voxa.agent.hearing import normalize, strip_wake

_PHRASES: tuple[tuple[str, str], ...] = (
    ("new paragraph", "\n\n"),
    ("exclamation mark", "!"),
    ("exclamation point", "!"),
    ("question mark", "?"),
    ("new line", "\n"),
    ("full stop", "."),
    ("open quote", '"'),
    ("close quote", '"'),
    ("semicolon", ";"),
    ("period", "."),
    ("comma", ","),
    ("colon", ":"),
)

_STOP = {"stop dictating", "stop dictation", "end dictation", "that's all", "thats all"}
_UNDO = {"scratch that", "undo that"}
_SEND = {"send it", "send the email"}


def _tokens(text: str) -> list[str]:
    return [token for token in text.split() if token]


def format_dictation(text: str) -> str:
    """Turn one transcribed utterance into the text to type.

    Ordinary words are left exactly as Whisper gave them; spoken punctuation
    and line breaks are expanded. Punctuation attaches to the previous word
    (a quote with nothing before it attaches to the word that follows), and
    the result ends with a single trailing space unless it ends in a newline,
    so a following utterance continues cleanly.
    """
    tokens = _tokens(text)
    out: list[str] = []
    index = 0
    while index < len(tokens):
        matched = False
        for size in (2, 1):
            if index + size <= len(tokens):
                key = " ".join(tokens[index:index + size]).lower().strip(".,!?;:\"'")
                for phrase, replacement in _PHRASES:
                    if key == phrase:
                        out.append(replacement)
                        index += size
                        matched = True
                        break
            if matched:
                break
        if not matched:
            out.append(tokens[index])
            index += 1

    result = ""
    glue_next = False
    for piece in out:
        if piece in ("\n", "\n\n"):
            result += piece
            glue_next = False
        elif piece in (".", ",", "?", "!", ":", ";"):
            if result and not result.endswith((" ", "\n")):
                result += piece
            else:
                result += piece
            glue_next = False
        elif piece == '"':
            if result and not result.endswith((" ", "\n")):
                result += piece
                glue_next = False
            else:
                result += piece
                glue_next = True
        else:
            if result and not result.endswith((" ", "\n")) and not glue_next:
                result += " "
            result += piece
            glue_next = False

    if result and not result.endswith("\n"):
        result += " "
    return result


def _trailing_control(text: str) -> tuple[str | None, list[str]]:
    """Split off trailing control phrases; return (action, remaining_tokens)."""
    tokens = _tokens(normalize(text))
    action: str | None = None
    while True:
        removed = False
        for size in (3, 2, 1):
            if len(tokens) >= size:
                tail = " ".join(tokens[-size:]).lower().strip(".,!?;:\"'")
                if tail in _STOP:
                    action = "stop"
                elif tail in _UNDO:
                    action = "undo"
                elif tail in _SEND:
                    action = "send"
                else:
                    continue
                tokens = tokens[:-size]
                removed = True
                break
        if not removed:
            break
    return action, tokens


def dictation_prefix(text: str) -> str:
    """Text to type when an utterance ends with a control phrase (the part before it)."""
    _action, tokens = _trailing_control(text)
    return format_dictation(" ".join(tokens))


def parse_dictation_control(text: str) -> str | None:
    """Return a dictation control action, or None for ordinary speech."""
    if text.strip() and not strip_wake(text):
        return "stop"
    key = " ".join(_tokens(normalize(text))).lower().strip(".,!?;:\"'")
    if key in _STOP:
        return "stop"
    if key in _UNDO:
        return "undo"
    if key in _SEND:
        return "send"
    action, _remaining = _trailing_control(text)
    return action
