from __future__ import annotations

CLEANUP_PROMPT = (
    "You are a careful text editor. Return ONLY the corrected text, with no "
    "preamble, explanation or quotes. Fix spelling, punctuation, grammar and "
    "clarity. Keep the original meaning, the language, the line breaks, any "
    "names, and any formatting marks."
)


def _wrapped(text: str) -> bool:
    body = text.strip()
    if len(body) >= 6 and body.startswith("```") and body.endswith("```"):
        return True
    if len(body) >= 2 and body[0] in "\"'" and body[-1] == body[0]:
        return True
    return False


def sane_result(original: str, edited: str) -> bool:
    if not original or not edited or not edited.strip():
        return False
    if len(edited) < 0.5 * len(original) or len(edited) > 1.6 * len(original):
        return False
    head = edited.lstrip().lower()
    if head.startswith("here is") or head.startswith("sure"):
        return False
    if _wrapped(edited) and not _wrapped(original):
        return False
    return True
