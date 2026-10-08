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


PIECE_CHARS = 500


def split_pieces(text: str, max_chars: int = PIECE_CHARS) -> list[str]:
    """Cut text into pieces a small model can edit reliably; joining the pieces gives the text back exactly.

    Line breaks always end a piece (so they survive untouched); inside a line, whole sentences are
    grouped up to ``max_chars``. A single sentence longer than that stays whole.
    """
    import re

    pieces: list[str] = []
    for line in re.findall(r"[^\n]*\n+|[^\n]+$", text):
        current = ""
        for sentence in re.findall(r"[^.!?]+[.!?]*\s*|[.!?]+\s*", line):
            if current and len(current) + len(sentence) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            pieces.append(current)
    return pieces


def clean_in_pieces(text: str, ask, max_chars: int = PIECE_CHARS) -> tuple[str, int, int]:
    """Clean ``text`` piece by piece with ``ask(piece) -> edited``.

    A whole page at once makes a small model run away (measured: 7,000 characters in, 110,000 out);
    in pieces of 500 characters the same model gets 15 of 16 right in four seconds. A piece whose
    edit fails ``sane_result`` (or raises) is kept as it was. Returns (text, pieces edited, pieces
    with something to edit).
    """
    out: list[str] = []
    edited_count = total = 0
    for piece in split_pieces(text, max_chars):
        core = piece.strip()
        if sum(ch.isalpha() for ch in core) < 3:
            out.append(piece)
            continue
        total += 1
        lead = piece[: len(piece) - len(piece.lstrip())]
        tail = piece[len(piece.rstrip()):]
        try:
            edited = ask(core)
        except Exception:  # noqa: BLE001 - one failed piece must not lose the rest
            edited = ""
        if edited and sane_result(core, edited):
            out.append(lead + edited.strip() + tail)
            edited_count += 1
        else:
            out.append(piece)
    return "".join(out), edited_count, total
