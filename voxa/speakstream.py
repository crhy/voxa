"""Speak while the model is still writing.

Voxa used to wait for the whole answer before saying a word. :class:`SentenceFeeder` receives the reply as it
streams in and hands back complete sentences as soon as they exist, so the first sentence can be spoken while
the rest is still being written. Pure Python: the window owns the queue and the speech engine.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? … (optionally followed by a closing quote or bracket) and then whitespace.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
# CJK sentences end at 。！？｡ with no following space.
_CJK_SENTENCE_END = re.compile(r"[。！？｡]+[\"'”’)\]]*")
# Models that write their working first: their stream must not be read aloud until the answer is known.
_THINKING_MODEL = re.compile(r"qwen3|qwq|deepseek-r1|\br1\b|think|reason|gpt-oss|magistral", re.IGNORECASE)

FIRST_CHUNK_MIN_CHARS = 20   # do not start on a two-word fragment such as "Sure."
LATER_CHUNK_MIN_CHARS = 60   # later chunks are a sentence or two, so playback is not chopped up
CJK_CHUNK_MIN_CHARS = 20     # CJK sentences are dense: 20 characters is already a full sentence
FIRST_CLAUSE_MIN_PENDING = 45  # only hunt for a clause break once this much text has arrived
FIRST_CLAUSE_MIN_PIECE = 25    # the early piece must still be long enough to be worth speaking
CJK_CLAUSE_MIN_CHARS = 12      # space-less CJK may start at 、or ，after this many characters

_SPACED_CLAUSE_BREAKS = (", ", "; ", ": ", " — ")
_CJK_CLAUSE_BREAKS = ("、", "，")


def _last_clause_cut(pending: str, breaks: tuple[str, ...], min_piece: int) -> int:
    """Latest position after which the text can be cut at a clause break, or 0.

    The cut sits after the break's punctuation (before any trailing space), so pieces
    joined back with a space (or nothing, for CJK) rebuild the text exactly.
    """
    cut = 0
    for br in breaks:
        start = 0
        while True:
            index = pending.find(br, start)
            if index < 0:
                break
            candidate = index + len(br.rstrip())
            if candidate >= min_piece and candidate > cut:
                cut = candidate
            start = index + 1
    return cut


def is_thinking_model(name: str) -> bool:
    return bool(_THINKING_MODEL.search(name or ""))


def looks_like_reasoning(text: str) -> bool:
    """True when the stream so far contains a scratchpad marker."""
    return "<think" in text or "</think>" in text


class SentenceFeeder:
    """Accumulates streamed text and releases it in speakable pieces."""

    def __init__(self) -> None:
        self._text = ""
        self._taken = 0
        self.released_any = False

    @property
    def spoken_chars(self) -> int:
        return self._taken

    def feed(self, chunk: str) -> list[str]:
        """Add streamed text; return the complete sentences that are now ready (possibly none)."""
        self._text += chunk
        return self._release(final=False)

    def finish(self, full_text: str | None = None) -> list[str]:
        """The answer is complete: return everything not yet released.

        ``full_text`` replaces the accumulated text when given (the cleaned final answer). If what was already
        released is not a prefix of it, only the part after the released length is returned.
        """
        if full_text is not None:
            self._text = full_text
            self._taken = min(self._taken, len(self._text))
        return self._release(final=True)

    def _release(self, final: bool) -> list[str]:
        pending = self._text[self._taken:]
        if final:
            cut = len(pending)
        else:
            cut = 0
            for match in _SENTENCE_END.finditer(pending):
                minimum = LATER_CHUNK_MIN_CHARS if self.released_any else FIRST_CHUNK_MIN_CHARS
                if match.end() >= minimum:
                    cut = max(cut, match.end())
            for match in _CJK_SENTENCE_END.finditer(pending):
                minimum = CJK_CHUNK_MIN_CHARS if self.released_any else 0
                if match.end() >= minimum:
                    cut = max(cut, match.end())
            if not self.released_any and cut == 0:  # nothing long enough ended yet (a short "¡Claro!" does not count)
                # First piece and no sentence end at all: try to start at a clause break.
                if len(pending) >= FIRST_CLAUSE_MIN_PENDING:
                    cut = _last_clause_cut(pending, _SPACED_CLAUSE_BREAKS + _CJK_CLAUSE_BREAKS, FIRST_CLAUSE_MIN_PIECE)
                if cut == 0 and " " not in pending and len(pending) >= CJK_CLAUSE_MIN_CHARS:
                    cut = _last_clause_cut(pending, _CJK_CLAUSE_BREAKS, CJK_CLAUSE_MIN_CHARS)
        if cut <= 0:
            return []
        piece = pending[:cut].strip()
        self._taken += cut
        if not piece:
            return []
        self.released_any = True
        return [piece]
