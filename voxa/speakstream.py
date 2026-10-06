"""Speak while the model is still writing.

Voxa used to wait for the whole answer before saying a word. :class:`SentenceFeeder` receives the reply as it
streams in and hands back complete sentences as soon as they exist, so the first sentence can be spoken while
the rest is still being written. Pure Python: the window owns the queue and the speech engine.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? … (optionally followed by a closing quote or bracket) and then whitespace.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
# Models that write their working first: their stream must not be read aloud until the answer is known.
_THINKING_MODEL = re.compile(r"qwen3|qwq|deepseek-r1|\br1\b|think|reason|gpt-oss|magistral", re.IGNORECASE)

FIRST_CHUNK_MIN_CHARS = 20   # do not start on a two-word fragment such as "Sure."
LATER_CHUNK_MIN_CHARS = 60   # later chunks are a sentence or two, so playback is not chopped up


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
            minimum = LATER_CHUNK_MIN_CHARS if self.released_any else FIRST_CHUNK_MIN_CHARS
            cut = 0
            for match in _SENTENCE_END.finditer(pending):
                if match.end() >= minimum:
                    cut = match.end()
            # `cut` is the LAST sentence end that satisfies the minimum: release as much as is complete.
        if cut <= 0:
            return []
        piece = pending[:cut].strip()
        self._taken += cut
        if not piece:
            return []
        self.released_any = True
        return [piece]
