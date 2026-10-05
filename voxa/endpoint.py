"""Decide whether a spoken utterance is finished, so Voxa can stop waiting.

Used at the early pause: Voxa takes a quick look at what it has heard so far
with the small always-loaded model and asks :func:`is_complete` whether that
is already a whole command. Questions and ordinary speech are left to wait the
full pause, because people routinely pause mid-sentence.
"""

from __future__ import annotations

import re

from voxa.agent.hearing import normalize
from voxa.agent.intents import is_compound, is_start_dictation, route

__all__ = ["is_complete", "is_followup"]

# A final word from this set promises more is coming ("turn off the", "play
# some"), so the utterance is not finished even if a tool already matched.
_TRAILING_PROMISE = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "for",
        "of",
        "on",
        "in",
        "at",
        "with",
        "and",
        "or",
        "my",
        "some",
        "please",
        "then",
        "um",
        "uh",
    }
)

# Whole utterances that end things on their own.
_STOP_PHRASES = frozenset({"stop", "cancel", "never mind", "go offline"})

# Words and short phrases that clearly continue a previous exchange, so an
# utterance without the wake word is still addressed to Voxa.
_CONTINUATION_WORDS = frozenset(
    {
        "and",
        "also",
        "then",
        "now",
        "next",
        "again",
        "no",
        "not",
        "actually",
        "instead",
        "another",
        "more",
        "less",
        "louder",
        "quieter",
        "yes",
        "yeah",
        "okay",
    }
)
_CONTINUATION_PHRASES = frozenset({"what about", "how about", "make it", "the other"})

# Leading words that open a question.
_QUESTION_STARTS = frozenset(
    {"who", "what", "when", "where", "why", "how", "is", "are", "can", "could", "do", "does", "will"}
)


def _bare(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9'\s]", " ", text.casefold()).split())


def is_complete(text: str, wake_word: str = "voxa") -> bool:
    """True when the utterance is a finished command, stop phrase or dictation start."""
    if not text or not text.strip():
        return False

    wake = _bare(wake_word)
    if wake and _bare(text) == wake:
        return False

    if is_compound(text):
        return False

    normalized = normalize(text)
    if not normalized:
        return False

    if is_start_dictation(text):
        return True

    if _bare(normalized) in _STOP_PHRASES:
        return True

    call = route(text)
    if call is None:
        return False
    if not all(value.strip() for value in call.args.values()):
        return False

    words = re.findall(r"[a-z']+", normalized.casefold())
    if words and words[-1] in _TRAILING_PROMISE:
        return False
    return True


def is_followup(text: str) -> bool:
    """True when an utterance without the wake word still looks addressed to Voxa.

    A routed tool call, a continuation word/phrase, or a question all count;
    anything else is treated as ordinary speech not meant for Voxa.
    """
    if route(text) is not None:
        return True

    bare = _bare(text)
    if not bare:
        return False

    words = bare.split()
    if words[0] in _CONTINUATION_WORDS:
        return True
    if " ".join(words[:2]) in _CONTINUATION_PHRASES:
        return True
    if words[0] in _QUESTION_STARTS:
        return True
    if text.strip().endswith("?"):
        return True
    return False
