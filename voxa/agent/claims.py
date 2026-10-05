"""Detect replies that claim Voxa carried out an action it cannot carry out.

The ordinary chat model has no tools: it can only talk. When its answer reads as
if an action had just been done or was being done ("Done. Brutal Chess is closed.",
"Sending it now."), the window replaces it with an honest note. This module is pure
text: it never runs anything, it only decides whether a reply makes such a claim.
"""

from __future__ import annotations

import re

# Only the opening of a reply is judged: a claim of a completed or in-progress
# action, in the first person or the passive voice, belongs at the front.
_CLAIM_PATTERNS = [
    # "Done." / "Done, ..." at the very start of the reply.
    re.compile(r"\Adone\b", re.IGNORECASE),
    # "I opened ...", "I've closed ...", "I have sent ...", "I set ...".
    re.compile(
        r"\bi(?:'ve| have)?\s+"
        r"(opened|closed|sent|played|started|saved|deleted|booked|turned|set|created|launched)\b",
        re.IGNORECASE,
    ),
    # "Opening it ...", "Closing the ...", "Sending your ...".
    re.compile(
        r"\b(opening|closing|sending|playing|launching|starting)\s+(it|that|the|your)\b",
        re.IGNORECASE,
    ),
    # "Brutal Chess is closed.", "It is now sent.", "The song is playing."
    re.compile(r"\b\w+ is (now )?(closed|open|sent|playing)\b", re.IGNORECASE),
    # "sending it now"
    re.compile(r"\bsending it now\b", re.IGNORECASE),
]


def first_sentences(text: str, count: int = 2) -> str:
    """The first ``count`` sentences of ``text``, joined with a space."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:count])


def claims_action(text: str) -> bool:
    """True when the reply's first two sentences claim a completed or in-progress action."""
    if not text or not text.strip():
        return False
    head = first_sentences(text, 2)
    return any(pattern.search(head) for pattern in _CLAIM_PATTERNS)
