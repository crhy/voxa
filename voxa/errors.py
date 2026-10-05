"""Turn raw AI-server failures into one short, friendly sentence."""

from __future__ import annotations

import re

MAX_CHARS = 160
FALLBACK_CHARS = 120

GPU_OUT_OF_MEMORY = "The graphics card is out of memory. Another model is probably using it."
SERVER_NOT_RUNNING = "The AI server isn't running."
SERVER_TIMED_OUT = "The AI server took too long to answer."
MODEL_NOT_INSTALLED = "That model isn't installed."
SERVER_HTTP_ERROR = "The AI server hit an error."


def _first_sentence(text: str) -> str:
    """The leading sentence of ``text``, cut to FALLBACK_CHARS characters."""
    match = re.search(r"[.!?]", text)
    sentence = text[: match.end()] if match else text
    return sentence.strip()[:FALLBACK_CHARS]


def friendly_error(raw: str) -> str:
    """One short sentence a user can act on, never raw JSON, never over 160 chars."""
    text = raw or ""
    low = text.lower()
    if (
        "failed to allocate" in low
        or "out of memory" in low
        or "cuda" in low
        or "vulkan" in low
    ):
        message = GPU_OUT_OF_MEMORY
    elif "connection refused" in low or "could not connect" in low:
        message = SERVER_NOT_RUNNING
    elif "timed out" in low:
        message = SERVER_TIMED_OUT
    elif "model" in low and "not found" in low:
        message = MODEL_NOT_INSTALLED
    elif re.search(r"http\s*5\d\d", low):
        message = SERVER_HTTP_ERROR
    else:
        message = _first_sentence(text)
    message = message.replace("{", "").replace("}", "")
    return message[:MAX_CHARS]
