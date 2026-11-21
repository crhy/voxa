"""Put the window in front of the user into the model's prompt.

Pure Python: this module touches the host only through the two callables it is
given, and only when the request actually refers to the screen. A request that
says "this", "here" or "on screen" is meaningless without the focused window, so
its title and visible text are attached to the prompt; every other request pays
no cost at all and reads nothing. Screenshots and vision models are a later,
heavier step (crhy/voxa#17).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from voxa.agent import hearing, ui
from voxa.agent.planner import BROWSER_SITE_HINTS

__all__ = [
    "MAX_CHARS",
    "MAX_LINES",
    "gather",
    "refers_to_screen",
    "window_text",
]

log = logging.getLogger("voxa.agent.screencontext")

MAX_LINES = 40
MAX_CHARS = 2000

# Phrasing that points at the screen the user is looking at.
_SCREEN_PHRASE = re.compile(
    r"\bhere\b|\bthere\b"
    r"|\bon (?:my |the |a )?screen\b|\b(?:the|my) screen\b"
    r"|\b(?:this|that|the) window\b"
    r"|\bwhat (?:you|it) see\b"
    r"|\bin front of (?:me|you)\b",
    re.IGNORECASE,
)

# "this" counts only when it points at something: at the end of the request, before
# a noun, or before a linking word. "is this working" and "what is this button" stay
# out of it unless another hint matches.
_THIS_POINTING = re.compile(
    r"\bthis\b"
    r"(?:\s+(?:window|screen|document|doc|file|invoice|text|message|letter|selection|"
    r"paragraph|sentence|line|table|list|thing|one|program|app|application|editor|terminal))??"
    r"(?=\s*$|\s+(?:is|are|says?|means?|want|wants?|needs?|need|open|in|on|to|for)\b)",
    re.IGNORECASE,
)

# Web phrasing keeps its existing route: planner.BROWSER_SITE_HINTS owns "on this
# page" and "on the website", and those requests must still go to the browser.
WEB_HINTS: tuple[str, ...] = BROWSER_SITE_HINTS + (
    "this page",
    "that page",
    "the page",
    "web page",
    "webpage",
    "web site",
    "this site",
    "that site",
    "the site",
    "the browser",
    "in the browser",
    "wikipedia",
    "github",
    "amazon",
)

_PUNCT = re.compile(r"[^\w\s]")


def _fold(text: str) -> str:
    """Lower-case the request with the wake word and punctuation removed."""
    text = hearing.strip_wake(text or "")
    text = _PUNCT.sub(" ", text.replace("\u2019", " "))
    return re.sub(r"\s+", " ", text).strip()


def refers_to_screen(text: str) -> bool:
    """True when the request points at the screen the user is looking at."""
    folded = _fold(text)
    if not folded:
        return False
    if any(hint in folded for hint in WEB_HINTS):
        return False
    return _SCREEN_PHRASE.search(folded) is not None or _THIS_POINTING.search(folded) is not None


def window_text(
    title: str,
    lines: Sequence[str],
    max_lines: int = MAX_LINES,
    max_chars: int = MAX_CHARS,
) -> str:
    """A labelled block with the window's title and visible text; "" when there is nothing to say."""
    title = (title or "").strip()
    cleaned = [line.strip() for line in (str(item) for item in lines)]
    cleaned = [line for line in cleaned if len(line) >= 2]
    if not title and not cleaned:
        return ""
    kept = cleaned[:max_lines] if max_lines > 0 else []
    parts: list[str] = []
    if title:
        parts.append(f'Focused window: "{title}"')
    if kept:
        parts.append("Visible text:")
        parts.extend(kept)
    block = "\n".join(parts)
    if max_chars > 0 and len(block) > max_chars:
        block = block[:max_chars].rstrip()
    return block


def gather(
    request: str,
    *,
    title: Callable[[], str] = ui.active_window_title,
    text: Callable[..., object] = ui.text,
    max_lines: int = MAX_LINES,
    max_chars: int = MAX_CHARS,
) -> str:
    """The screen block for this request, or "" when it does not refer to the screen.

    Reads the window in front, then that window's text. Never raises: a window that
    cannot be named gives "", a window whose text cannot be read gives the title alone.
    """
    if not refers_to_screen(request):
        return ""
    try:
        name = (title() or "").strip()
    except Exception:
        log.debug("screen context: no window title", exc_info=True)
        return ""
    if not name:
        return ""
    try:
        raw = text(name)
    except Exception:
        log.debug("screen context: no text for %s", name, exc_info=True)
        return window_text(name, ())
    if isinstance(raw, dict):
        raw = raw.get("text") or []
    return window_text(name, list(raw or ()), max_lines=max_lines, max_chars=max_chars)
