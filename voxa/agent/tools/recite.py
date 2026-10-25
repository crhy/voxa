"""Recite a whole poem: "Voxa, recite a poem by Edgar Allan Poe", "recite The Raven by Poe".

A small model cannot recite a poem from memory (it stops early or invents lines), so the text comes
from PoetryDB, a free library of public-domain poems. Voxa reads the entire poem through the normal
reply path; "Voxa, stop" or the Pause button ends it like any other reply.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.parse
import urllib.request

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.readfile import plain_text

API = "https://poetrydb.org"
FIELDS = "title,author,lines"


def fetch(path: str) -> list[dict]:
    """PoetryDB answers a list of poems, or an object with a status when nothing matched."""
    request = urllib.request.Request(
        API + path, headers={"User-Agent": "Mozilla/5.0 (compatible; Voxa/1.0)"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - fixed https host
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def _quote(text: str) -> str:
    return urllib.parse.quote(text.strip(), safe="")


def _same_author(poem: dict, author: str) -> bool:
    # "Edgar Allen Poe" is a common misspelling; the surname decides.
    surname = author.split()[-1].lower()
    return surname in str(poem.get("author", "")).lower().split()


def find_poems(author: str = "", title: str = "") -> list[dict]:
    if title:
        poems = fetch(f"/title/{_quote(title)}/{FIELDS}")
        return [poem for poem in poems if not author or _same_author(poem, author)]
    for name in (author, author.split()[-1]):
        poems = [poem for poem in fetch(f"/author/{_quote(name)}/{FIELDS}") if _same_author(poem, author)]
        if poems:
            return poems
    return []


def poem_speech(poem: dict) -> str:
    body = plain_text("\n".join(poem.get("lines", [])).encode("utf-8")) or ""
    return f"{poem.get('title', 'Untitled')}, by {poem.get('author', 'an unknown poet')}. {body}"


def recite_poem(args: dict[str, str]) -> ToolResult:
    author = (args.get("author") or "").strip()
    title = (args.get("title") or "").strip()
    if not author and not title:
        return ToolResult.failure("Tell me a poet or a poem, for example a poem by Edgar Allan Poe.")
    try:
        poems = find_poems(author, title)
    except (urllib.error.URLError, OSError, ValueError):
        return ToolResult.failure("I could not reach the poetry library. Check the internet connection.")
    if not poems:
        wanted = f"{title} by {author}" if title and author else title or f"a poem by {author}"
        return ToolResult.failure(f"I could not find {wanted} in the poetry library.")
    poem = poems[0] if title else random.choice(poems)
    return ToolResult.success(poem_speech(poem), detail=f"{poem.get('title')} — {poem.get('author')}")


def recite_tools() -> list[Tool]:
    return [
        Tool(
            name="recite_poem",
            description="Recite a whole public-domain poem by a poet, or a named poem, from an online library.",
            parameters={"author": "the poet's name", "title": "the poem's title, if one was named"},
            risk=RiskLevel.READ_ONLY,
            handler=recite_poem,
            required=(),
            timeout_seconds=25.0,
        ),
    ]
