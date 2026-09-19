"""Web search for questions a local model cannot answer from its own memory.

Uses DuckDuckGo's HTML endpoint, which needs no account or API key. Only the
question text leaves the machine, and only when the question looks like it
needs current information (or the user explicitly asks to search).
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Voxa"
TIMEOUT_SECONDS = 10
MAX_RESULTS = 5

_EXPLICIT = re.compile(
    r"^\s*(?:please\s+)?(?:search|google|look\s+up|look\s+online|find\s+online)"
    r"(?:\s+(?:the\s+web|the\s+internet|online|for|up))*\s*(?P<query>.*)$",
    re.IGNORECASE,
)
_FRESHNESS = re.compile(
    r"\b(latest|news|today|tonight|tomorrow|yesterday|current(?:ly)?|right now|"
    r"this (?:week|month|year|weekend)|weather|forecast|score|scores|standings|"
    r"price of|stock|who won|release date|what happened|202[4-9]|203\d)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def search_query_for(prompt: str) -> str | None:
    """The query to search for, or None when the question needs no web lookup."""
    text = prompt.strip()
    if not text:
        return None
    explicit = _EXPLICIT.match(text)
    if explicit:
        return explicit.group("query").strip(" ?.") or text
    if _FRESHNESS.search(text):
        return text.strip(" ?.")
    return None


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _real_url(href: str) -> str:
    href = html.unescape(href)
    parsed = urllib.parse.urlparse(href)
    target = urllib.parse.parse_qs(parsed.query).get("uddg")
    return target[0] if target else href


def parse_results(page: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    links = list(re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S))
    for index, link in enumerate(links):
        end = links[index + 1].start() if index + 1 < len(links) else len(page)
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', page[link.end() : end], re.S)
        results.append(
            SearchResult(_clean(link.group(2)), _real_url(link.group(1)), _clean(snippet.group(1)) if snippet else "")
        )
        if len(results) >= MAX_RESULTS:
            break
    return results


def search(query: str) -> list[SearchResult]:
    data = urllib.parse.urlencode({"q": query}).encode()
    request = urllib.request.Request(SEARCH_URL, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed https URL
        return parse_results(response.read().decode("utf-8", "replace"))


def format_for_prompt(query: str, results: list[SearchResult]) -> str:
    lines = [f"Web search results for “{query}”:"]
    for number, item in enumerate(results, 1):
        lines.append(f"{number}. {item.title} — {item.snippet} ({item.url})")
    lines.append("Answer using these results, and say which source you relied on.")
    return "\n".join(lines)
