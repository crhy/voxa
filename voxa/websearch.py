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
MAX_QUERY_WORDS = 12

_EXPLICIT = re.compile(
    r"^\s*(?:please\s+)?(?:search|google|look\s+up|look\s+online|find\s+online)"
    r"(?:\s+(?:the\s+web|the\s+internet|online|for|up))*\s*(?P<query>.*)$",
    re.IGNORECASE,
)
_FRESHNESS = re.compile(
    r"\b(latest|news|today|tonight|tomorrow|yesterday|current(?:ly)?|right now|now|"
    r"this (?:week|month|year|weekend)|weather|forecast|score|scores|standings|"
    r"price of|prices?|stock|who won|release date|releases?|version|schedule|"
    r"election|record|records|championship|sports?|202[4-9]|203\d)\b",
    re.IGNORECASE,
)
_ABOUT = re.compile(
    r"\b(tell\s+me\s+about|what\s+do\s+you\s+know\s+about|who\s+(?:is|was|are)|"
    r"what\s+(?:is|was|are)|what's|who's|describe|explain|"
    r"give\s+me\s+(?:some\s+)?(?:info(?:rmation)?|details?|background)\s+(?:on|about)|"
    r"have\s+you\s+heard\s+of|do\s+you\s+know|look\s+up|info\s+on|history\s+of|"
    r"when\s+(?:did|was|is)|where\s+(?:is|was|are)|how\s+(?:many|much|old|tall|far)|which)\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^\s*(?:what|who|how|where|when|which|why|when|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|will|tell|explain|describe|give)\b",
    re.IGNORECASE,
)
_TELL_EXPLAIN = re.compile(r"\b(tell\s+me|explain|describe)\b", re.IGNORECASE)
_SMALL_TALK = re.compile(
    r"^\s*(?:hi|hello|hey|yo|hiya|howdy|good\s+(?:morning|afternoon|evening|day|night)|"
    r"how\s+are\s+you|how\s'?s\s+it\s+going|how\s+do\s+you\s+do|nice|nice\s+one|"
    r"cool|ok(?:ay)?|sure|yes|no|nope|thanks?|thank\s+you|cheers|bye|goodbye|"
    r"see\s+you|later|sorry|pardon|excuse\s+me|who\s+are\s+you|what\s+can\s+you\s+do|"
    r"what'?s\s+your\s+name|who\s'?s\s+voxa|voxa)[\s.!?,]*$",
    re.IGNORECASE,
)
_CREATIVE = re.compile(
    r"\b(write|compose|make|generate|draw|invent)\b.*\b(poem|story|joke|song|haiku|essay|"
    r"limerick|fable|tale)\b|\btell\s+me\s+a\s+(?:joke|story|poem|tale)\b",
    re.IGNORECASE,
)
_VOX_ABOUT = re.compile(
    r"^\s*(?:what\s+can\s+(?:you|i)\s+do|what\s+(?:can|should)\s+i\s+(?:say|ask)|"
    r"who\s+(?:are|is)\s+(?:you|voxa)|what'?s\s+your\s+name|"
    r"what\s+(?:is|are)\s+voxa|what\s+are\s+voxa'?s)\b[\s\S]*$",
    re.IGNORECASE,
)
_PRONOUN = re.compile(
    r"\b(they|them|their|he|she|it|his|her|its|that|those)\b",
    re.IGNORECASE,
)
_LEAD = re.compile(
    r"^\s*(?:please\s+|voxa[,.\s]+|hey\s+|ok(?:ay)?[,.\s]+|can\s+you\s+|could\s+you\s+|"
    r"would\s+you\s+|will\s+you\s+|tell\s+me\s+about\s+|tell\s+me\s+|explain\s+|describe\s+|"
    r"look\s+up\s+|search\s+for\s+|find\s+)+",
    re.IGNORECASE,
)
_MATH_WORDS = {
    "plus", "minus", "times", "multiply", "multiplied", "divide", "divided", "over",
    "by", "squared", "cubed", "percent", "modulo", "mod", "equals", "equal", "to",
    "the", "of", "is", "are", "what", "add", "subtract", "total", "sum", "double",
    "half", "x",
}
COMMON_WORDS = frozenset(
    """
    a about above across after again against all almost alone already also always
    and any anything around as at away back be because been before begin behind being
    below best better big bigger both bottom buy can cannot cant could day days deal
    did do does doing done down each early enough every everyone first five four free
    full get getting got great half hand happen happy has have head hear hearing heard
    hello help here high him his hold home house how however i if in inside into is
    it its itself just keep kept kind know known knows large last late least left less
    let like likely line little long look looking made make making man many may mean
    means might more most move moved much must name near next no not nothing now off
    old on once one only other our out over own part people person place possible put
    quite rather real right run same say saying says second see seen she should side
    simple since small so some something soon still such sure system take taken tell
    tells than that the their them there these they thing things think this those three
    through time to told too top toward two under until up upon us use used very voice
    wait want wanted was watch water way ways well went were what when where whether
    which while who whom whose why will with within without work working world year
    yes you your
    bubble sort
    """.split()
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean_query(text: str) -> str:
    """Prompt with leading request words removed, punctuation trimmed, 12 words max."""
    cleaned = text.strip()
    while True:
        stripped = _LEAD.sub("", cleaned)
        if stripped == cleaned or not stripped:
            break
        cleaned = stripped
    cleaned = re.sub(r"^[\s?!.:,;\"'\u201c\u201d]+|[\s?!.:,;\"'\u201c\u201d]+$", "", cleaned)
    words = cleaned.split()
    return " ".join(words[:MAX_QUERY_WORDS])


def _has_entity(fragment: str) -> bool:
    """A likely named entity: a capitalised word, a number, or two uncommon words."""
    if re.search(r"\b\d[\d,.]*\b", fragment):
        return True
    words = re.findall(r"[A-Za-z][\w'-]*", fragment)
    run = 0
    for word in words:
        if len(word) > 1 and word[0].isupper():
            return True
        if word.lower() in COMMON_WORDS:
            run = 0
        else:
            run += 1
            if run >= 2:
                return True
    return False


def _is_math(query: str) -> bool:
    tokens = re.findall(r"[\w'.%]+", query)
    return bool(tokens) and all(token.isdigit() or token.lower() in _MATH_WORDS for token in tokens)


def search_query_for(prompt: str, mode: str = "auto", previous_query: str | None = None) -> str | None:
    """The query to search for, or None when the question needs no web lookup."""
    text = prompt.strip()
    if not text or mode == "never":
        return None
    if mode == "always":
        if text.endswith("?") or _QUESTION_START.match(text) or _TELL_EXPLAIN.search(text):
            return _clean_query(text)
        return None
    if _SMALL_TALK.match(text) or _CREATIVE.search(text) or _VOX_ABOUT.search(text):
        return None
    explicit = _EXPLICIT.match(text)
    if explicit:
        return explicit.group("query").strip(" ?.") or text
    if _FRESHNESS.search(text):
        return _clean_query(text)
    about = _ABOUT.search(text)
    if about and _has_entity(text[about.end():]) and not _is_math(_clean_query(text)):
        return _clean_query(text)
    if previous_query and len(text.split()) <= 9 and _PRONOUN.search(text):
        return _clean_query(f"{previous_query} {text}")
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


def _site(url: str) -> str:
    """Site name for a result URL: the host without a leading www."""
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def format_for_prompt(query: str, results: list[SearchResult]) -> str:
    """Instruction first, then numbered results, today's date, question last."""
    lines = [
        "Answer ONLY from these search results. Do not add facts from memory."
    ]
    for number, item in enumerate(results[:MAX_RESULTS], 1):
        lines.append(f"{number}. {item.title} — {item.snippet[:300]} ({item.url})")
    lines.append("Today is 6 October 2026.")
    lines.append(f"Question: {query}")
    return "\n".join(lines)


def source_line(results: list[SearchResult]) -> str:
    """Display-only 'Source:' line for the first result, empty when there are none."""
    if not results:
        return ""
    return f"Source: {_site(results[0].url)}"


def search_record_fields(query: str, results: list[SearchResult], failed: bool) -> dict:
    """Action-log fields for a request that attempted a web search."""
    if failed:
        return {"route": "model", "args": None, "detail": "web search failed"}
    if not results:
        return {"route": "model", "args": None, "detail": "web search found nothing"}
    first = results[0].title[:80]
    return {"route": "search", "args": {"query": query}, "detail": f"{len(results)} results: {first}"}
