import re

FILE_NAME = "Voxa Memory.md"

_ECHO_MAX_WORDS = 25

_ECHO_RULES = [
    (r"\bI am\b", "you're", False),
    (r"\bI'm\b", "you're", False),
    (r"\bI\b(?=\s|$)", "you", False),
    (r"\bmine\b", "yours", True),
    (r"\bme\b", "you", True),
    (r"\bmy\b", "your", True),
]


def echo_fact(fact: str) -> str:
    """Turn a remembered fact into the second person for a spoken echo."""
    t = fact.strip()
    for pattern, replacement, preserve_case in _ECHO_RULES:
        flags = re.IGNORECASE if preserve_case else 0

        def _sub(match: re.Match, rep: str = replacement, cap: bool = preserve_case) -> str:
            if cap and match.group(0)[0].isupper():
                return rep[0].upper() + rep[1:]
            return rep

        t = re.sub(pattern, _sub, t, flags=flags)
    words = t.split()
    if len(words) > _ECHO_MAX_WORDS:
        t = " ".join(words[:_ECHO_MAX_WORDS]) + "\u2026"
    return t

STOPWORDS = {
    "the",
    "a",
    "an",
    "my",
    "your",
    "is",
    "are",
    "was",
    "do",
    "you",
    "what",
    "where",
    "when",
    "who",
    "how",
    "about",
    "remember",
    "know",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "i",
    "me",
}


def clean_fact(spoken: str) -> str:
    t = spoken.strip()
    if t[:5].lower() == "that ":
        t = t[5:].strip()
    if t and t[-1] in ".!?":
        t = t[:-1]
    if t:
        t = t[0].upper() + t[1:]
    return t


def facts(content: str) -> list[str]:
    out: list[str] = []
    for line in content.splitlines():
        if line.startswith("- "):
            body = line[2:]
            if " | " in body:
                out.append(body.split(" | ", 1)[1])
    return out


def add(content: str, day, fact: str) -> str:
    text = clean_fact(fact)
    if not text:
        return content
    existing = facts(content)
    for e in existing:
        if e.lower() == text.lower():
            return content
    line = f"- {day.isoformat()} | {text}\n"
    if content and not content.endswith("\n"):
        content += "\n"
    return content + line


def _words(text: str) -> set[str]:
    out: set[str] = set()
    for w in text.lower().split():
        w = w.strip(".,!?;:'\"")
        if w.endswith("s") and len(w) > 1:
            w = w[:-1]
        if w and w not in STOPWORDS:
            out.add(w)
    return out


def search(items: list[str], question: str) -> list[str]:
    q = _words(question)
    if not q:
        return []
    scored: list[tuple[int, int, str]] = []
    for i, item in enumerate(items):
        shared = len(q & _words(item))
        if shared > 0:
            scored.append((shared, i, item))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:3]]


def forget(content: str, spoken: str) -> tuple[str, str | None]:
    found = search(facts(content), spoken)
    if not found:
        return (content, None)
    target = found[0]
    out: list[str] = []
    removed: str | None = None
    for line in content.splitlines():
        if removed is None and line.startswith("- ") and " | " in line[2:]:
            if line[2:].split(" | ", 1)[1] == target:
                removed = target
                continue
        out.append(line)
    new = "".join(f"{line}\n" for line in out)
    return (new, removed)


def spoken_recall(found: list[str], topic: str) -> str:
    if not found:
        return f"I don't have anything about {topic}."
    if len(found) == 1:
        return f"You told me: {found[0]}."
    parts = [f"You told me: {found[0]}."] + [f"Also: {f}." for f in found[1:]]
    return " ".join(parts)


def spoken_all(items: list[str]) -> str:
    if not items:
        return "I don't remember anything yet."
    return f"I remember {len(items)} things. The latest: {items[-1]}."
