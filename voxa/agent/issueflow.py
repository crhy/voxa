"""Posting a GitHub issue by voice: which project, then a title, then a description, then a prefilled page."""

import json
import re
import urllib.parse
import urllib.request

KNOWN_REPOS = {"voxa": "crhy/voxa"}
ASK_TITLE = "What should we title the issue?"
ASK_BODY = "Describe the issue in detail. Say stop dictation when you are done."
CANCELLED = "Okay, no issue posted."

CANCEL_WORDS = {"cancel", "cancel that", "never mind", "forget it"}


def parse_post_issue(text: str) -> str | None:
    t = text.lower().strip()
    t = re.sub(r"^[.!?]+|[.!?]+$", "", t).strip()
    head = re.match(
        r"(post|file|open|create|make|submit|report)\s+"
        r"(?:(?:a|an|new)\s+)?"
        r"(?:github\s+)?"
        r"(issue|bug report|bug)\b(.*)",
        t,
    )
    if not head:
        return None
    tail = head.group(3).strip()
    tail = re.sub(r"(?:^|\s)(?:to|on|in)\s+github\b", "", tail).strip()
    if tail == "":
        return ""
    m = re.fullmatch(r"(for|in|on|about)\s+(.+)", tail)
    if not m:
        return None
    project = m.group(2).strip()
    project = re.sub(r"^the\s+", "", project)
    return project


def find_repo(project: str, known: dict[str, str] | None = None, fetch=None) -> str | None:
    if known is None:
        known = KNOWN_REPOS
    if project == "":
        return known["voxa"]

    def norm(s: str) -> str:
        return re.sub(r"[\s\-]", "", s).lower()

    np = norm(project)
    for key, val in known.items():
        if norm(key) == np:
            return val
    if re.fullmatch(r"[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+", project):
        return project

    if fetch is None:
        def fetch(url):
            req = urllib.request.Request(url, headers={"User-Agent": "Voxa"})
            return urllib.request.urlopen(req, timeout=6).read().decode()

    url = "https://api.github.com/search/repositories?q=" + urllib.parse.quote(project) + "+in:name&per_page=1"
    try:
        data = json.loads(fetch(url))
        return data["items"][0]["full_name"]
    except Exception:
        return None


def issue_url(repo: str, title: str, body: str) -> str:
    body = body[:6000]
    t = urllib.parse.quote(title, safe="")
    b = urllib.parse.quote(body, safe="")
    return f"https://github.com/{repo}/issues/new?title={t}&body={b}"


def tidy(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip())
    if s:
        s = s[0].upper() + s[1:]
    return s


class IssueFlow:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.step = "title"
        self.title = ""
        self.parts = []

    def feed(self, text: str, control: str | None = None) -> tuple[str, str | None]:
        low = re.sub(r"^[.!?]+|[.!?]+$", "", text.lower().strip()).strip()
        if low in CANCEL_WORDS:
            self.step = "done"
            return (CANCELLED, None)

        if self.step == "title":
            if control == "stop":
                self.step = "done"
                return (CANCELLED, None)
            t = re.sub(r"\.+$", "", tidy(text)).strip()
            self.title = t
            self.step = "body"
            return (ASK_BODY, None)

        if self.step == "body":
            if control == "undo":
                if self.parts:
                    self.parts.pop()
                    return ("Removed.", None)
                return ("Nothing to remove.", None)
            if control == "stop" or control == "send":
                self.step = "done"
                body = " ".join(self._sentence(p) for p in self.parts)
                return (
                    f"Opening the issue for {self.repo}. Check it and press Submit.",
                    issue_url(self.repo, self.title, body),
                )
            self.parts.append(text)
            return ("", None)

        return ("", None)

    def _sentence(self, part: str) -> str:
        s = tidy(part)
        if s and s[-1] not in ".!?":
            s += "."
        return s

    @property
    def active(self) -> bool:
        return self.step != "done"

    @property
    def caption(self) -> str:
        if self.step == "title":
            return "Issue title"
        if self.step == "body":
            return "Issue description"
        return ""
