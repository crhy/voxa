from __future__ import annotations

import logging
import re

from voxa import apps
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.applications")

# Well-known sites that have no desktop app: when the user says "open CNN" and no app
# matches, we open the site straight away. Kept small and explicit; we never guess an
# address for a name that is not in here.
_KNOWN_SITES: dict[str, str] = {
    "google": "https://www.google.com/",
    "youtube": "https://www.youtube.com/",
    "wikipedia": "https://en.wikipedia.org/",
    "github": "https://github.com/",
    "cnn": "https://www.cnn.com/",
    "bbc": "https://www.bbc.co.uk/",
    "reddit": "https://www.reddit.com/",
    "amazon": "https://www.amazon.com/",
    "gmail": "https://mail.google.com/",
    "maps": "https://www.google.com/maps",
    "netflix": "https://www.netflix.com/",
    "spotify": "https://open.spotify.com/",
    "archive.org": "https://archive.org/",
    "duckduckgo": "https://duckduckgo.com/",
    "twitter": "https://twitter.com/",
    "facebook": "https://www.facebook.com/",
    "instagram": "https://www.instagram.com/",
    "linkedin": "https://www.linkedin.com/",
    "twitch": "https://www.twitch.tv/",
    "discord": "https://discord.com/",
    "stackoverflow": "https://stackoverflow.com/",
    "medium": "https://medium.com/",
    "imdb": "https://www.imdb.com/",
    "ebay": "https://www.ebay.com/",
    "paypal": "https://www.paypal.com/",
}

_EXTENSION_RE = re.compile(r"\.[A-Za-z]{1,8}$")


def _site_for(name: str) -> str | None:
    key = name.casefold().strip()
    if key.startswith("the "):
        key = key[4:].strip()
    return _KNOWN_SITES.get(key)


def _spoken_name(name: str) -> bool:
    words = name.split()
    return 1 <= len(words) <= 3 and not _EXTENSION_RE.search(name)


def open_app(args: dict[str, str]) -> ToolResult:
    # The menu is read on every call (a few milliseconds) rather than cached for the
    # life of the process, so a newly installed app is found straight away.
    name = args["name"]
    app = apps.match_app(name, apps.list_apps())
    if app is not None:
        apps.launch(app)
        return ToolResult.success(f"Opening {app.name}.", detail=app.path)
    site = _site_for(name)
    if site is not None:
        from voxa.agent.tools.browser import open_url

        return open_url({"url": site})
    if _spoken_name(name):
        return ToolResult.failure(
            f'I couldn\'t find an app called {name}. Say "open {name} dot com" for the website.'
        )
    return ToolResult.failure(f"I couldn't find an app called {name}.")


def application_tools() -> list[Tool]:
    return [
        Tool(
            name="open_app",
            description="Launch a desktop application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=open_app,
            required=("name",),
        )
    ]
