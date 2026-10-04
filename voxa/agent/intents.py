from __future__ import annotations

import re
from dataclasses import dataclass

from voxa.agent.hearing import normalize
from voxa.agent.tools.browser import site_url

__all__ = ["ToolCall", "route", "is_compound", "ROUTED_TOOLS"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    args: dict[str, str]


_LEADING = re.compile(
    r"^(?:please|can you|could you|would you|hey|hi|ok|okay)\s*[,:]?\s*",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"\s+(?:please|for me|thanks|thank you)$", re.IGNORECASE)
_QUESTION = re.compile(r"(?:how|what|why|when|where|who)\b", re.IGNORECASE)

_OPEN_MAIL = re.compile(
    r"(?:open|check|go to|show|look at)\s+(?:my\s+)?(?:gmail|google mail|e-?mail|mail|inbox)",
    re.IGNORECASE,
)
_COMPOSE = re.compile(
    r"(?:write|compose|start|draft|new)\s+(?:a|an|the)?\s*(?:new\s+)?(?:e-?mail|message)"
    r"(?:\s+in\s+gmail)?",
    re.IGNORECASE,
)
_PLAY_YT_MUSIC = re.compile(r"play\s+(.+?)\s+on\s+youtube\s+music", re.IGNORECASE)
_PLAY_MUSIC_QUERY = re.compile(r"play\s+music\s+(.+)", re.IGNORECASE)
_PLAY_SOME_MUSIC = re.compile(r"play\s+(?:some\s+)?music", re.IGNORECASE)
_PLAY_YT = re.compile(r"(?:play|put on|watch)\s+(.+?)(?:\s+on\s+youtube)?", re.IGNORECASE)
_SEARCH_YT = re.compile(r"search\s+youtube\s+for\s+(.+)", re.IGNORECASE)
_FIND_YT = re.compile(r"find\s+(.+?)\s+on\s+youtube", re.IGNORECASE)
_SEARCH_WEB = re.compile(
    r"search\s+(?:the\s+)?(?:web|google|online|internet)\s+for\s+(.+)", re.IGNORECASE
)
_GOOGLE = re.compile(r"google\s+(.+)", re.IGNORECASE)
_LOOK_UP = re.compile(r"look\s+up\s+(.+)", re.IGNORECASE)
_URL = re.compile(r"(?:open|go to|visit|navigate to)\s+([^\s]+)", re.IGNORECASE)
_SITE = re.compile(r"(?:open|go to|visit)\s+(.+)", re.IGNORECASE)
_PLAY_PAUSE = re.compile(
    r"(?:pause|resume)(?:\s+(?:the\s+)?(?:music|video|song|playback))?", re.IGNORECASE
)
_NEXT = re.compile(
    r"(?:next|skip)(?:\s+(?:the\s+)?(?:song|track|video))?", re.IGNORECASE
)
_PREVIOUS = re.compile(
    r"previous\s+(?:song|track|video)|go\s+back\s+a\s+song", re.IGNORECASE
)
_VOLUME_UP = re.compile(
    r"volume\s+up|louder|turn\s+(?:it|the\s+volume)\s+up|turn\s+up\s+(?:the\s+)?volume",
    re.IGNORECASE,
)
_VOLUME_DOWN = re.compile(
    r"volume\s+down|quieter|turn\s+(?:it|the\s+volume)\s+down|turn\s+down\s+(?:the\s+)?volume",
    re.IGNORECASE,
)
_MUTE = re.compile(r"(?:un)?mute(?:\s+(?:the\s+)?(?:music|video))?", re.IGNORECASE)
_PRESS_KEY = re.compile(
    r"press\s+(enter|tab|escape|backspace|space|home|end|delete)", re.IGNORECASE
)
_BARE_KEYS = {
    "select all",
    "copy",
    "paste",
    "undo",
    "redo",
    "save",
    "new tab",
    "close tab",
    "full screen",
    "fullscreen",
}
_TYPE = re.compile(r"(?:type|write|say)\s+(.+)", re.IGNORECASE)
_COPY_THAT = re.compile(r"copy (?:that|this)", re.IGNORECASE)
_NEW_TAB = re.compile(r"(?:open|add|create)\s+(?:a\s+)?new tab", re.IGNORECASE)
_SAVE = re.compile(r"save\s+(?:the|this)\s+(?:file|document)", re.IGNORECASE)
_DOCUMENT = re.compile(
    r"^(?:a|an|the)?\s*(?:new\s+)?(?:e-?mail|message|document|report|letter|essay|"
    r"note|notes|post|article|memo|draft|resume|summary)",
    re.IGNORECASE,
)
_SEND = re.compile(r"send\s+(?:it|the\s+email|this\s+email|the\s+message)", re.IGNORECASE)
_OPEN_APP = re.compile(r"(?:open|launch|start|run)\s+(\w+(?:\s+\w+){0,4})", re.IGNORECASE)
_APP_NAME_FIXES = {"the file manager": "file manager"}
_CLOSE_WINDOW = re.compile(
    r"close (?:this|it|that|the app|(?:this|the|that) window)", re.IGNORECASE
)
_CLOSE_APP = re.compile(
    r"(?:close|quit|exit|kill|shut down)\s+(?:the\s+)?(\w+(?:\s+\w+){0,4})", re.IGNORECASE
)
_SWITCH_TO = re.compile(
    r"(?:switch to|go to|focus|bring up|show me)\s+(\w+(?:\s+\w+){0,4})", re.IGNORECASE
)

ROUTED_TOOLS = frozenset(
    {
        "open_site",
        "open_url",
        "compose_gmail",
        "play_youtube",
        "search_youtube",
        "web_search",
        "press_key",
        "type_text",
        "send_gmail",
        "open_app",
        "close_app",
        "close_window",
        "switch_to",
    }
)


def _prepare(text: str) -> str:
    s = text.strip()
    while True:
        match = _LEADING.match(s)
        if not match:
            break
        s = s[match.end():]
    s = s.strip(".!?,;").strip()
    while True:
        match = _TRAILING.search(s)
        if not match:
            break
        s = s[: match.start()]
    return s.strip(".!?,;").strip()


_DICTATION_STARTS = frozenset(
    {
        "start dictating",
        "start dictation",
        "begin dictation",
        "take dictation",
        "take a dictation",
        "dictate",
        "dictate this",
        "dictation mode",
        "start typing",
        "type what i say",
    }
)


def is_start_dictation(text: str) -> bool:
    """True when the utterance asks the assistant to start dictating into the window."""
    s = normalize(text).strip().casefold()
    if s.startswith("please "):
        s = s[len("please ") :].strip()
    s = s.strip(".,!?;:\"'")
    return s in _DICTATION_STARTS


_COMPOUND_JOIN = re.compile(r"\s*(?:and then|then|after that|and also|and|,)\s+")


def is_compound(text: str) -> bool:
    """True when the utterance chains two commands, e.g. "close X and open Y"."""
    from voxa.agent.planner import COMMAND_VERBS

    s = text.casefold()
    for match in _COMPOUND_JOIN.finditer(s):
        word = re.match(r"[a-z]+", s[match.end():])
        if word is not None and word.group(0) in COMMAND_VERBS:
            return True
    return False


def route(text: str) -> ToolCall | None:
    """Map a spoken command to a tool call, or None when no rule matches."""
    s = _prepare(normalize(text))
    if not s or _QUESTION.match(s):
        return None
    if is_start_dictation(s):
        return None
    if is_compound(s):
        type_match = _TYPE.fullmatch(s)
        if type_match and not _DOCUMENT.match(type_match.group(1)):
            return ToolCall("type_text", {"text": type_match.group(1)})
        return None

    def call(tool: str, **args: str) -> ToolCall:
        return ToolCall(tool, args)

    if _OPEN_MAIL.fullmatch(s):
        return call("open_site", name="gmail")
    if _COMPOSE.fullmatch(s):
        return call("compose_gmail")

    for pattern, tool in (
        (_PLAY_YT_MUSIC, "play_youtube"),
        (_PLAY_MUSIC_QUERY, "play_youtube"),
        (_PLAY_SOME_MUSIC, "play_youtube"),
        (_PLAY_YT, "play_youtube"),
        (_SEARCH_YT, "search_youtube"),
        (_FIND_YT, "search_youtube"),
        (_SEARCH_WEB, "web_search"),
        (_GOOGLE, "web_search"),
        (_LOOK_UP, "web_search"),
    ):
        match = pattern.fullmatch(s)
        if match:
            query = "music" if pattern is _PLAY_SOME_MUSIC else match.group(1)
            return call(tool, query=query)

    if s.casefold() == "google":
        return call("web_search", query="google")

    url_match = _URL.fullmatch(s)
    if url_match and "." in url_match.group(1):
        return call("open_url", url=url_match.group(1))
    site_match = _SITE.fullmatch(s)
    if site_match and site_url(site_match.group(1)) is not None:
        return call("open_site", name=site_match.group(1))

    if _PLAY_PAUSE.fullmatch(s) or s.casefold() == "play":
        return call("press_key", key="play pause")
    if _NEXT.fullmatch(s):
        return call("press_key", key="next track")
    if _PREVIOUS.fullmatch(s):
        return call("press_key", key="previous track")
    if _VOLUME_UP.fullmatch(s):
        return call("press_key", key="volume up")
    if _VOLUME_DOWN.fullmatch(s):
        return call("press_key", key="volume down")
    if _MUTE.fullmatch(s):
        return call("press_key", key="mute")
    if _COPY_THAT.fullmatch(s):
        return call("press_key", key="copy")
    key_match = _PRESS_KEY.fullmatch(s)
    if key_match:
        return call("press_key", key=key_match.group(1))
    if s.casefold() in _BARE_KEYS:
        return call("press_key", key=s.casefold())

    if _CLOSE_WINDOW.fullmatch(s):
        return call("close_window")
    close_match = _CLOSE_APP.fullmatch(s)
    if close_match:
        return call("close_app", name=close_match.group(1))
    switch_match = _SWITCH_TO.fullmatch(s)
    if switch_match:
        name = switch_match.group(1)
        if site_url(name) is not None and "." not in name:
            return call("open_site", name=name.casefold())
        return call("switch_to", name=name)

    type_match = _TYPE.fullmatch(s)
    if type_match and not _DOCUMENT.match(type_match.group(1)):
        return call("type_text", text=type_match.group(1))
    if _SEND.fullmatch(s):
        return call("send_gmail")

    if _SAVE.fullmatch(s):
        return call("press_key", key="save")
    if _NEW_TAB.fullmatch(s):
        return call("press_key", key="new tab")
    app_match = _OPEN_APP.fullmatch(s)
    if app_match:
        name = app_match.group(1)
        return call("open_app", name=_APP_NAME_FIXES.get(name.casefold(), name))
    return None
