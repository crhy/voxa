from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from voxa.agent.deals import parse_request
from voxa.agent.filematch import folder_key
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
_PLAY_LATEST = re.compile(r"play\s+(?:the\s+)?latest\s+(?:video\s+)?from\s+(.+)", re.IGNORECASE)
_SET_CHANNEL = re.compile(r"my (?:you ?tube )?channel(?: name)? is (?:called )?(.+)", re.IGNORECASE)
_PLAY_YT_ON = re.compile(r"(?:play|watch|put on)\s+(.+?)\s+on\s+youtube", re.IGNORECASE)
_PLAY_MUSIC_QUERY = re.compile(r"play\s+music\s+(.+)", re.IGNORECASE)
_PLAY_SOME_MUSIC = re.compile(r"play\s+(?:some\s+)?music", re.IGNORECASE)
_LISTEN_TO = re.compile(r"listen\s+to\s+(.+)", re.IGNORECASE)
_PUT_ON_SOME = re.compile(r"put\s+on\s+some\s+(.+)", re.IGNORECASE)
_PLAY_YT = re.compile(r"(?:play|put on|watch)\s+(.+?)(?:\s+on\s+youtube)?", re.IGNORECASE)
_GENRE = r"(?:jazz|rock|classical|blues|lo-?fi|hip\s?hop|country|metal|ambient|reggae|soul|funk|pop|techno|house|folk|piano|guitar)"
_MOOD = r"(?:relaxing|upbeat|chill|calm)"
_PLAY_MUSIC_BY = re.compile(r"play\s+music\s+by\s+(.+)", re.IGNORECASE)
_PLAY_GENRE_MUSIC = re.compile(r"play\s+(?:(?:some|a little|a bit of)\s+)?(.+?)\s+music", re.IGNORECASE)
_PLAY_SOMETHING_MOOD = re.compile(r"play\s+something\s+(" + _MOOD + r")", re.IGNORECASE)
_PLAY_SOME_GENRE = re.compile(r"play\s+(?:some|a little|a bit of)\s+(" + _GENRE + r")", re.IGNORECASE)
_PLAY_THE_SONG = re.compile(r"play\s+the\s+song\s+(.+)", re.IGNORECASE)
_PLAY_BY_ARTIST = re.compile(r"play\s+(.+?)\s+by\s+(.+)", re.IGNORECASE)
_PLAY_THE_VIDEO = re.compile(r"play\s+the\s+(?:video|movie|trailer|episode)\s+(.+)", re.IGNORECASE)
_PLAY_VIDEO_SUFFIX = re.compile(r"play\s+(.+?)\s+video", re.IGNORECASE)
_SEARCH_YT = re.compile(r"search\s+youtube\s+for\s+(.+)", re.IGNORECASE)
_FIND_YT = re.compile(r"find\s+(.+?)\s+on\s+youtube", re.IGNORECASE)
_SEARCH_WEB = re.compile(
    r"search\s+(?:the\s+)?(?:web|google|online|internet)\s+for\s+(.+)", re.IGNORECASE
)
_GOOGLE = re.compile(r"google\s+(.+)", re.IGNORECASE)
_LOOK_UP = re.compile(r"look\s+up\s+(.+)", re.IGNORECASE)
_IMAGE_SEARCH_FOR = re.compile(
    r"(?:image|picture|photo)\s+search\s+(?:for\s+)?(.+)", re.IGNORECASE
)
_IMAGE_SEARCH_DO = re.compile(
    r"do\s+an?\s+(?:brave\s+)?image\s+search\s+(?:for|of)\s+(.+)", re.IGNORECASE
)
_IMAGE_LOOK_LIKE = re.compile(r"what\s+does\s+(.+?)\s+look\s+like", re.IGNORECASE)
# "Give me directions to X", "how do I get to X", "navigate to X", "take me to X" -> Google Maps directions.
_DIRECTIONS = re.compile(
    r"(?:(?:can you |could you |please )?(?:give me|get me|get|show me|show|find|i need|i want)\s+)?"
    r"(?:driving |walking |the )?directions?\s+(?:to|for)\s+(.+)"
    r"|how do i get to\s+(?:the\s+)?(.+)|navigate (?:me )?to\s+(.+)|take me to\s+(.+)",
    re.IGNORECASE,
)
# "Where is the nearest X", "find the closest X", "nearest X" and "X near me" -> Google Maps search.
_NEAREST = re.compile(
    r"(?:where(?:'s| is| are)\s+|find\s+(?:me\s+)?|show\s+(?:me\s+)?|what(?:'s| is)\s+)?"
    r"(?:the\s+|a\s+)?(?:nearest|closest)\s+(.+)",
    re.IGNORECASE,
)
_NEAR_ME = re.compile(r"(?:find\s+(?:me\s+)?|show\s+(?:me\s+)?|where(?:'s| is| are)\s+)?(?:a\s+|an\s+|the\s+)?(.+?)\s+(?:near me|nearby|around here|close to me)", re.IGNORECASE)
_IMAGE_OF = re.compile(
    r"(?:show\s+me|find|look\s+for|search\s+for|get)\s+(?:\w+\s+)?"
    r"(?:images|pictures|photos|pics)\s+of\s+(.+)",
    re.IGNORECASE,
)
_LOCATION_SET = re.compile(
    r"(?:i[\u2019']?m\s+in|my\s+location\s+is)\s+(.+)", re.IGNORECASE
)
_LOCATION_USE = re.compile(r"use\s+(.+?)\s+as\s+my\s+location", re.IGNORECASE)
_LOCATION_FORGET = re.compile(r"forget\s+my\s+location", re.IGNORECASE)
_URL = re.compile(r"(?:open|go to|visit|navigate to)\s+([^\s]+)", re.IGNORECASE)
_SITE = re.compile(r"(?:open|go to|visit)\s+(.+)", re.IGNORECASE)
_PLAY_PAUSE = re.compile(
    r"(?:pause|resume)(?:\s+(?:the\s+)?(?:music|video|song|playback))?", re.IGNORECASE
)
_NEXT = re.compile(
    r"(?:next|skip)(?:\s+(?:the\s+)?(?:song|track|video))?", re.IGNORECASE
)
_CLEANUP_TEXT = re.compile(
    r"clean\s+up\s+(?:the\s+|this\s+|my\s+)?text", re.IGNORECASE
)
_FIX_TEXT = re.compile(
    r"fix\s+(?:the\s+|this\s+|my\s+)?(?:text|writing|grammar|spelling)", re.IGNORECASE
)
_PROOFREAD = re.compile(r"proofread\s+(?:this|it)", re.IGNORECASE)
_EDIT_CLARITY = re.compile(r"edit\s+(?:this|it)\s+for\s+clarity", re.IGNORECASE)
_CLEANUP_SELECTION = re.compile(
    r"clean\s+up\s+(?:this|the|my)\s+(?:selection|paragraph|sentence)", re.IGNORECASE
)
_FIX_SELECTION = re.compile(
    r"fix\s+(?:this|the|my)\s+(?:selection|paragraph|sentence)", re.IGNORECASE
)
_CLEAN_IT_UP = re.compile(r"clean\s+(?:this|it)\s+up", re.IGNORECASE)
_COPY_FILE = re.compile(r"copy (?:the )?(?:file )?(.+?) from (.+?) to (.+)", re.IGNORECASE)
_MOVE_FILE = re.compile(r"move (?:the )?(?:file )?(.+?) from (.+?) to (.+)", re.IGNORECASE)
_TRASH_FILE = re.compile(
    r"(?:delete|trash|remove) (?:the )?(?:file )?(.+?) (?:from|in) (.+)", re.IGNORECASE
)
_EMPTY_TRASH = re.compile(r"empty (?:the |my )?(?:trash|rubbish|recycle bin)(?: can)?", re.IGNORECASE)
_LARGE_FILES = re.compile(
    r"(?:find|locate|show)(?: me)?(?: all)?(?: the)? files (?:bigger|larger) than (\d+) ?"
    r"(megabytes?|gigabytes?|mb|gb|megs?|gigs?)",
    re.IGNORECASE,
)
_FIND_FILE = re.compile(
    r"(?:find|locate|where is|where's) (?:the |my |a )?file (?:called |named )?(.+)", re.IGNORECASE
)
_OPEN_FOLDER = re.compile(
    r"(?:open|show)(?: me)? (?:my |the )?(.+?) (?:folder|directory)", re.IGNORECASE
)
_LIST_FOLDER = re.compile(
    r"(?:what(?:'s| is)|list(?: the files)?) in (?:my |the )?(.+?)(?: folder| directory)?",
    re.IGNORECASE,
)
_RENAME_FILE = re.compile(
    r"rename (?:the )?(?:file )?(.+?) (?:in|from) (.+?) to (.+)", re.IGNORECASE
)
_MAKE_FOLDER = re.compile(
    r"(?:create|make|add) (?:a )?(?:new )?(?:folder|directory) (?:called |named )?(.+?) in (.+)",
    re.IGNORECASE,
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
_STOP = re.compile(r"stop\s+(?:the\s+)?(?:music|video|playback)", re.IGNORECASE)
_SKIP_AHEAD = re.compile(r"skip\s+ahead|forward\s+(?:30\s+)?seconds?", re.IGNORECASE)
_GO_BACK_30 = re.compile(r"go\s+back\s+(?:30\s+)?seconds", re.IGNORECASE)
_MUTE = re.compile(r"(?:un)?mute(?:\s+(?:the\s+)?(?:music|video|computer|sound|pc))?", re.IGNORECASE)
_SET_VOLUME = re.compile(r"set\s+(?:the\s+)?volume\s+to\s+(\d{1,3})\s*(?:percent|%)?", re.IGNORECASE)
_VOLUME_NUM = re.compile(r"volume\s+(\d{1,3})(?:\s*(?:percent|%))?", re.IGNORECASE)
_MAX_VOLUME = re.compile(r"max(?:imum)?\s+volume", re.IGNORECASE)
_HALF_VOLUME = re.compile(r"half\s+volume", re.IGNORECASE)
_GET_VOLUME = re.compile(r"what(?:'s| is)?\s+the\s+volume|current\s+volume", re.IGNORECASE)
_CUBE_RIGHT = re.compile(r"rotate\s+(?:the\s+)?cube\s+right", re.IGNORECASE)
_CUBE_LEFT = re.compile(r"rotate\s+(?:the\s+)?cube\s+left", re.IGNORECASE)
_ZOOM_IN_MORE = re.compile(r"zoom\s+in\s+more", re.IGNORECASE)
_ZOOM_IN = re.compile(r"zoom\s+in", re.IGNORECASE)
_ZOOM_LEFT = re.compile(r"zoom\s+left", re.IGNORECASE)
_ZOOM_RIGHT = re.compile(r"zoom\s+right", re.IGNORECASE)
_HOME_TURN_ON = re.compile(r"(?:turn|switch)\s+on\s+(?:the\s+)?(.+)", re.IGNORECASE)
_HOME_TURN_OFF = re.compile(r"(?:turn|switch)\s+off\s+(?:the\s+)?(.+)", re.IGNORECASE)
_HOME_TURN_SUFFIX = re.compile(r"turn\s+(?:the\s+)?(.+?)\s+(on|off)", re.IGNORECASE)
_HOME_DIM = re.compile(r"dim\s+(.+?)\s+to\s+(\d+)\s*(?:percent|%)", re.IGNORECASE)
_HOME_THERMOSTAT = re.compile(r"set\s+the\s+thermostat\s+to\s+(\d+)", re.IGNORECASE)
_HOME_SET = re.compile(r"set\s+(.+?)\s+to\s+(\d+)\s*(?:percent|%|degrees?|\u00b0)?", re.IGNORECASE)
_HOME_SCENE = re.compile(r"(?:run|start|activate)\s+(?:the\s+)?(.+?)\s+scene", re.IGNORECASE)
_HOME_STATUS = re.compile(r"is\s+(?:the\s+)?(.+?)\s+(on|off|open|closed|locked)", re.IGNORECASE)
_HOME_APP_NOUNS = frozenset({"computer", "pc", "laptop", "it", "the computer"})
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
_TYPE = re.compile(r"(?:type|write|dictate|enter)\s+(.+)", re.IGNORECASE)
_SAY_TEXT = re.compile(r"say\s+(.+)", re.IGNORECASE)
_REPEAT_AFTER_ME = re.compile(r"repeat\s+after\s+me\s+(.+)", re.IGNORECASE)
_REPEAT_LAST = re.compile(
    r"(?:say|repeat)\s+that(?:\s+again)?|what\s+did\s+you\s+say",
    re.IGNORECASE,
)
_COPY_THAT = re.compile(r"copy (?:that|this)", re.IGNORECASE)
_NEW_TAB = re.compile(r"(?:open|add|create)\s+(?:a\s+)?new tab", re.IGNORECASE)
_SAVE_FILE = re.compile(
    r"(?:save\s+(?:the\s+|this\s+)?(?:file|document)(?:\s+as\s+(\S+))?"
    r"|save\s+as\s+(\S+))",
    re.IGNORECASE,
)
_LOAD_FILE = re.compile(r"(?:load|open)\s+(?:the\s+|a\s+|this\s+)?(?:file|document)", re.IGNORECASE)
_CLOSE_FILE = re.compile(r"(?:close|quit|exit)\s+(?:the\s+|this\s+)?(?:file|document)", re.IGNORECASE)
_NEW_DOC = re.compile(r"(?:create|start|make)?\s*(?:a\s+)?new\s+(?:document|file)", re.IGNORECASE)
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
_LOCK_SCREEN = re.compile(
    r"lock (?:the |my )?(?:screen|computer|pc|desktop)|lock it|lock up", re.IGNORECASE
)
_SWITCH_TO = re.compile(
    r"(?:switch to|go to|focus|bring up)\s+(\w+(?:\s+\w+){0,4})"
    r"|show\s+me\s+(?!images\b|pictures\b|photos\b|pics\b|a\b|an\b|the\b|some\b|how\b|what\b)"
    r"(\w+(?:\s+\w+){0,2})",
    re.IGNORECASE,
)

_MINIMIZE_ALL = re.compile(
    r"(?:minimi[sz]e)\s+(?:everything|all windows)|show the desktop", re.IGNORECASE
)
_MINIMIZE_APP = re.compile(
    r"(?:minimi[sz]e|hide)\s+(\w+(?:\s+\w+){0,4})", re.IGNORECASE
)
_MAXIMIZE_APP = re.compile(
    r"(?:maximi[sz]e)\s+(\w+(?:\s+\w+){0,4})|make\s+(\w+(?:\s+\w+){0,4})\s+(?:bigger|full size)",
    re.IGNORECASE,
)
_RESTORE_APP = re.compile(
    r"(?:restore|bring back|unminimi[sz]e)\s+(\w+(?:\s+\w+){0,4})", re.IGNORECASE
)
_MINIMIZE_ACTIVE = re.compile(
    r"(?:minimi[sz]e|hide)\s+(?:this|it|that|the window|the app|window|app)", re.IGNORECASE
)
_MAXIMIZE_ACTIVE = re.compile(
    r"(?:maximi[sz]e)\s+(?:this|it|that|the window|the app|window|app)"
    r"|make\s+(?:this|it|that|the window|the app|window|app)\s+(?:bigger|full size)",
    re.IGNORECASE,
)
_RESTORE_ACTIVE = re.compile(
    r"(?:restore|bring back|unminimi[sz]e)\s+(?:this|it|that|the window|the app|window|app)", re.IGNORECASE
)
_UNDO_FILE = re.compile(
    r"undo that"
    r"|undo the (?:move|copy|rename|delete|last file action)"
    r"|put (?:it|that|the file) back"
    r"|move it back"
    r"|bring (?:it|that) back"
    r"|restore (?:it|that|the file)",
    re.IGNORECASE,
)

_TIMER_SET = re.compile(
    r"(?:set\s+(?:a\s+|the\s+|an\s+)?timer|start\s+(?:a\s+|the\s+)?timer|timer)\s*(?:for\s+|of\s+)?(.+)",
    re.IGNORECASE,
)
_REMIND_TO = re.compile(r"remind\s+me\s+to\s+(.+?)\s+(?:in|at)\s+(.+)", re.IGNORECASE)
_REMIND_WHEN = re.compile(r"remind\s+me\s+(in|at)\s+(.+?)\s+to\s+(.+)", re.IGNORECASE)
_LIST_REMINDERS = re.compile(
    r"(?:what|any|list|show(?:\s+me)?)\s+(?:the\s+|my\s+)?(?:timers?\s*(?:and|or)\s*)?reminders?\s*(?:do\s+i\s+have)?",
    re.IGNORECASE,
)
_CANCEL_REMINDERS = re.compile(
    r"cancel\s+(?:(?:my|all|the)\s+)?(?:timers?|reminders?)(?:\s+now)?",
    re.IGNORECASE,
)

_BROWSE = re.compile(
    r"(?:browse to|go to the website|open the page)\s+(.+)", re.IGNORECASE
)
_READ_PAGE = re.compile(
    r"read (?:this|the) page|read this to me|what does (?:this|the) page say", re.IGNORECASE
)
_CLICK = re.compile(
    r"click (?:on )?(.+)|press the (.+) button|follow the (.+) link", re.IGNORECASE
)
_SEARCH_SITE = re.compile(
    r"search (?:this site|here|this page) for (.+)", re.IGNORECASE
)
_SCROLL = re.compile(
    r"scroll (down|up)|go to the (top|bottom)(?: of the page)?", re.IGNORECASE
)
_GO_BACK = re.compile(r"go back|previous page", re.IGNORECASE)

ROUTED_TOOLS = frozenset(
    {
        "open_site",
        "open_url",
        "compose_gmail",
        "play_youtube",
        "play_video",
        "play_music",
        "play_latest",
        "set_youtube_channel",
        "media_control",
        "system_volume",
        "compiz_control",
        "search_youtube",
        "web_search",
        "press_key",
        "copy_file",
        "move_file",
        "trash_file",
        "empty_trash",
        "find_file",
        "find_large_files",
        "open_folder",
        "list_folder",
        "rename_file",
        "make_folder",
        "undo_file_action",
        "type_text",
        "file_dialog",
        "send_gmail",
        "open_app",
        "close_app",
        "close_window",
        "lock_screen",
        "switch_to",
        "minimize_app",
        "maximize_app",
        "restore_app",
        "minimize_all",
        "browse",
        "read_page",
        "click_on",
        "search_site",
        "scroll",
        "go_back",
        "set_timer",
        "set_reminder",
        "list_reminders",
        "cancel_reminders",
        "home_turn",
        "home_set",
        "home_scene",
        "home_status",
        "find_deal",
        "set_location",
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
        "dictation",
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
    if not s:
        return None

    def call(tool: str, **args: str) -> ToolCall:
        return ToolCall(tool, args)

    if is_start_dictation(s):
        return None
    if is_compound(s):
        type_match = _TYPE.fullmatch(s)
        if type_match and not _DOCUMENT.match(type_match.group(1)):
            return ToolCall("type_text", {"text": type_match.group(1)})
        return None

    if _REPEAT_LAST.fullmatch(s):
        return call("repeat_last")
    after_me_match = _REPEAT_AFTER_ME.fullmatch(s)
    if after_me_match:
        return call("say_text", text=after_me_match.group(1))
    say_match = _SAY_TEXT.fullmatch(s)
    if say_match and normalize(text).strip().casefold().startswith("say "):
        return call("say_text", text=say_match.group(1))

    timer_match = _TIMER_SET.fullmatch(s)
    if timer_match:
        return call("set_timer", duration=timer_match.group(1))
    remind_to = _REMIND_TO.fullmatch(s)
    if remind_to:
        return call("set_reminder", when=remind_to.group(2), text=remind_to.group(1))
    remind_when = _REMIND_WHEN.fullmatch(s)
    if remind_when:
        return call("set_reminder", when=f"{remind_when.group(1)} {remind_when.group(2)}", text=remind_when.group(3))
    if _LIST_REMINDERS.fullmatch(s):
        return call("list_reminders")
    if _CANCEL_REMINDERS.fullmatch(s):
        return call("cancel_reminders")

    browse_match = _BROWSE.fullmatch(s)
    if browse_match and "." in browse_match.group(1):
        return call("browse", url=browse_match.group(1))
    if _READ_PAGE.fullmatch(s):
        return call("read_page")
    click_match = _CLICK.fullmatch(s)
    if click_match:
        text_value = next(g for g in click_match.groups() if g is not None)
        return call("click_on", text=text_value)
    search_match = _SEARCH_SITE.fullmatch(s)
    if search_match:
        return call("search_site", query=search_match.group(1))
    scroll_match = _SCROLL.fullmatch(s)
    if scroll_match:
        direction = next(g for g in scroll_match.groups() if g is not None)
        return call("scroll", direction=direction)
    if _GO_BACK.fullmatch(s):
        return call("go_back")

    directions_match = _DIRECTIONS.fullmatch(s)
    if directions_match:
        destination = next(g for g in directions_match.groups() if g).strip()
        # "navigate to github.com" is a web address, not a place: leave it to the URL rules below.
        if not re.fullmatch(r"[\w-]+(?:\.[\w-]+)+(?:/\S*)?", destination):
            return call("directions", destination=destination)
    nearby_match = _NEAREST.fullmatch(s) or _NEAR_ME.fullmatch(s)
    if nearby_match:
        return call("find_nearby", what=nearby_match.group(1).strip())

    for pattern in (_IMAGE_SEARCH_FOR, _IMAGE_SEARCH_DO, _IMAGE_LOOK_LIKE, _IMAGE_OF):
        image_match = pattern.fullmatch(s)
        if image_match:
            return call("image_search", query=image_match.group(1))

    location_match = _LOCATION_SET.fullmatch(s)
    if location_match:
        return call("set_location", city=location_match.group(1))
    location_use = _LOCATION_USE.fullmatch(s)
    if location_use:
        return call("set_location", city=location_use.group(1))
    if _LOCATION_FORGET.fullmatch(s):
        return call("set_location", city="")

    if parse_request(s, date.today()):
        return call("find_deal", request=s)

    list_folder_match = _LIST_FOLDER.fullmatch(s)
    if list_folder_match and folder_key(list_folder_match.group(1)) is not None:
        return call("list_folder", folder=list_folder_match.group(1))

    where_file_match = _FIND_FILE.fullmatch(s)
    if where_file_match and re.match(r"where(?:'s| is)\s+(?:the |my |a )file\b", s, re.IGNORECASE):
        return call("find_file", name=where_file_match.group(1))

    if _QUESTION.match(s):
        return None

    if _OPEN_MAIL.fullmatch(s):
        return call("open_site", name="gmail")
    if _COMPOSE.fullmatch(s):
        return call("compose_gmail")

    set_channel_match = _SET_CHANNEL.fullmatch(s)
    if set_channel_match:
        return call("set_youtube_channel", name=set_channel_match.group(1))

    for pattern, tool in (
        (_PLAY_YT_MUSIC, "play_youtube"),
        (_PLAY_LATEST, "play_latest"),
        (_PLAY_YT_ON, "play_youtube"),
        (_PLAY_MUSIC_BY, "play_music"),
        (_PLAY_MUSIC_QUERY, "play_music"),
        (_PLAY_SOME_MUSIC, "play_music"),
        (_PLAY_GENRE_MUSIC, "play_music"),
        (_PLAY_SOMETHING_MOOD, "play_music"),
        (_PLAY_SOME_GENRE, "play_music"),
        (_LISTEN_TO, "play_music"),
        (_PUT_ON_SOME, "play_music"),
        (_PLAY_THE_SONG, "play_music"),
        (_PLAY_BY_ARTIST, "play_music"),
        (_PLAY_THE_VIDEO, "play_video"),
        (_PLAY_VIDEO_SUFFIX, "play_video"),
        (_PLAY_YT, "play_video"),
        (_SEARCH_YT, "search_youtube"),
        (_FIND_YT, "search_youtube"),
        (_SEARCH_WEB, "web_search"),
        (_GOOGLE, "web_search"),
        (_LOOK_UP, "web_search"),
    ):
        match = pattern.fullmatch(s)
        if not match:
            continue
        if pattern is _PLAY_LATEST:
            return call(tool, channel=match.group(1))
        if pattern is _PLAY_SOME_MUSIC:
            query = "music"
        elif pattern in (_PLAY_MUSIC_BY, _PLAY_GENRE_MUSIC, _PLAY_SOMETHING_MOOD, _PLAY_SOME_GENRE, _PUT_ON_SOME):
            query = match.group(1) + " music"
        elif pattern is _PLAY_BY_ARTIST:
            query = match.group(1) + " " + match.group(2)
        else:
            query = match.group(1)
        return call(tool, query=query)

    if s.casefold() == "google":
        return call("web_search", query="google")

    url_match = _URL.fullmatch(s)
    if url_match and "." in url_match.group(1):
        return call("open_url", url=url_match.group(1))
    site_match = _SITE.fullmatch(s)
    if site_match and site_url(site_match.group(1)) is not None:
        return call("open_site", name=site_match.group(1))

    if _PLAY_PAUSE.fullmatch(s) and s.casefold() != "pause":
        # A bare "pause" is for Voxa herself (handled by the window); media needs to be named.
        action = "resume" if "resume" in s.casefold() else "pause"
        return call("media_control", action=action)
    if s.casefold() == "play":
        return call("press_key", key="play pause")
    if _STOP.fullmatch(s):
        return call("media_control", action="stop")
    if _SKIP_AHEAD.fullmatch(s):
        return call("media_control", action="forward")
    if _GO_BACK_30.fullmatch(s):
        return call("media_control", action="back")
    if _NEXT.fullmatch(s):
        return call("media_control", action="next")
    if _PREVIOUS.fullmatch(s):
        return call("media_control", action="previous")
    if _VOLUME_UP.fullmatch(s):
        return call("system_volume", action="louder")
    if _VOLUME_DOWN.fullmatch(s):
        return call("system_volume", action="quieter")
    if _MUTE.fullmatch(s):
        return call("system_volume", action="unmute" if s.casefold().startswith("unmute") else "mute")
    if _SET_VOLUME.fullmatch(s):
        return call("system_volume", action="set", percent=_SET_VOLUME.fullmatch(s).group(1))
    if _MAX_VOLUME.fullmatch(s):
        return call("system_volume", action="set", percent="100")
    if _HALF_VOLUME.fullmatch(s):
        return call("system_volume", action="set", percent="50")
    if _GET_VOLUME.fullmatch(s):
        return call("system_volume", action="get")
    if _VOLUME_NUM.fullmatch(s):
        return call("system_volume", action="set", percent=_VOLUME_NUM.fullmatch(s).group(1))
    if _CUBE_RIGHT.fullmatch(s):
        return call("compiz_control", action="cube_right")
    if _CUBE_LEFT.fullmatch(s):
        return call("compiz_control", action="cube_left")
    if _ZOOM_IN_MORE.fullmatch(s):
        return call("compiz_control", action="zoom_in_more")
    if _ZOOM_IN.fullmatch(s):
        return call("compiz_control", action="zoom_in")
    if re.fullmatch(r"zoom\s+(?:back\s+)?out(?:\s+more)?", s, re.IGNORECASE):
        return call("compiz_control", action="zoom_out")
    if re.fullmatch(r"(?:reset|stop|cancel)\s+(?:the\s+)?zoom|zoom\s+(?:all the way out|reset|off)|normal size", s, re.IGNORECASE):
        return call("compiz_control", action="zoom_reset")
    if re.fullmatch(r"what(?:'s| is)\s+the\s+volume(?:\s+(?:at|level))?|how loud is it|volume level", s, re.IGNORECASE):
        return call("system_volume", action="get")
    if _ZOOM_LEFT.fullmatch(s):
        return call("compiz_control", action="zoom_left")
    if _ZOOM_RIGHT.fullmatch(s):
        return call("compiz_control", action="zoom_right")
    if _COPY_THAT.fullmatch(s):
        return call("press_key", key="copy")
    key_match = _PRESS_KEY.fullmatch(s)
    if key_match:
        return call("press_key", key=key_match.group(1))
    if s.casefold() in _BARE_KEYS:
        return call("press_key", key=s.casefold())

    home_on = _HOME_TURN_ON.fullmatch(s)
    home_off = _HOME_TURN_OFF.fullmatch(s)
    home_suffix = _HOME_TURN_SUFFIX.fullmatch(s)
    if home_on or home_off or home_suffix:
        if home_on:
            name, state = home_on.group(1), "on"
        elif home_off:
            name, state = home_off.group(1), "off"
        else:
            name, state = home_suffix.group(1), home_suffix.group(2)
        if name.strip().casefold() not in _HOME_APP_NOUNS:
            return call("home_turn", name=name, state=state)
    dim_match = _HOME_DIM.fullmatch(s)
    if dim_match:
        return call("home_set", name=dim_match.group(1), value=dim_match.group(2))
    thermo_match = _HOME_THERMOSTAT.fullmatch(s)
    if thermo_match:
        return call("home_set", name="thermostat", value=thermo_match.group(1))
    set_match = _HOME_SET.fullmatch(s)
    if set_match:
        return call("home_set", name=set_match.group(1), value=set_match.group(2))
    scene_match = _HOME_SCENE.fullmatch(s)
    if scene_match:
        return call("home_scene", name=scene_match.group(1))
    status_match = _HOME_STATUS.fullmatch(s)
    if status_match:
        return call("home_status", name=status_match.group(1))

    if _CLOSE_WINDOW.fullmatch(s):
        return call("close_window")
    if _LOCK_SCREEN.fullmatch(s):
        return call("lock_screen")
    close_file_match = _CLOSE_FILE.fullmatch(s)
    if close_file_match:
        return call("file_dialog", action="close")
    close_match = _CLOSE_APP.fullmatch(s)
    if close_match:
        return call("close_app", name=close_match.group(1))
    switch_match = _SWITCH_TO.fullmatch(s)
    if switch_match:
        name = switch_match.group(1) or switch_match.group(2)
        if site_url(name) is not None and "." not in name:
            return call("open_site", name=name.casefold())
        return call("switch_to", name=name)
    if _MINIMIZE_ALL.fullmatch(s):
        return call("minimize_all")
    if _MINIMIZE_ACTIVE.fullmatch(s):
        return call("minimize_app", name="")
    minimize_match = _MINIMIZE_APP.fullmatch(s)
    if minimize_match:
        return call("minimize_app", name=minimize_match.group(1))
    if _MAXIMIZE_ACTIVE.fullmatch(s):
        return call("maximize_app", name="")
    maximize_match = _MAXIMIZE_APP.fullmatch(s)
    if maximize_match:
        return call("maximize_app", name=maximize_match.group(1) or maximize_match.group(2))
    if _UNDO_FILE.fullmatch(s):
        return call("undo_file_action")
    if _RESTORE_ACTIVE.fullmatch(s):
        return call("restore_app", name="")
    restore_match = _RESTORE_APP.fullmatch(s)
    if restore_match:
        return call("restore_app", name=restore_match.group(1))

    copy_match = _COPY_FILE.fullmatch(s)
    if copy_match and folder_key(copy_match.group(2)) is not None and folder_key(copy_match.group(3)) is not None:
        return call(
            "copy_file",
            name=copy_match.group(1),
            source=copy_match.group(2),
            destination=copy_match.group(3),
        )
    move_match = _MOVE_FILE.fullmatch(s)
    if move_match and folder_key(move_match.group(2)) is not None and folder_key(move_match.group(3)) is not None:
        return call(
            "move_file",
            name=move_match.group(1),
            source=move_match.group(2),
            destination=move_match.group(3),
        )
    trash_match = _TRASH_FILE.fullmatch(s)
    if trash_match and folder_key(trash_match.group(2)) is not None:
        return call("trash_file", name=trash_match.group(1), folder=trash_match.group(2))
    if _EMPTY_TRASH.fullmatch(s):
        return call("empty_trash")
    large_match = _LARGE_FILES.fullmatch(s)
    if large_match:
        return call("find_large_files", amount=large_match.group(1), unit=large_match.group(2))
    find_match = _FIND_FILE.fullmatch(s)
    if find_match:
        return call("find_file", name=find_match.group(1))
    open_folder_match = _OPEN_FOLDER.fullmatch(s)
    if open_folder_match and folder_key(open_folder_match.group(1)) is not None:
        return call("open_folder", folder=open_folder_match.group(1))
    rename_match = _RENAME_FILE.fullmatch(s)
    if rename_match and folder_key(rename_match.group(2)) is not None:
        return call(
            "rename_file",
            name=rename_match.group(1),
            folder=rename_match.group(2),
            new_name=rename_match.group(3),
        )
    make_folder_match = _MAKE_FOLDER.fullmatch(s)
    if make_folder_match and folder_key(make_folder_match.group(2)) is not None:
        return call("make_folder", name=make_folder_match.group(1), folder=make_folder_match.group(2))

    if (
        _CLEANUP_TEXT.fullmatch(s)
        or _FIX_TEXT.fullmatch(s)
        or _PROOFREAD.fullmatch(s)
        or _EDIT_CLARITY.fullmatch(s)
        or _CLEANUP_SELECTION.fullmatch(s)
        or _FIX_SELECTION.fullmatch(s)
        or _CLEAN_IT_UP.fullmatch(s)
    ):
        return call("cleanup_text")

    type_match = _TYPE.fullmatch(s)
    if type_match and not _DOCUMENT.match(type_match.group(1)):
        return call("type_text", text=type_match.group(1))
    if _SEND.fullmatch(s):
        return call("send_gmail")

    save_match = _SAVE_FILE.fullmatch(s)
    if save_match:
        name = save_match.group(1) or save_match.group(2) or ""
        if name:
            return call("file_dialog", action="save", name=name)
        return call("file_dialog", action="save")
    load_match = _LOAD_FILE.fullmatch(s)
    if load_match:
        return call("file_dialog", action="load")
    new_match = _NEW_DOC.fullmatch(s)
    if new_match:
        return call("file_dialog", action="new")
    if _NEW_TAB.fullmatch(s):
        return call("press_key", key="new tab")
    app_match = _OPEN_APP.fullmatch(s)
    if app_match:
        name = app_match.group(1)
        return call("open_app", name=_APP_NAME_FIXES.get(name.casefold(), name))
    return None
