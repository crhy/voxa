from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from voxa.agent import calc, dates, livetv, lookups, radio, worldtime
from voxa.agent.calendar_local import parse_time, split_when
from voxa.agent.convert import target_extension
from voxa.agent.deals import parse_request
from voxa.agent.filematch import folder_key
from voxa.agent.hearing import normalize
from voxa.agent.helptext import topic_for
from voxa.agent.mailflow import spoken_address
from voxa.agent.tools import translate
from voxa.agent.tools.browser import site_url
from voxa.apps import clean_spoken_app_name
from voxa.ui.avatars import find_character

__all__ = [
    "ToolCall",
    "route",
    "route_spoken",
    "is_compound",
    "is_start_dictation",
    "dictation_target",
    "ROUTED_TOOLS",
]


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    args: dict[str, str]


_LEADING = re.compile(
    r"^(?:please|can you|could you|would you|hey|hi|ok|okay)\b\s*[,:]?\s*",
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
_HELP_ASK = re.compile(
    r"(?:what can you do|what do you do|what are you able to do|what can i (?:say|ask)(?: you)?|help|help me"
    r"|what are your (?:abilities|skills|commands))",
    re.IGNORECASE,
)
_HELP_TOPIC = re.compile(
    r"(?:what can you do|what can i (?:say|do)|help(?: me)?) (?:with|about|for) (.+)"
    r"|(.+) help",
    re.IGNORECASE,
)
_TELL_TIME = re.compile(
    r"(?:what(?:'s| is) the time|what time is it(?: now| right now)?|do you (?:have|know) the time"
    r"|tell me the time|time please)",
    re.IGNORECASE,
)
_WORLD_TIME = re.compile(
    r"(?:what(?:'s| is) the time|what time is it|what time is it right now|tell me the time|current time)"
    r" (?:in|at) (?:the )?(.+)",
    re.IGNORECASE,
)
_TIME_DIFF = re.compile(
    r"(?:what(?:'s| is) the time difference (?:between here and|with|to|from here to) (?:the )?(.+)"
    r"|how many hours (?:ahead|behind) is (?:the )?(.+)"
    r"|how far ahead is (?:the )?(.+))",
    re.IGNORECASE,
)
_TELL_DATE = re.compile(
    r"(?:what(?:'s| is) (?:the|today'?s) date(?: today)?|what day is it(?: today)?"
    r"|what(?:'s| is) today(?:'?s date)?|what day of the week is it)",
    re.IGNORECASE,
)
_DISK_SPACE = re.compile(
    r"how much (?:disk |drive |hard drive |storage )?space (?:do i have|is (?:left|free)|have i got)"
    r"(?: left| free)?(?: on (?:my|the|this) (?:computer|disk|drive))?",
    re.IGNORECASE,
)
_MEMORY_FREE = re.compile(
    r"how much (?:memory|ram) (?:is (?:free|left|available)|do i have(?: free| left)?)",
    re.IGNORECASE,
)
_GPU_STATUS = re.compile(
    r"how hot is (?:my|the) (?:gpu|graphics card|video card)"
    r"|what(?:'s| is) (?:my|the) (?:gpu|graphics card) (?:temperature|temp|usage|doing|status)"
    r"|how is (?:my|the) (?:gpu|graphics card)(?: doing)?"
    r"|gpu (?:temperature|status|usage)",
    re.IGNORECASE,
)
_CPU_STATUS = re.compile(
    r"how hot is (?:my|the) (?:cpu|processor|computer)"
    r"|what(?:'s| is) (?:my|the) (?:cpu|processor) (?:temperature|temp|usage|load)"
    r"|cpu (?:temperature|usage|load)"
    r"|how busy is (?:my|the) (?:cpu|processor|computer)",
    re.IGNORECASE,
)
_SYSTEM_HEALTH = re.compile(
    r"how is (?:my|the|this) (?:computer|pc|system|machine)(?: (?:doing|running))?"
    r"|system (?:health|status)(?: (?:check|report))?"
    r"|is (?:my|the) (?:computer|pc) (?:ok|okay|healthy|overheating)"
    r"|(?:run )?(?:a )?health check",
    re.IGNORECASE,
)
_BATTERY_LEVEL = re.compile(
    r"(?:how much battery (?:is left|do i have(?: left)?)|what(?:'s| is) (?:the|my) battery"
    r"(?: level| percentage)?|battery (?:level|status))",
    re.IGNORECASE,
)
_WHY_SLOW = re.compile(
    r"why (?:is|are) (?:my |the |this )?(?:computer|pc|machine|system|it|everything|things|stuff|performance) (?:so |very |too |really |extremely )?(?:slow|laggy|sluggish|lagging|slowing(?: down)?)"
    r"|why (?:is|are)? ?(?:everything|things|it) (?:so )?slow"
    r"|why so slow"
    r"|what(?:'s| is) (?:slowing|lagging)(?: down)?(?: (?:my|the|this)? ?(?:computer|pc|machine|system|it|everything|things|stuff))?"
    r"|what(?:'s| is) (?:using|eating|hogging|consuming) (?:all |all of )?(?:the|my|this)? ?(?:cpu|processor)(?: (?:usage|power|time|load))?",
    re.IGNORECASE,
)
_WHY_MEMORY = re.compile(
    r"what(?:'s| is) (?:using|eating|hogging|consuming|taking(?: up)?|using up) (?:all |all of )?(?:my|the) (?:memory|ram)",
    re.IGNORECASE,
)
_TROUBLE = re.compile(
    r"(?:is |do you (?:see|notice|think) )?(?:anything|something) (?:wrong|unusual|odd|weird|amiss)(?: going on)?(?: with (?:my|the|this) (?:computer|pc|system|machine))?"
    r"|(?:is |do you (?:see|notice) )?there (?:anything|something) (?:wrong|unusual|odd|weird)(?: with (?:my|the|this) (?:computer|pc|system|machine))?(?: going on)?"
    r"|any (?:problems|issues|trouble|errors|warnings)(?: with (?:my|the|this) (?:computer|pc|system|machine))?"
    r"|(?:diagnose|troubleshoot|check for problems|check for issues)(?: with| on)? (?:the|my|this)? ?(?:pc|computer|system|machine|it)?",
    re.IGNORECASE,
)
_SET_GITHUB_OWNER = re.compile(
    r"my (?:git ?hub|get hub) (?:user ?name|name|account|handle)(?: is)? (?:called )?(.+)",
    re.IGNORECASE,
)
_NOISE_ON = re.compile(
    r"(?:(?:turn|switch|set)\s+(?:the\s+)?(?:adaptive\s+)?noise\s+cancellation\s+on"
    r"|(?:enable)\s+(?:the\s+)?(?:adaptive\s+)?noise\s+cancellation)",
    re.IGNORECASE,
)
_NOISE_OFF = re.compile(
    r"(?:(?:turn|switch|set)\s+(?:the\s+)?(?:adaptive\s+)?noise\s+cancellation\s+off"
    r"|(?:disable)\s+(?:the\s+)?(?:adaptive\s+)?noise\s+cancellation)",
    re.IGNORECASE,
)
_NOISE_GET = re.compile(
    r"is\s+(?:the\s+)?(?:adaptive\s+)?noise\s+cancellation\s+on",
    re.IGNORECASE,
)
_SET_EMAIL = re.compile(r"(?:my )?(.+?)(?:'s|s')? e-?mail(?: address)? is (.+)", re.IGNORECASE)
_GET_EMAIL = re.compile(r"what(?:'s| is) (?:my )?(.+?)(?:'s|s')? e-?mail(?: address)?", re.IGNORECASE)
_FORGET_EMAIL = re.compile(r"forget (?:my )?(.+?)(?:'s|s')? e-?mail(?: address)?", re.IGNORECASE)
_PLAY_YT_ON = re.compile(r"(?:play|watch|put on)\s+(.+?)\s+on\s+youtube", re.IGNORECASE)
# "play boz skaggs on my audacious", "play UFO in my Audacious player" -> the Music folder, in that app (#109).
_PLAY_ON_APP = re.compile(
    r"(?:play|put on|listen to)\s+(.+?)\s+(?:on|in|with|using)\s+(?:my\s+|the\s+)?(.+?)"
    r"(?:\s+(?:music\s+)?(?:player|app|application))?",
    re.IGNORECASE,
)
_NOT_AN_APP = frozenset({"youtube", "you tube", "yt", "youtube music", "spotify", "radio", "tv", "television", "repeat", "loop",
                        "shuffle", "speaker", "speakers", "headphones", "full screen", "fullscreen", "background",
                        "volume", "low", "high"})
_BRIEFING = re.compile(
    r"good\s+(?:morning|afternoon|evening)|brief\s+me|(?:give\s+me\s+)?(?:my\s+|the\s+)?(?:daily\s+|morning\s+)?briefing"
    r"|what\s+does\s+my\s+day\s+look\s+like|what(?:'s|\s+is)\s+on\s+(?:for\s+)?today",
    re.IGNORECASE,
)
_SAY_IN = re.compile(
    r"(?:how\s+(?:do\s+(?:i|you)|would\s+(?:i|you))\s+say|say|translate|what(?:'s|\s+is))\s+(.+?)\s+(?:in|into|to)\s+([a-z]+)",
    re.IGNORECASE,
)
_RADIO_PLAY = re.compile(
    r"(?:play|put\s+on|turn\s+on|listen\s+to|tune\s+(?:in\s+)?to)\s+(?:the\s+|some\s+|me\s+some\s+)?"
    r"(?:(.+?)\s+)?(?:internet\s+)?radio(?:\s+station)?",
    re.IGNORECASE,
)
# "change to Windows 7 dark theme", "switch the desktop theme to mac", "use the ubuntu theme" (#115)
_CHANGE_THEME = re.compile(
    r"(?:change|switch|set)\s+(?:the\s+|my\s+)?(?:desktop\s+)?theme\s+to\s+(?:the\s+)?(.+?)(?:\s+theme)?"
    r"|(?:change|switch)\s+to\s+(?:the\s+)?(.+?)\s+theme|(?:use|apply)\s+(?:the\s+)?(.+?)\s+theme",
    re.IGNORECASE,
)
_PLAY_ANY = re.compile(r"(?:play|put\s+on|listen\s+to)\s+(.+)", re.IGNORECASE)
_RADIO_LIST = re.compile(
    r"(?:what|which)\s+(?:radio\s+)?stations\s+(?:do\s+you\s+have|are\s+there|can\s+you\s+play)"
    r"|list\s+(?:the\s+|your\s+)?(?:radio\s+)?stations",
    re.IGNORECASE,
)
_RADIO_WHAT = re.compile(
    r"(?:what|which)\s+(?:radio\s+)?station\s+is\s+(?:this|playing|that)|what\s+genre\s+is\s+this(?:\s+station)?",
    re.IGNORECASE,
)
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
_LOOK_UP = re.compile(
    r"look\s+up\s+(?!the\s+company\b)(?!.*\bon\s+wikipedia$)(.+)",
    re.IGNORECASE,
)
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
    r"(?:driving |walking |biking |cycling |transit |bus |the )?directions?\s+(?:to|for)\s+(.+)"
    r"|how do i get to\s+(?:the\s+)?(.+)|navigate (?:me )?to\s+(.+)|take me to\s+(.+)",
    re.IGNORECASE,
)
# "Where is the nearest X", "find the closest X", "nearest X" and "X near me" -> Google Maps search.
# Channel names distinctive enough to play without the word "TV" ("watch Al Jazeera").
_TV_NAMES = frozenset(
    livetv.clean_request(name) for channel in livetv.CHANNELS for name in (channel.name, *channel.aliases)
) - {"news", "world news", "space", "dw", "d w"}
_DIRECTIONS_FROM = re.compile(
    r"(?:(?:can you |could you |please )?(?:give me|get me|get|show me|show|find|i need|i want)\s+)?"
    r"(?:(?:driving |walking |biking |cycling |transit |bus |the )?(?:directions?|route|way)|navigate(?: me)?"
    r"|how (?:do|can|would) i get|how to get|take me|drive(?: me)?)"
    r"\s+from\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)
_DIRECTIONS_LOOSE = re.compile(
    r"(?:(?:can you |could you |please )?(?:give me|get me|show me|show|find|i need|i want)\s+)?"
    r"(?:(?:the\s+)?(?:route|way|best way|fastest way|quickest way)\s+to|how (?:can|would) i get to|how to get to"
    r"|drive (?:me )?to|guide me to|map (?:me )?(?:a route )?to)\s+(?:the\s+)?(.+)",
    re.IGNORECASE,
)
_TV_LIST = re.compile(
    r"(?:what|which)\s+(?:live\s+)?(?:tv\s+|television\s+)?channels\s+(?:do\s+you\s+have|are\s+there|can\s+you\s+play)"
    r"|(?:list|show(?:\s+me)?)\s+(?:the\s+|your\s+)?(?:live\s+)?(?:tv|television)\s+(?:channels|stations)",
    re.IGNORECASE,
)
_TV_PLAY = re.compile(
    r"(?:open|play|watch|put\s+on|turn\s+on|show(?:\s+me)?|start)\s+(?:the\s+|some\s+)?"
    r"(?:live\s+|internet\s+)*(?:tv|television)",
    re.IGNORECASE,
)
_TV_CHANNEL = re.compile(r"(?:open|play|watch|put\s+on|turn\s+on|show(?:\s+me)?|start)\s+(?:the\s+)?(.+)", re.IGNORECASE)
_LOCAL_WEATHER = re.compile(
    r"(?:(?:show|tell|give)(?:\s+me)?\s+|what(?:'s| is)\s+|how(?:'s| is)\s+|check\s+|open\s+)?(?:the\s+|my\s+)?"
    r"(?:local\s+)?(?:weather|forecast|weather\s+forecast)(?:\s+(?:like|report))?"
    r"(?:\s+(?:today|tonight|tomorrow|right\s+now|outside|here|near\s+me|in\s+my\s+area|for\s+today))?",
    re.IGNORECASE,
)
_WEATHER_IN = re.compile(
    r"(?:(?:show|tell|give)(?:\s+me)?\s+|what(?:'s| is)\s+|how(?:'s| is)\s+|check\s+)?(?:the\s+)?"
    r"(?:weather|forecast)(?:\s+like)?\s+(?:in|for|at)\s+(.+)",
    re.IGNORECASE,
)
_WHERE_AM_I = re.compile(
    r"where\s+am\s+i(?:\s+(?:right\s+now|now))?|what(?:'s| is)\s+my\s+location|what\s+(?:city|town)\s+am\s+i\s+in"
    r"|where\s+do\s+you\s+think\s+i\s+am",
    re.IGNORECASE,
)
_LOCAL_NEWS = re.compile(
    r"(?:play|show(?:\s+me)?|open|watch|put\s+on|what(?:'s| is))\s+(?:the\s+|my\s+|some\s+|a\s+)?local\s+news"
    r"(?:\s+(?:today|tonight|now|near\s+me|in\s+my\s+area))?",
    re.IGNORECASE,
)
_LOCAL_EVENTS = re.compile(
    r"(?:(?:show(?:\s+me)?|find(?:\s+me)?|what(?:'s| is)|is\s+there(?:\s+any)?|any)\s+)?(?:the\s+|some\s+)?"
    r"(live\s+music|concerts?|events|shows|things\s+to\s+do|comedy(?:\s+shows?)?|farmers\s+markets?)"
    r"(?:\s+(?:on|happening|playing|going\s+on))?"
    r"(?:\s+(tonight|today|tomorrow|this\s+weekend|this\s+week))?"
    r"\s+(?:in\s+my\s+area|near\s+me|around\s+here|nearby|in\s+town|locally)"
    r"(?:\s+(tonight|today|tomorrow|this\s+weekend|this\s+week))?",
    re.IGNORECASE,
)
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
_IMAGE_BARE = re.compile(
    r"(?:images|pictures|photos|pics)\s+of\s+(.+)", re.IGNORECASE
)
_IMAGE_KIND = re.compile(
    r"(?:show\s+me|find(?:\s+me)?|look\s+for|search\s+for|get(?:\s+me)?|pull\s+up|bring\s+up)\s+"
    r"(?:some\s+|a\s+few\s+|more\s+)?(.+?)\s+(?:images|pictures|photos|pics|wallpapers)",
    re.IGNORECASE,
)
_IMAGE_OWN_WORDS = frozenset({"my", "the", "your", "our", "these", "those", "all my"})
_LOCATION_SET = re.compile(
    r"(?:i[\u2019']?m\s+in|my\s+location\s+is)\s+(.+)", re.IGNORECASE
)
_LOCATION_USE = re.compile(r"use\s+(.+?)\s+as\s+my\s+location", re.IGNORECASE)
_LOCATION_FORGET = re.compile(r"forget\s+my\s+location", re.IGNORECASE)
_URL = re.compile(r"(?:open|go to|visit|navigate to)\s+([^\s]+)", re.IGNORECASE)
# "Open open air ships dot com", "go to spaced linux .org": a spoken web address, words run together.
_SPOKEN_URL = re.compile(
    r"(?:open|go to|visit|navigate to|bring up|show me)\s+(?:the\s+website\s+)?(.+?)\s*(?:\s+dot\s+|\.)\s*"
    r"(com|org|net|io|edu|gov|co|us|uk|ca|de|info|tv|app|dev|ai|me|xyz)",
    re.IGNORECASE,
)
_SITE = re.compile(r"(?:open|go to|visit)\s+(.+)", re.IGNORECASE)
_PLAY_PAUSE = re.compile(
    r"(?:pause|resume)(?:\s+(?:the\s+)?(?:music|video|song|playback|radio|tv|television|vlc))?",
    re.IGNORECASE,
)
_NEXT = re.compile(
    r"(?:next|skip)(?:\s+(?:the\s+)?(?:song|track|video))?", re.IGNORECASE
)
_CLEANUP_TEXT = re.compile(
    r"clean\s+up\s+(?:the|this|that|my|it)(?:\s+(?:text|writing|paragraph|sentence|selection|document|email))?",
    re.IGNORECASE,
)
_FIX_TEXT = re.compile(
    r"fix\s+(?:the|this|that|my|it)(?:\s+(?:text|writing|grammar|spelling|paragraph|sentence|selection|document|email))?",
    re.IGNORECASE,
)
_PROOFREAD = re.compile(
    r"proofread\s+(?:this|it|that|(?:the\s+)?(?:text|writing|paragraph|sentence|selection|document|email))",
    re.IGNORECASE,
)
_EDIT_CLARITY = re.compile(r"edit\s+(?:this|it)\s+for\s+clarity", re.IGNORECASE)
_CLEANUP_SELECTION = re.compile(
    r"clean\s+up\s+(?:this|the|my|that)\s+(?:selection|paragraph|sentence)", re.IGNORECASE
)
_FIX_SELECTION = re.compile(
    r"fix\s+(?:this|the|my|that)\s+(?:selection|paragraph|sentence)", re.IGNORECASE
)
_CLEAN_IT_UP = re.compile(r"clean\s+(?:this|it|that)\s+up", re.IGNORECASE)
_READ_SELECTION = re.compile(
    r"read (?:that|this|it|the selection|the selected text|what i selected)(?: to me| out loud| aloud)?",
    re.IGNORECASE,
)
_READ_THIS_FILE = re.compile(
    r"read (?:this|that|the selected|the) (?:text )?file(?: to me| out loud| aloud)?", re.IGNORECASE
)
_READ_FILE = re.compile(
    r"read (?:me )?(?:the |my )?(?:text )?file (?:called |named )?(.+?)(?: to me| out loud| aloud)?",
    re.IGNORECASE,
)
_RECITE_BY = re.compile(
    r"(?:recite|read|say)\s+(?:me\s+)?(?:a|an|some|one|another)\s+poem\s+(?:by|from)\s+(.+)", re.IGNORECASE
)
_RECITE_TITLE = re.compile(r"recite\s+(?:me\s+)?(?!(?:a|an|some|one|another)\s+poem\b)(.+?)(?:\s+by\s+(.+))?", re.IGNORECASE)
_READ_CLIPBOARD = re.compile(
    r"(?:what(?:'s| is) (?:on|in) (?:my |the )?clipboard|read (?:my |the )?clipboard(?: to me)?)",
    re.IGNORECASE,
)
_SUMMARIZE_SELECTION = re.compile(
    r"(?:summari[sz]e|sum up) (?:that|this|it|the selection|the selected text)(?: for me)?"
    r"|give me (?:a|the) summary of (?:that|this|the selection)",
    re.IGNORECASE,
)
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
_HIDDEN_FILES = re.compile(
    r"(show|check|turn on|hide|uncheck|turn off|toggle) hidden files",
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
_STOP = re.compile(r"stop\s+(?:the\s+)?(?:music|video|playback|radio)", re.IGNORECASE)
_SKIP_AHEAD = re.compile(r"skip\s+ahead|forward\s+(?:30\s+)?seconds?", re.IGNORECASE)
_GO_BACK_30 = re.compile(r"go\s+back\s+(?:30\s+)?seconds", re.IGNORECASE)
_MUTE = re.compile(r"(?:un)?mute(?:\s+(?:the\s+)?(?:music|video|computer|sound|pc))?", re.IGNORECASE)
_SET_VOLUME = re.compile(r"set\s+(?:the\s+)?volume\s+(?:to|of|at)\s+(\d{1,3})\s*(?:percent|%)?", re.IGNORECASE)
_VOLUME_NUM = re.compile(r"volume\s+(\d{1,3})(?:\s*(?:percent|%))?", re.IGNORECASE)
_MAX_VOLUME = re.compile(r"max(?:imum)?\s+volume", re.IGNORECASE)
_HALF_VOLUME = re.compile(r"half\s+volume", re.IGNORECASE)
_GET_VOLUME = re.compile(r"what(?:'s| is)?\s+the\s+volume|current\s+volume", re.IGNORECASE)
_CUBE_RIGHT = re.compile(r"(?:rotate|spin|turn)\s+(?:the\s+)?cube\s+(?:to\s+the\s+)?right", re.IGNORECASE)
_CUBE_LEFT = re.compile(r"(?:rotate|spin|turn)\s+(?:the\s+)?cube\s+(?:to\s+the\s+)?left", re.IGNORECASE)
_CUBE_SPIN = re.compile(r"(?:spin|start\s+spinning|keep\s+spinning)\s+(?:the\s+)?cube", re.IGNORECASE)
_CUBE_SPIN_STOP = re.compile(r"stop\s+(?:spinning|the\s+cube|spinning\s+the\s+cube)", re.IGNORECASE)
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
_OPEN_APP = re.compile(
    r"(?:(?:i\s+need\s+to|try\s+to)\s+)?(?:open|launch|start|run)\s+(.+)", re.IGNORECASE
)
_OPEN_FILE_FOLDER = re.compile(
    r"open (?:the |my )?(?:file|document) (?:called |named )?(.+?) (?:in|from|on) (.+)",
    re.IGNORECASE,
)
_OPEN_FILE = re.compile(
    r"open (?:the |my )?(?:file|document) (?:called |named )?(.+)",
    re.IGNORECASE,
)
_INSTALL_APP = re.compile(
    r"open spaced bazaar and install (.+)"
    r"|(?:please )?install (.+?)(?: (?:from|with|using|in|through) (?:spaced )?(?:bazaar|flathub|the app store|the store))?",
    re.IGNORECASE,
)
_UNINSTALL_APP = re.compile(
    r"(?:uninstall|remove the (?:app|program)|delete the (?:app|program)) (.+)",
    re.IGNORECASE,
)
_INSTALL_SKIP = (
    "ollama",
    "updates",
    "the updates",
    "all updates",
    "the latest updates",
    "all the latest updates",
    "the high face",
    "the neural face",
    "it",
    "that",
    "this",
)
_CLOSE_WINDOW = re.compile(
    r"close (?:this|it|that|the app|(?:this|the|that) window)", re.IGNORECASE
)
_CLOSE_APP = re.compile(
    r"(?:close|quit|exit|kill|shut down)\s+(?:the\s+)?(\w+(?:\s+\w+){0,4})", re.IGNORECASE
)
_CLOSE_YOURSELF = re.compile(
    r"(?:close|quit|exit)\s+(?:yourself|voxa)"
    r"|shut\s+yourself\s+down"
    r"|goodbye\s+voxa",
    re.IGNORECASE,
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
_CHARACTER_CHANGE = re.compile(
    r"(?:change\s+(?:the\s+)?characters?\s*(?:to\s+)?|be\s+)([a-z][\w'-]*)",
    re.IGNORECASE,
)
_LIST_CHARACTERS = re.compile(
    r"(?:list|show)\s+(?:the\s+)?(?:available\s+)?characters?"
    r"|who can you be"
    r"|what characters (?:can|do) you (?:be|have)",
    re.IGNORECASE,
)

_MINIMIZE_ALL = re.compile(
    r"(?:minimi[sz]e|hide)\s+(?:everything|all(?:\s+(?:of\s+)?(?:the|my))?(?:\s+open)?"
    r"\s+(?:windows|apps|applications|programs))"
    r"|show\s+(?:me\s+)?(?:the\s+|my\s+)?desktop",
    re.IGNORECASE,
)
_RESTORE_ALL = re.compile(
    r"(?:restore|unminimi[sz]e)\s+(?:everything|all(?:\s+(?:of\s+)?(?:the|my))?(?:\s+open)?"
    r"\s+(?:windows|apps|applications|programs))"
    r"|bring\s+(?:everything|all)\s+back",
    re.IGNORECASE,
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
_MINIMIZE_SELF = re.compile(
    r"(?:minimi[sz]e|hide)\s+(?:yourself|your\s+window|voxa|the\s+voxa\s+window)", re.IGNORECASE
)
_MAXIMIZE_SELF = re.compile(
    r"(?:maximi[sz]es?)\s+(?:yourself|your\s+window|voxa|the\s+voxa\s+window)"
    r"|make\s+(?:yourself|your\s+window)\s+(?:bigger|full\s+(?:size|screen))",
    re.IGNORECASE,
)
_WHATS_OPEN = re.compile(
    r"(?:what(?:'s| is) open"
    r"|what(?: programs| apps| windows)? (?:do i have|are) open"
    r"|list (?:the |my )?(?:open )?windows"
    r"|which (?:programs|apps|windows) are (?:open|running))",
    re.IGNORECASE,
)
_WHICH_WINDOW = re.compile(
    r"(?:what window is this"
    r"|which window is (?:this|in front|active)"
    r"|what(?:'s| is) (?:this|the active|the current) window"
    r"|what am i looking at)",
    re.IGNORECASE,
)
_MAXIMIZE_ACTIVE = re.compile(
    r"(?:maximi[sz]e)\s+(?:this|it|that|the window|the app|window|app)"
    r"|make\s+(?:this|it|that|the window|the app|window|app)\s+(?:bigger|full\s+(?:size|screen))"
    r"|full\s*screen",
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
    r"(?:(?:set|start)\s+(?:a\s+|the\s+|an\s+)?)?"
    r"(?:(?P<dur>(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)\s+(?:minutes?|hours?|seconds?))\s+timer"
    r"|timer\s*(?:for\s+|of\s+)?(?P<dur2>.+))",
    re.IGNORECASE,
)
_REMIND_TO = re.compile(r"remind\s+me\s+to\s+(.+?)\s+(?:in|at)\s+(.+)", re.IGNORECASE)
_REMIND_WHEN = re.compile(r"remind\s+me\s+(in|at)\s+(.+?)\s+to\s+(.+)", re.IGNORECASE)
_LIST_REMINDERS = re.compile(
    r"(?:what(?:\s+are)?|any|list|show(?:\s+me)?|read(?:\s+me)?)\s+(?:the\s+|my\s+)?(?:timers?\s*(?:and|or)\s*)?"
    r"(?:reminders?|timers?)\s*(?:do\s+i\s+have|are\s+(?:set|running))?",
    re.IGNORECASE,
)
_CANCEL_REMINDERS = re.compile(
    r"cancel\s+(?:(?:my|all|the)\s+)?(?:timers?|reminders?)(?:\s+now)?",
    re.IGNORECASE,
)
_TAKE_NOTE = re.compile(
    r"(?:take|make|write|add|save) (?:a |this |down a )?note(?:,|:)?(?: that| saying| to| of)?(?:,|:)? (.+)",
    re.IGNORECASE,
)
_NOTE_SAYS = re.compile(r"note (?:that |down )?(.+)", re.IGNORECASE)
_WRITE_DOWN = re.compile(r"(?:write|jot) (?:this|that) down:? (.+)", re.IGNORECASE)
_READ_NOTES = re.compile(
    r"(?:read|what are|show me|tell me) (?:me )?my notes|what(?:'s| is) in my notes|do i have any notes",
    re.IGNORECASE,
)
_DELETE_LAST_NOTE = re.compile(r"(?:delete|remove|erase|scratch) (?:my |the )?last note", re.IGNORECASE)
_OPEN_NOTES = re.compile(r"open (?:my )?notes", re.IGNORECASE)
# A list is named either by a word that can only mean one ("shopping", "to-do") or by any word followed
# by "list" ("packing list"). A bare unknown word is not a list: "what's on TV", "what's on my calendar".
_LIST_KNOWN = r"to[- ]?do|todo|task|tasks|shopping|grocery|groceries"
_LIST = rf"(?:my |the )?(?:(?:{_LIST_KNOWN})(?: list)?|(?!clipboard\b)[a-z]+ list)"
_LIST_NAMED = rf"(?:my |the )?(?:{_LIST_KNOWN}|(?!clipboard\b)[a-z]+(?: plan)?) list"
_LIST_ADD = re.compile(rf"(?:add|put) (?P<item>.+?) (?:to|on) (?P<list>{_LIST})", re.IGNORECASE)
_PRESENTATION = re.compile(
    r"(?:make|create|build|give|do)\s+(?:me\s+)?(?:a\s+)?"
    r"(?:(?P<count>\d+|one|two|three|four|five|six|seven|eight)\s+slides?"
    r"|presentation|slide\s+deck|deck|slides?)\s+"
    r"(?:about|on|regarding)\s+(?P<topic>.+)",
    re.IGNORECASE,
)
_FINANCE_SORT = re.compile(
    r"(?:organise|organize|sort|file) (?:my |the )?(?:financial|finance) (?:documents?|docs?|papers?)",
    re.IGNORECASE,
)
_FINANCE_UNDO = re.compile(r"undo (?:the |my )?(?:filing|sorting|sort)", re.IGNORECASE)
_FINANCE_KIND = r"(?:invoices?|receipts?|warrant(?:y|ies)|tax|(?:(?:bank )?statements?))"
_FINANCE_COUNT = re.compile(
    rf"(?:"
    rf"how many (?P<kind1>{_FINANCE_KIND})(?: (?:do i have|are there))?"
    rf"|where (?:are|is) (?:my |the )?(?P<kind2>{_FINANCE_KIND})"
    rf")",
    re.IGNORECASE,
)
_FAMILY_ADD = re.compile(
    r"(?:add|put) (?P<event>.+?) for (?P<person>[a-z]+) (?:on |for )?(?P<when>.+)",
    re.IGNORECASE,
)
_FAMILY_WEEK = re.compile(r"what(?:'s| is) on for (?:the |our )?family(?: this week)?", re.IGNORECASE)
_FAMILY_REMOVE = re.compile(r"(?:remove|delete) (?P<person>[a-z]+)'s (?P<event>.+)", re.IGNORECASE)
_TRAVEL_OPTIONS = re.compile(
    r"(?:"
    r"(?:my |the )?(?P<what>flight|train|bus|connection) (?:to|for) (?P<place>.+?) "
    r"(?:was |got |is )?(?P<problem>cancelled|canceled|delayed)"
    r"|(?:i )?missed (?:my |the )?(?P<what2>flight|train|bus|connection) (?:to|for) (?P<place2>.+)"
    r")"
    r"(?:\s+(?:what are my options|what can i do|options|what now).*)?",
    re.IGNORECASE,
)
_DOC_EXTRACT = re.compile(
    r"(?:"
    r"what (?:are|is) the (?:dates?|amounts?|details?|important details?)"
    r"(?: and (?:amounts?|dates?|details?))? in (?P<n1>.+)"
    r"|read the (?:important )?details? (?:from|in|of) (?P<n2>.+)"
    r"|what does (?P<n3>.+?) say about .+"
    r")",
    re.IGNORECASE,
)
_TRAVEL_DOC_NOUN = (
    r"passport|visa|licen[cs]e|id|identification|insurance|registration|"
    r"green card|credit card|sim card|certificate|health certificate|vaccination"
)
_TRAVEL_DOC_ADD = re.compile(
    rf"(?:remember|note|save|store)\s+(?:my\s+|the\s+)?(?P<doc>{_TRAVEL_DOC_NOUN})\s+"
    rf"(?:that\s+)?(?:it\s+)?expires?\s+(?:on\s+|in\s+|by\s+)?(?P<expires>.+)",
    re.IGNORECASE,
)
_TRAVEL_DOC_WHEN = re.compile(
    rf"when does (?:my\s+|the\s+)?(?P<doc>{_TRAVEL_DOC_NOUN})\s+expire[s]?",
    re.IGNORECASE,
)
_TRAVEL_DOC_LIST = re.compile(
    r"(?:what|which) travel documents (?:do i have|i have|are there|do you have)"
    r"|(?:list|show)(?: me)? (?:my )?travel documents",
    re.IGNORECASE,
)
_TRAVEL_DOC_FORGET = re.compile(
    rf"(?:forget|delete|remove|drop) (?:my\s+|the\s+)?(?P<doc>{_TRAVEL_DOC_NOUN})\b",
    re.IGNORECASE,
)
_EVENT_NOUN = (
    r"birthday party|birthday|dinner party|dinner|barbecue|barbeque|bbq|cookout|"
    r"picnic|wedding|gathering|get[- ]?together|reunion|fundraiser|celebration|"
    r"feast|banquet|party"
)
_EVENT_PLAN = re.compile(
    rf"(?:help me )?(?:plan|organise|organize|arrange)\s+(?:an?\s+|the\s+|my\s+)?"
    rf"(?P<event>{_EVENT_NOUN})\s*(?P<tail>.*)",
    re.IGNORECASE,
)
_CONVO_SUMMARY = re.compile(
    r"(?:summarize|summarise|sum up|recap)(?: (?:our|the|this|my))? conversation"
    r"|what did we (?:decide|agree)(?: on)?"
    r"|what are my action items(?: from (?:this|our) conversation)?",
    re.IGNORECASE,
)
_TRIP_DAYS = r"one|two|three|four|five|six|seven|weekend"
_TRIP_NOUN = r"trip|trips|itinerary|itineraries|getaway|getaways|vacation|vacations|holiday|holidays|weekend"
_TRIP_PLAN = re.compile(
    rf"(?:plan|make) (?:an? |my )?(?:(?P<days>{_TRIP_DAYS})|(?P<noun>{_TRIP_NOUN}))(?P<tail>.+)",
    re.IGNORECASE,
)
_TRIP_DAY_WORD = re.compile(r"\b(one|two|three|four|five|six|seven|weekend)\b", re.IGNORECASE)
_TRIP_CONNECTOR = re.compile(r"\b(?:to|in|for)\b", re.IGNORECASE)
_TRIP_READ = re.compile(
    r"(?:read|show) (?:me )?(?:my |the )?(?:trip|itinerary|getaway|vacation|holiday)(?: (?:to|for))? (?P<place>.+)",
    re.IGNORECASE,
)
_TRIP_FORGET = re.compile(
    r"(?:forget|delete|remove) (?:my |the )?(?:trip|itinerary|getaway|vacation|holiday)(?: (?:to|for))? (?P<place>.+)",
    re.IGNORECASE,
)
_PROJECT_STEPS = re.compile(
    r"break down (?P<g1>.+?)(?: into steps)?"
    r"|help me plan (?P<g2>.+)"
    r"|what are the steps to (?P<g3>.+)"
    r"|make a plan for (?P<g4>.+)",
    re.IGNORECASE,
)
_BACKUP_NOW = re.compile(
    r"back up (?:my |voxa(?:'s)? )?(?:data|notes|lists|files|everything)"
    r"|(?:make|create) a backup",
    re.IGNORECASE,
)
_BACKUP_LAST = re.compile(
    r"when (?:was|did) (?:my |the )?last backup(?: run| happen)?"
    r"|do i have a backup",
    re.IGNORECASE,
)
_SORT_PREVIEW = re.compile(
    r"(?:how|what) (?:do|does|would|will|should) (?:my |the )?(.+?) (?:folder|directory)? (?:sort|sorts|sorted)"
    r"|(?:preview|sort) (?:my |the )?(.+?) (?:folder|directory)?",
    re.IGNORECASE,
)
_MEAL_IDEA = re.compile(
    r"what(?:'s| is| should we have| should i make| can i cook) for dinner(?: tonight)?"
    r"|(?:give me|suggest) (?:a |an )?(?P<tag>quick|vegetarian|chicken|beef|fish|pasta|soup|salad)? ?(?:dinner|meal|recipe)(?: idea)?"
    r"|what should i cook(?: tonight)?",
    re.IGNORECASE,
)
_MEAL_PLAN = re.compile(
    r"plan (?:my |our |the )?(?:meals|dinners)(?: for (?:the|this) week)?"
    r"|(?:make|give me) a meal plan",
    re.IGNORECASE,
)
_MEAL_SHOP = re.compile(
    r"add (?:the |its |those )?ingredients(?: for (?P<name>.+?))? to (?:my |the )?shopping list"
    r"|i(?:'ll| will) make (?P<name2>that|.+)",
    re.IGNORECASE,
)
_PROJECT_STATUS = re.compile(
    r"what(?:'s| is) the git status of (?:the |my )?(?P<n1>.+?)"
    r"|what(?:'s| is) the status of (?:the |my )?(?P<n2>.+?)(?: project| repo(?:sitory)?)"
    r"|how is (?:the |my )?(?P<n3>.+?) (?:project| repo(?:sitory)?)(?: doing)?",
    re.IGNORECASE,
)
_MODEL_STATUS = re.compile(
    r"is (?:the local model|strata|the coding model) (?:running|up|working|busy)"
    r"|what is (?:the local model|strata) (?:doing|working on)"
    r"|(?:local model|strata) status",
    re.IGNORECASE,
)
_HEADS_UP = re.compile(
    r"what should i know(?: today| about today)?|what(?:'s| is) (?:important|going on) today"
    r"|anything i should know|(?:give me )?a heads up",
    re.IGNORECASE,
)
_FORGOTTEN = re.compile(
    r"what am i forgetting|(?:did|have) i forg(?:et|otten)? anything|what have i missed",
    re.IGNORECASE,
)
_SUGGEST_NEXT = re.compile(
    r"what do you (?:recommend|suggest)(?: i do)?(?: next| now)?|what would you do(?: next| now)?",
    re.IGNORECASE,
)
_MEETING_BRIEF = re.compile(
    r"(?:brief me on|prepare me for|what do i need for|tell me about) (?:my |the )?(?P<title>.+?)"
    r"(?: appointment| meeting)",
    re.IGNORECASE,
)
_CAL_WEEK_Q = re.compile(
    r"what(?:'s| is) (?:on |happening )?this week"
    r"|what does (?:my|the|our) week look like"
    r"|what(?:'s| is) (?:the|our) family schedule(?: this week)?",
    re.IGNORECASE,
)
_CAL_WEEK_PERSON = re.compile(
    r"what does (?P<name>[a-z]+) have(?: on)? this week"
    r"|what(?:'s| is) (?P<name2>[a-z]+) doing this week"
    r"|what(?:'s| is) on (?P<name3>[a-z]+)'s (?:calendar|schedule)(?: this week)?",
    re.IGNORECASE,
)
_CAL_BUSY = re.compile(
    r"how busy (?:am i|is my week|is this week)|how many appointments do i have this week",
    re.IGNORECASE,
)
_CAL_CONFLICTS = re.compile(
    r"(?:do i have|are there) any (?:conflicts|clashes)(?: on my calendar| this week)?"
    r"|check my calendar for (?:conflicts|clashes)",
    re.IGNORECASE,
)
_WEEKDAY_WORD = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_CAL_WEEKLY1 = re.compile(rf"(?:add|put|schedule) (?P<title>.+?) every (?P<day>{_WEEKDAY_WORD}) at (?P<time>.+?)(?: (?:on|to|in) my calendar)?", re.IGNORECASE)
_CAL_WEEKLY2 = re.compile(rf"i have (?P<title>.+?) every (?P<day>{_WEEKDAY_WORD}) at (?P<time>.+)", re.IGNORECASE)
_CAL_WEEKLY3 = re.compile(rf"(?P<title>.+?) is every (?P<day>{_WEEKDAY_WORD}) at (?P<time>.+)", re.IGNORECASE)
_CONVERT_TO = re.compile(
    r"convert (?:the |my )?(?:file |document )?(?P<name>.+?) (?:into|in|to) (?:a |an )?(?P<fmt>.+?)"
    r"(?: file| document)?(?: in (?:the |my )?(?P<folder>.+?)(?: folder)?)?",
    re.IGNORECASE,
)
_CONVERT_AS = re.compile(r"(?:save|export) (?:the |my )?(?P<name>.+?) as (?:a |an )?(?P<fmt>.+?)", re.IGNORECASE)
_CONVERT_MAKE = re.compile(r"make (?:a |an )?(?P<fmt>.+?) (?:of|from) (?:the |my )?(?P<name>.+?)", re.IGNORECASE)
_TRANSCRIBE = re.compile(
    r"transcrib(?:e|ption) (?:the |my |this )?(?:file |recording |audio |clip |voice ?memo )?(?P<name>.+?)"
    r"(?: in (?:the |my )?(?P<folder>.+?)(?: folder)?)?",
    re.IGNORECASE,
)
_FOCUS_DURATION = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)[\s-]+(?:minutes?|hours?)"
_FOCUS_START = re.compile(
    rf"(?:start|begin)\s+(?:a |an |at )?(?:(?P<dur1>{_FOCUS_DURATION})\s+)?focus\s+(?:session|time|timer|block)"
    rf"(?:\s+for\s+(?P<dur2>{_FOCUS_DURATION}))?(?:\s+(?:on|for)\s+(?P<what>.+))?",
    re.IGNORECASE,
)
_FOCUS_NEED = re.compile(rf"i\s+need\s+to\s+focus(?:\s+for\s+(?P<dur>{_FOCUS_DURATION}))?", re.IGNORECASE)
_FOCUS_POMODORO = re.compile(r"(?:start|begin)\s+(?:a\s+)?(?:pomm?[ao]doro|pal[ai]doro|pamodoro)(?:\s+(?:timer|session))?", re.IGNORECASE)
_FOCUS_PROTECT = re.compile(rf"protect\s+(?:the\s+next\s+)?(?P<dur>{_FOCUS_DURATION})(?:\s+for\s+(?P<what>.+))?", re.IGNORECASE)
_FOCUS_STATUS = re.compile(
    r"how\s+much\s+focus\s+time\s+(?:is|do\s+i\s+have)\s+left"
    r"|how\s+long\s+(?:is\s+left|until\s+my\s+break)"
    r"|am\s+i\s+(?:still\s+)?in\s+(?:a\s+)?focus(?:\s+(?:session|time))?",
    re.IGNORECASE,
)
_FOCUS_STOP = re.compile(
    r"(?:stop|end|cancel)\s+(?:the\s+|my\s+)?focus(?:\s+(?:session|time|timer))?"
    r"|i'?m\s+done\s+focusing",
    re.IGNORECASE,
)
_LIST_READ = re.compile(
    rf"(?:what's|what is) on (?P<a>{_LIST})|read(?: me)? (?P<b>{_LIST_NAMED})|show (?:me )?(?P<c>{_LIST_NAMED})",
    re.IGNORECASE,
)
_LIST_DONE = re.compile(
    rf"check(?:ed)? off (?P<i1>.+?) (?:on|from) (?P<l1>{_LIST})|mark(?:ed)? (?P<i2>.+?) (?:as |is )?done(?: (?:on|from) (?P<l2>{_LIST}))?"
    rf"|i finished (?P<i3>.+?)(?: (?:on|from) (?P<l3>{_LIST}))?|cross(?:ed)? off (?P<i4>.+?)(?: (?:on|from) (?P<l4>{_LIST}))?",
    re.IGNORECASE,
)
_LIST_REMOVE = re.compile(rf"(?:remove|take) (?P<item>.+?) (?:from|off) (?P<list>{_LIST})", re.IGNORECASE)
_LIST_CLEAR = re.compile(rf"(?:clear|empty) (?P<list>{_LIST_NAMED})", re.IGNORECASE)
_LIST_PROGRESS = re.compile(
    r"how am i doing(?: on my (?P<list>to[- ]?do|todo|task|tasks)? ?list)?"
    r"|(?:give me )?(?:a |my )?progress report"
    r"|what have i (?:finished|done|completed)(?: today)?",
    re.IGNORECASE,
)
_LIST_PENDING = re.compile(
    r"what(?:'s| is) (?:still )?(?:pending|left|outstanding)(?: on my lists?)?"
    r"|what(?:'s| is) still open(?: on my lists)?"
    r"|what do i still have to do"
    r"|what(?:'s| is) left to do",
    re.IGNORECASE,
)
_LIST_NEXT = re.compile(
    r"what should i (?:do|work on)(?: next| first| now)?"
    r"|what(?:'s| is) (?:my )?(?:next task|top priority|most important)(?: today)?"
    r"|what do i do next",
    re.IGNORECASE,
)
_MEMORY_REMEMBER = re.compile(
    r"(?:remember|don't forget|keep in mind) (?:that )?(?!to\b)(?P<fact>.+)",
    re.IGNORECASE,
)
_MEMORY_RECALL = re.compile(
    r"what do you (?:remember|know) about (?P<t1>.+)"
    r"|do you (?:remember|know) (?:anything about )?(?P<t2>.+)"
    r"|what did i (?:tell you|say) about (?P<t3>.+)"
    r"|tell me what you (?:remember|know) about (?P<t4>.+)",
    re.IGNORECASE,
)
_MEMORY_LIST = re.compile(
    r"what do you remember|what have i told you|show (?:me )?(?:your|my) memory",
    re.IGNORECASE,
)
_MEMORY_FORGET = re.compile(
    r"forget (?:what i said |what i told you )?about (?P<t1>.+)|forget that (?P<t2>.+)",
    re.IGNORECASE,
)
_CAL = r"calendar|schedule|agenda"
_CAL_DAY = (
    r"today|tomorrow|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
)
_CAL_ADD1 = re.compile(r"(?:add|put|book|plan) (?P<title>.+?) (?:on|to|in) my calendar(?: (?:for|on )?(?P<when>.+))?", re.IGNORECASE)
_CAL_ADD2 = re.compile(r"(?:add|put|book|plan) (?P<title>.+?) (?P<when>.+?) (?:on|to|in) my calendar", re.IGNORECASE)
_CAL_ADD3 = re.compile(r"(?:schedule|book|plan) (?P<title>.+?) (?:for|on) (?P<when>.+)", re.IGNORECASE)
_CAL_ADD4 = re.compile(r"i have (?P<title>.+?) (?P<when>.+)", re.IGNORECASE)
_CAL_DAY1 = re.compile(rf"what(?:'s| is) on my (?:{_CAL})(?: (?P<day>{_CAL_DAY}))?", re.IGNORECASE)
_CAL_DAY2 = re.compile(rf"what do i have (?:on )?(?P<day>{_CAL_DAY})", re.IGNORECASE)
_CAL_DAY3 = re.compile(rf"what(?:'s| is) my (?:schedule|agenda|day)(?: look like)?(?: (?P<day>{_CAL_DAY}))?", re.IGNORECASE)
_CAL_DAY4 = re.compile(rf"do i have anything(?: (?P<day>{_CAL_DAY}))?", re.IGNORECASE)
_CAL_NEXT = re.compile(
    r"what(?:'s| is) (?:my )?next(?: appointment| meeting| event)?|what(?:'s| is) next on my calendar",
    re.IGNORECASE,
)
_CAL_CANCEL = re.compile(
    r"(?:cancel|remove|delete) (?:my |the )?(?P<title>.+?)(?: appointment| meeting)?(?: from my calendar)?",
    re.IGNORECASE,
)
_CAL_NAMED = re.compile(r"calendar|appointment|meeting", re.IGNORECASE)
_EXPENSE_CATEGORY = r"food|transport|home|fun|health|other"
_EXPENSE_PERIOD = r"today|this week|this month|this year"
_EXPENSE_ADD = re.compile(
    r"i spent (?P<a1>.+?) on (?P<w1>.+)|i paid (?P<a2>.+?) for (?P<w2>.+)"
    r"|add an expense of (?P<a3>.+?) for (?P<w3>.+)|log (?P<a4>.+?) for (?P<w4>.+)",
    re.IGNORECASE,
)
_EXPENSE_TOTAL = re.compile(
    rf"how much (have i|did i) spen[dt](?: on (?P<c1>{_EXPENSE_CATEGORY}))?(?: (?P<p1>{_EXPENSE_PERIOD}))?"
    rf"|what did i spend(?: on (?P<c2>{_EXPENSE_CATEGORY}))?(?: (?P<p2>{_EXPENSE_PERIOD}))?",
    re.IGNORECASE,
)
_EXPENSE_BREAKDOWN = re.compile(
    rf"(?:show|give me|what is|what's) my (?:spending|expenses|budget)(?: breakdown)?(?: (?P<p1>{_EXPENSE_PERIOD}))?"
    rf"|where (?:did|does) my money go(?: (?P<p2>{_EXPENSE_PERIOD}))?",
    re.IGNORECASE,
)
_EXPENSE_UNDO = re.compile(r"(?:remove|delete|undo) (?:the |my )?last expense", re.IGNORECASE)
_EXPENSE_SHEET = re.compile(
    r"(?:make|create|open) (?:a |my )?(?:spreadsheet|sheet) (?:of|with|from) my (?:expenses|spending)"
    r"|export my (?:expenses|spending)(?: to a spreadsheet)?",
    re.IGNORECASE,
)
_BUDGET_SET = re.compile(
    rf"set (?:my |the )?(?P<c1>{_EXPENSE_CATEGORY}) budget to (?P<a1>.+?)(?: a month)?"
    rf"|my (?P<c2>{_EXPENSE_CATEGORY}) budget is (?P<a2>.+)",
    re.IGNORECASE,
)
_BUDGET_STATUS = re.compile(
    rf"how am i doing on my(?: (?P<c1>{_EXPENSE_CATEGORY}))? budget"
    rf"|how much is left in my (?P<c2>{_EXPENSE_CATEGORY}) budget"
    rf"|(?:show|what is|what's) my budgets?",
    re.IGNORECASE,
)
_TRIP_ADD = re.compile(r"i spent (?P<a1>.+?) on (?P<w1>.+?) (?:for|on) the (?P<t1>.+) trip", re.IGNORECASE)
_TRIP_COST = re.compile(
    r"how much (?:did|has) the (?P<n1>.+?) trip cost(?: so far)?"
    r"|what did i spend on the (?P<n2>.+?) trip",
    re.IGNORECASE,
)
_DATE_KIND = r"birthday|anniversary"
_DATE_REMEMBER = re.compile(
    rf"(?:remember (?:that )?)?(?P<name>.+?)'s (?P<kind>{_DATE_KIND}) is (?:on )?(?P<when>.+)",
    re.IGNORECASE,
)
_DATE_DUE = re.compile(r"(?:the )?(?P<name>.+?) is due (?:on |by )?(?P<when>.+)", re.IGNORECASE)
_DATE_DEADLINE = re.compile(r"(?P<name>.+?) deadline is (?:on )?(?P<when>.+)", re.IGNORECASE)
_DATE_WHEN = re.compile(
    rf"when is (?P<name>.+?)'s (?P<kind>{_DATE_KIND})|when is (?:the )?(?P<due>.+?) due",
    re.IGNORECASE,
)
_DATE_UPCOMING = re.compile(
    r"(?:what|any|which) (?:birthdays|anniversaries|deadlines|dates)(?: (?:are|do i have))?(?: (?:coming up|this week|this month|soon))?"
    r"|what(?:'s| is) (?:coming up|due)(?: (?:this week|this month|soon))?",
    re.IGNORECASE,
)
_DATE_FORGET = re.compile(
    r"forget (?P<name>.+?)'s (?:birthday|anniversary)|forget the (?P<deadline>.+?) deadline"
    r"|forget the (?P<bill>.+?) bill"
    r"|forget the (?P<subscription>.+?) subscription"
    r"|forget the (?P<service>.+?\b(?:service|inspection|registration|maintenance|oil change|check-?up))",
    re.IGNORECASE,
)
_DATE_BILL = re.compile(
    r"(?:the |my )?(?P<name>.+?) bill is due (?:on |by )?(?P<when>.+)"
    r"|i pay (?:the |my )?(?P<pay>.+?)(?: bill)? (?:on |by )(?P<paywhen>.+)",
    re.IGNORECASE,
)
_DATE_SUBSCRIPTION = re.compile(
    r"(?:the |my )?(?P<name>.+?) subscription (?:renews|is due|bills|charges) (?:on |by )?(?P<when>.+)"
    r"|(?P<renew>.+?) renews (?:on |by )?(?P<renewwhen>.+)",
    re.IGNORECASE,
)
_DATE_SERVICE = re.compile(
    r"(?:the |my )?(?P<name>.+?\b(?:service|inspection|registration|oil change|check-?up|maintenance)) is due (?:on |in )?(?P<when>.+)",
    re.IGNORECASE,
)
_DATE_BILLS_QUESTION = re.compile(
    r"what bills (?:are|do i have)(?: coming up| due)?(?: this week| this month)?",
    re.IGNORECASE,
)
_DATE_SUBSCRIPTIONS_QUESTION = re.compile(
    r"what subscriptions do i have|(?:list|show) my subscriptions",
    re.IGNORECASE,
)
_DATE_WHEN_BILL = re.compile(r"when is (?:the |my )?(?P<name>.+?) bill due", re.IGNORECASE)
_DATE_WHEN_RENEW = re.compile(r"when does (?:the |my )?(?P<name>.+?)(?: subscription)? renew", re.IGNORECASE)


def _date_name(phrase: str) -> str:
    """"my mom" -> "mom": the bare name the date tools expect."""
    return re.sub(r"^(?:my|the)\s+", "", (phrase or "").strip(), flags=re.IGNORECASE)


def _list_word(phrase: str | None) -> str:
    """"my shopping list" -> "shopping": the bare name the list tools expect."""
    return re.sub(r"\s+list$", "", re.sub(r"^(?:my|the)\s+", "", (phrase or "").strip(), flags=re.IGNORECASE), flags=re.IGNORECASE)


_TAKE_PICTURE = re.compile(
    r"(?:take|snap|shoot|grab)\s+(?:a\s+|my\s+|me\s+a\s+)?(?:picture|photo|photograph|selfie|pic)"
    r"(?:\s+of\s+me|\s+with\s+the\s+(?:web\s?cam|camera))?(?:\s+please)?",
    re.IGNORECASE,
)
_SHOW_PICTURE = re.compile(
    r"(?:show|open)(?:\s+me)?\s+(?:the|that|my)\s+(?:picture|photo|selfie)|let\s+me\s+see\s+(?:it|the\s+(?:picture|photo))",
    re.IGNORECASE,
)
_TAKE_SCREENSHOT = re.compile(
    r"(?:take|grab|capture|make) (?:a |me a )?(?:screen ?shot|picture of (?:the|my) screen)"
    r"(?: of (?:the|my) screen)?(?: please)?|screen ?shot(?: this| that| the screen)?",
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
_UI_PRESS_IN = re.compile(
    r"(?:press|click|push|hit|tap)(?: on)?(?: the)? (.+?) in (?:the )?(.+?)(?: window| app| program)?",
    re.IGNORECASE,
)
_UI_PRESS_BUTTON = re.compile(
    r"(?:press|click|push|hit|tap)(?: on)?(?: the)? (.+?) button", re.IGNORECASE
)
_UI_LIST_BUTTONS = re.compile(
    r"(?:what|which) buttons (?:are there|can i press|do you see|are on (?:the |this )?screen)(?: in (.+))?"
    r"|list (?:the )?buttons(?: in (.+))?",
    re.IGNORECASE,
)
_UI_READ_WINDOW = re.compile(
    r"read (?:this|the) window(?: to me)?|what does (?:this|the) window say|what(?:'s| is) on (?:the|my) screen",
    re.IGNORECASE,
)
_SEARCH_SITE = re.compile(
    r"search (?:this site|here|this page) for (.+)", re.IGNORECASE
)
_SCROLL = re.compile(
    r"(?:scroll|page)\s+(down|up)(?:\s+a\s+little)?"
    r"|scroll\s+to\s+the\s+(top|bottom)"
    r"|go\s+to\s+the\s+(top|bottom)(?:\s+of\s+the\s+page)?",
    re.IGNORECASE,
)
_KEEP_SCROLLING = re.compile(r"keep scrolling", re.IGNORECASE)
_LAST_SCROLL_DIRECTION = "down"
_GO_BACK = re.compile(r"go back|previous page", re.IGNORECASE)
_UPDATE_SYSTEM = re.compile(
    r"(?:please )?(?:update|upgrade) (?:spaced(?: linux)?|the (?:system|computer|os|operating system)"
    r"|my (?:system|computer)|everything|this computer|linux)(?: now| please)?"
    r"|(?:install|get|run|do) (?:all )?(?:the )?(?:latest |system |available )?updates(?: now| please)?"
    r"|open spaced update and (?:install|run|do|apply) (?:all )?(?:the )?(?:latest )?updates?"
    r"|run (?:a |the )?(?:system|os|full) update",
    re.IGNORECASE,
)
_CHECK_VERSION = re.compile(
    r"(?:what|which) version of spaced(?: linux)? (?:am i|is this|do i have)(?: on| running)?"
    r"|(?:am i|is (?:this|my) (?:computer|system)) (?:up to date|on the latest(?: version| release)?)"
    r"|is spaced(?: linux)? up to date",
    re.IGNORECASE,
)

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
        "voxa_help",
        "set_github_owner",
        "set_noise_cancellation",
        "get_noise_cancellation",
        "set_contact_email",
        "get_contact_email",
        "forget_contact_email",
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
        "open_file",
        "open_folder",
        "install_app",
        "uninstall_app",
        "list_folder",
        "rename_file",
        "make_folder",
        "undo_file_action",
        "hidden_files",
        "type_text",
        "file_dialog",
        "send_gmail",
        "open_app",
        "close_app",
        "close_window",
        "lock_screen",
        "switch_to",
        "minimize_app",
        "update_system",
        "check_system_version",
        "maximize_app",
        "restore_app",
        "minimize_all",
        "minimize_self",
        "maximize_self",
        "change_character",
        "list_characters",
        "list_open_windows",
        "active_window_name",
        "browse",
        "read_page",
        "click_on",
        "search_site",
        "scroll",
        "go_back",
        "press_button",
        "list_buttons",
        "read_window",
        "set_timer",
        "set_reminder",
        "list_reminders",
        "cancel_reminders",
        "take_note",
        "read_notes",
        "delete_last_note",
        "open_notes",
        "list_add",
        "presentation_make",
        "finance_undo",
        "finance_count",
        "family_add",
        "family_week",
        "family_remove",
        "doc_extract",
        "event_plan",
        "conversation_summary",
        "trip_plan",
        "trip_read",
        "trip_forget",
        "travel_doc_add",
        "travel_doc_forget",
        "travel_doc_list",
        "travel_doc_when",
        "travel_options",
        "backup_now",
        "backup_last",
        "sort_folder_preview",
        "meal_idea",
        "meal_plan",
        "meal_shop",
        "project_status",
        "project_steps",
        "model_status",
        "flight_status",
        "track_package",
        "compare_prices",
        "look_up",
        "market",
        "exchange_rate",
        "fact_check",
        "convert_file",
        "list_done",
        "list_remove",
        "list_clear",
        "list_progress",
        "list_pending",
        "list_next",
        "focus_start",
        "focus_status",
        "focus_stop",
        "remember_fact",
        "recall_fact",
        "list_facts",
        "forget_fact",
        "calendar_add",
        "calendar_day",
        "calendar_next",
        "calendar_cancel",
        "calendar_conflicts",
        "calendar_add_weekly",
        "calendar_busy",
        "calendar_week",
        "expense_add",
        "expense_total",
        "expense_breakdown",
        "expense_undo",
        "expense_spreadsheet",
        "budget_set",
        "budget_status",
        "trip_cost",
        "date_remember",
        "date_when",
        "date_upcoming",
        "date_forget",
        "heads_up",
        "what_did_i_forget",
        "suggest_next_action",
        "meeting_brief",
        "take_screenshot",
        "take_picture",
        "say_in_language",
        "daily_briefing",
        "play_radio",
        "list_radio",
        "radio_genre",
        "play_tv",
        "list_tv",
        "local_weather",
        "where_am_i",
        "local_news",
        "local_events",
        "show_picture",
        "home_turn",
        "home_set",
        "home_scene",
        "home_status",
        "find_deal",
        "set_location",
        "tell_time",
        "transcribe_file",
        "tell_date",
        "world_time",
        "time_difference",
        "calculate",
        "disk_space",
        "memory_free",
        "gpu_status",
        "cpu_status",
        "system_health",
        "battery_level",
        "top_cpu",
        "top_memory",
        "system_problems",
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


def _dictation_request(text: str) -> tuple[bool, str | None]:
    """Return whether this starts dictation and its optional destination app."""
    s = _prepare(normalize(text))
    folded = s.casefold()
    if folded in _DICTATION_STARTS:
        return True, None
    # Keep this deliberately narrower than a general prefix match: for example,
    # "start typing the email" is an editing request, not a mode switch.
    for command in sorted(_DICTATION_STARTS, key=len, reverse=True):
        for separator in (" into the ", " in the ", " into ", " in "):
            prefix = command + separator
            if folded.startswith(prefix):
                target = s[len(prefix) :].strip()
                if target:
                    return True, clean_spoken_app_name(target)
    return False, None


def is_start_dictation(text: str) -> bool:
    """True when the utterance asks the assistant to start dictation."""
    return _dictation_request(text)[0]


def dictation_target(text: str) -> str | None:
    """The named destination app in a dictation request, if one was supplied."""
    matched, target = _dictation_request(text)
    return target if matched else None


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


_FLIGHT_STATUS = re.compile(
    r"(?:what(?:'s| is) the status of|track|is|check(?: if)?) flight (?P<flight>.+?)(?: is)?(?: on time| delayed| running late)?"
    r"|flight status (?:for |of )?(?P<flight2>.+)",
    re.IGNORECASE,
)
_TRACK_PACKAGE = re.compile(
    r"track (?:my |the )?(?:package|parcel|order|delivery)(?: number)? (?P<tracking>.+)"
    r"|where is (?:my )?(?:package|parcel) (?P<tracking2>.+)",
    re.IGNORECASE,
)
_COMPARE_PRICES = re.compile(
    r"compare prices (?:for |of |on )?(?P<item>.+)"
    r"|who has the (?:best |lowest |cheapest )?price (?:for |on )?(?P<item2>.+)"
    r"|where is (?P<item3>.+) cheapest",
    re.IGNORECASE,
)
_LOOK_UP_COMPANY = re.compile(
    r"(?:look up|research|tell me about|who is|what is) (?:the )?company (?P<name>.+)",
    re.IGNORECASE,
)
_LOOK_UP_WIKI = re.compile(
    r"(?:look up|research) (?:the person )?(?P<name>.+?) on wikipedia",
    re.IGNORECASE,
)
_LOOK_UP_PERSON = re.compile(r"who is (?P<name>.+)", re.IGNORECASE)
_EXCHANGE_RATE = re.compile(
    r"(?:what(?:'s| is) (?:the )?(?:price|value) of|what(?:'s| is)|how much is|convert) "
    r"(?:one |a |the )?(?P<base>.+?) (?:worth )?(?:in|to) (?P<quote>.+)"
    r"|(?:what(?:'s| is) )?(?:the )?exchange rate (?:from |for )?(?P<base2>.+?) (?:to|in) (?P<quote2>.+)",
    re.IGNORECASE,
)
_MARKET = re.compile(
    r"how is the (?:stock )?market(?: doing)?(?: today)?"
    r"|what(?:'s| is) (?:the )?(?:current )?price (?:of|for) (?P<what>.+)"
    r"|what(?:'s| is) (?:the )?(?:current )?(?:stock|share) price (?:of|for) (?P<what9>.+)"
    r"|what(?:'s| is) (?:the )?(?:current )?(?P<what_missing>(?:stock|share) price)(?: right now| today| currently)?"
    r"|what(?:'s| is) (?P<what3>.+?) (?:stock price|trading at)"
    r"|how much is (?P<what4>.+?) stock"
    r"|(?:tell me )?(?:the )?stock price (?:of|for) (?P<what5>.+)"
    r"|tell me (?P<what6>.+?) (?:stock|share) price"
    r"|how is (?P<what7>.+?) (?:stock|shares?)(?: doing)?"
    r"|what(?:'s| is) (?P<what8>.+?) (?:stock |shares? )?at"
    r"|how much is (?:a |one )?(?P<what2>bitcoin|ethereum|gold|silver)"
    r"|check (?:the )?(?:stock )?market(?: doing| today| right now)?",
    re.IGNORECASE,
)
_FACT_CHECK = re.compile(
    r"fact check (?:that )?(?P<claim>.+)"
    r"|is it true that (?P<claim2>.+)"
    r"|verify (?:that )?(?P<claim3>.+)",
    re.IGNORECASE,
)


def route(text: str) -> ToolCall | None:
    """Map a spoken command to a tool call, or None when no rule matches."""
    global _LAST_SCROLL_DIRECTION

    s = _prepare(normalize(text))
    if not s:
        return None

    def call(tool: str, **args: str) -> ToolCall:
        return ToolCall(tool, args)

    if is_start_dictation(s):
        return None
    if _BRIEFING.fullmatch(s):
        return call("daily_briefing")
    if is_compound(s):
        type_match = _TYPE.fullmatch(s)
        if type_match and not _DOCUMENT.match(type_match.group(1)):
            return ToolCall("type_text", {"text": type_match.group(1)})
        return None

    if _REPEAT_LAST.fullmatch(s):
        return call("repeat_last")
    say_in_match = _SAY_IN.fullmatch(s)
    if say_in_match and translate.language_name(say_in_match.group(2)) is not None:
        return call("say_in_language", phrase=say_in_match.group(1).strip("\"'“”"), language=say_in_match.group(2))
    after_me_match = _REPEAT_AFTER_ME.fullmatch(s)
    if after_me_match:
        return call("say_text", text=after_me_match.group(1))
    say_match = _SAY_TEXT.fullmatch(s)
    if say_match and normalize(text).strip().casefold().startswith("say "):
        return call("say_text", text=say_match.group(1))

    timer_match = _TIMER_SET.fullmatch(s)
    if timer_match:
        return call("set_timer", duration=(timer_match.group("dur") or timer_match.group("dur2") or "").strip())
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
    date_remember = _DATE_REMEMBER.fullmatch(s)
    if date_remember:
        return call("date_remember", kind=date_remember.group("kind").lower(), name=_date_name(date_remember.group("name")), when=date_remember.group("when"))
    date_bill = _DATE_BILL.fullmatch(s)
    if date_bill:
        bill_when = date_bill.group("when") or date_bill.group("paywhen")
        if dates.parse_monthly(bill_when) is None and dates.parse_day(bill_when, date.today()) is None:
            date_bill = None
    if date_bill:
        return call("date_remember", kind="bill", name=_date_name(date_bill.group("name") or date_bill.group("pay")), when=bill_when)
    date_subscription = _DATE_SUBSCRIPTION.fullmatch(s)
    if date_subscription:
        sub_when = date_subscription.group("when") or date_subscription.group("renewwhen")
        if dates.parse_monthly(sub_when) is None and dates.parse_day(sub_when, date.today()) is None:
            date_subscription = None
    if date_subscription:
        return call("date_remember", kind="subscription", name=_date_name(date_subscription.group("name") or date_subscription.group("renew")), when=sub_when)
    date_service = _DATE_SERVICE.fullmatch(s)
    if date_service and dates.parse_monthly(date_service.group("when")) is None and dates.parse_day(date_service.group("when"), date.today()) is None:
        date_service = None
    if date_service:
        return call("date_remember", kind="service", name=_date_name(date_service.group("name")), when=date_service.group("when"))
    date_due = _DATE_DUE.fullmatch(s)
    if date_due and (
        dates.parse_day(date_due.group("when"), date.today()) is None
        or re.match(r"(?:what|who|when|where|why|how)\b", date_due.group("name"), re.IGNORECASE)
    ):
        date_due = None  # "what is due process": not a deadline unless a real date follows
    if date_due:
        return call("date_remember", kind="deadline", name=_date_name(date_due.group("name")), when=date_due.group("when"))
    date_deadline = _DATE_DEADLINE.fullmatch(s)
    if date_deadline:
        return call("date_remember", kind="deadline", name=_date_name(date_deadline.group("name")), when=date_deadline.group("when"))
    if _DATE_BILLS_QUESTION.fullmatch(s):
        return call("date_upcoming", kind="bill")
    if _DATE_SUBSCRIPTIONS_QUESTION.fullmatch(s):
        return call("date_upcoming", kind="subscription", days="365")
    when_bill = _DATE_WHEN_BILL.fullmatch(s)
    if when_bill:
        return call("date_when", name=_date_name(when_bill.group("name")))
    when_renew = _DATE_WHEN_RENEW.fullmatch(s)
    if when_renew:
        return call("date_when", name=_date_name(when_renew.group("name")))
    date_when = _DATE_WHEN.fullmatch(s)
    if date_when:
        return call("date_when", name=_date_name(date_when.group("name") or date_when.group("due")))
    if _DATE_UPCOMING.fullmatch(s):
        return call("date_upcoming")
    date_forget = _DATE_FORGET.fullmatch(s)
    if date_forget:
        return call("date_forget", name=_date_name(date_forget.group("name") or date_forget.group("deadline") or date_forget.group("bill") or date_forget.group("subscription") or date_forget.group("service")))
    trip_add = _TRIP_ADD.fullmatch(s)
    if trip_add:
        return call("expense_add", amount=trip_add.group("a1"), what=trip_add.group("w1"), tag=trip_add.group("t1"))
    trip_cost = _TRIP_COST.fullmatch(s)
    if trip_cost:
        return call("trip_cost", name=trip_cost.group("n1") or trip_cost.group("n2"))
    budget_set = _BUDGET_SET.fullmatch(s)
    if budget_set:
        return call(
            "budget_set",
            category=budget_set.group("c1") or budget_set.group("c2"),
            amount=budget_set.group("a1") or budget_set.group("a2"),
        )
    budget_status = _BUDGET_STATUS.fullmatch(s)
    if budget_status:
        return call("budget_status", category=budget_status.group("c1") or budget_status.group("c2") or "")
    expense_add = _EXPENSE_ADD.fullmatch(s)
    if expense_add:
        return call(
            "expense_add",
            amount=expense_add.group("a1") or expense_add.group("a2") or expense_add.group("a3") or expense_add.group("a4"),
            what=expense_add.group("w1") or expense_add.group("w2") or expense_add.group("w3") or expense_add.group("w4"),
        )
    expense_total = _EXPENSE_TOTAL.fullmatch(s)
    if expense_total:
        return call(
            "expense_total",
            period=expense_total.group("p1") or expense_total.group("p2") or "",
            category=expense_total.group("c1") or expense_total.group("c2") or "",
        )
    expense_breakdown = _EXPENSE_BREAKDOWN.fullmatch(s)
    if expense_breakdown:
        return call("expense_breakdown", period=expense_breakdown.group("p1") or expense_breakdown.group("p2") or "")
    expense_undo = _EXPENSE_UNDO.fullmatch(s)
    if expense_undo:
        return call("expense_undo")
    if _EXPENSE_SHEET.fullmatch(s):
        return call("expense_spreadsheet")
    event_plan = _EVENT_PLAN.fullmatch(s)
    if event_plan:
        event = event_plan.group("event").strip()
        tail = event_plan.group("tail").strip()
        when = re.sub(r"^(?:for|on)\s+", "", tail, flags=re.IGNORECASE).strip()
        return call("event_plan", event=event, when=when)
    for _cal_add in (_CAL_ADD1, _CAL_ADD2, _CAL_ADD3, _CAL_ADD4):
        _cal_m = _cal_add.fullmatch(s)
        if _cal_m and _cal_m.group("when") and split_when(_cal_m.group("when"), date.today()) is not None:
            return call("calendar_add", title=_cal_m.group("title").strip(), when=_cal_m.group("when").strip())
    cal_day = _CAL_DAY1.fullmatch(s) or _CAL_DAY2.fullmatch(s) or _CAL_DAY3.fullmatch(s) or _CAL_DAY4.fullmatch(s)
    if cal_day:
        return call("calendar_day", day=cal_day.group("day") or "")
    if _CAL_NEXT.fullmatch(s):
        return call("calendar_next")
    cal_cancel = _CAL_CANCEL.fullmatch(s)
    if cal_cancel and _CAL_NAMED.search(s):
        return call("calendar_cancel", title=cal_cancel.group("title"))
    travel_add = _TRAVEL_DOC_ADD.fullmatch(s)
    if travel_add:
        return call("travel_doc_add", document=travel_add.group("doc").strip(), expires=travel_add.group("expires").strip())
    travel_when = _TRAVEL_DOC_WHEN.fullmatch(s)
    if travel_when:
        return call("travel_doc_when", document=travel_when.group("doc").strip())
    if _TRAVEL_DOC_LIST.fullmatch(s):
        return call("travel_doc_list")
    travel_forget = _TRAVEL_DOC_FORGET.fullmatch(s)
    if travel_forget:
        return call("travel_doc_forget", document=travel_forget.group("doc").strip())
    mem_remember = _MEMORY_REMEMBER.fullmatch(s)
    if mem_remember:
        return call("remember_fact", fact=mem_remember.group("fact"))
    mem_recall = _MEMORY_RECALL.fullmatch(s)
    if mem_recall:
        return call("recall_fact", topic=mem_recall.group("t1") or mem_recall.group("t2") or mem_recall.group("t3") or mem_recall.group("t4"))
    if _MEMORY_LIST.fullmatch(s):
        return call("list_facts")
    mem_forget = _MEMORY_FORGET.fullmatch(s)
    if mem_forget:
        return call("forget_fact", topic=mem_forget.group("t1") or mem_forget.group("t2"))
    list_progress = _LIST_PROGRESS.fullmatch(s)
    if list_progress:
        return call("list_progress", list=_list_word(list_progress.group("list")))
    if _LIST_PENDING.fullmatch(s):
        return call("list_pending")
    if _LIST_NEXT.fullmatch(s):
        return call("list_next")
    focus_start = _FOCUS_START.fullmatch(s)
    if focus_start:
        return call(
            "focus_start",
            duration=focus_start.group("dur1") or focus_start.group("dur2") or "",
            what=focus_start.group("what") or "",
        )
    focus_need = _FOCUS_NEED.fullmatch(s)
    if focus_need:
        return call("focus_start", duration=focus_need.group("dur") or "", what="")
    if _FOCUS_POMODORO.fullmatch(s):
        return call("focus_start", duration="", what="")
    focus_protect = _FOCUS_PROTECT.fullmatch(s)
    if focus_protect:
        return call("focus_start", duration=focus_protect.group("dur"), what=focus_protect.group("what") or "")
    if _FOCUS_STATUS.fullmatch(s):
        return call("focus_status")
    if _FOCUS_STOP.fullmatch(s):
        return call("focus_stop")
    convert_to = _CONVERT_TO.fullmatch(s)
    if convert_to and target_extension(convert_to.group("fmt")) is not None:
        return call("convert_file", name=convert_to.group("name"), format=convert_to.group("fmt"), folder=convert_to.group("folder") or "")
    convert_as = _CONVERT_AS.fullmatch(s)
    if convert_as and target_extension(convert_as.group("fmt")) is not None:
        return call("convert_file", name=convert_as.group("name"), format=convert_as.group("fmt"), folder="")
    convert_make = _CONVERT_MAKE.fullmatch(s)
    if convert_make and target_extension(convert_make.group("fmt")) is not None:
        return call("convert_file", name=convert_make.group("name"), format=convert_make.group("fmt"), folder="")
    transcribe = _TRANSCRIBE.fullmatch(s)
    if transcribe:
        return call("transcribe_file", name=transcribe.group("name"), folder=transcribe.group("folder") or "")
    if _CAL_CONFLICTS.fullmatch(s):
        return call("calendar_conflicts")
    for _cal_weekly in (_CAL_WEEKLY1, _CAL_WEEKLY2, _CAL_WEEKLY3):
        _cal_w = _cal_weekly.fullmatch(s)
        if _cal_w and parse_time(_cal_w.group("time")) is not None:
            return call(
                "calendar_add_weekly",
                title=_cal_w.group("title").strip(),
                weekday=_cal_w.group("day").lower(),
                time=_cal_w.group("time").strip(),
            )
    if _CAL_WEEK_Q.fullmatch(s):
        return call("calendar_week", person="")
    week_person = _CAL_WEEK_PERSON.fullmatch(s)
    if week_person:
        return call(
            "calendar_week",
            person=week_person.group("name") or week_person.group("name2") or week_person.group("name3"),
        )
    if _CAL_BUSY.fullmatch(s):
        return call("calendar_busy")
    if _WHY_SLOW.fullmatch(s):
        return call("top_cpu")
    if _WHY_MEMORY.fullmatch(s):
        return call("top_memory")
    if _TROUBLE.fullmatch(s):
        return call("system_problems")
    if _HEADS_UP.fullmatch(s):
        return call("heads_up")
    if _FORGOTTEN.fullmatch(s):
        return call("what_did_i_forget")
    if _SUGGEST_NEXT.fullmatch(s):
        return call("suggest_next_action")
    meeting_brief = _MEETING_BRIEF.fullmatch(s)
    if meeting_brief:
        return call(
            "meeting_brief",
            title=meeting_brief.group("title") or meeting_brief.group("title2"),
        )
    project_status = _PROJECT_STATUS.fullmatch(s)
    if project_status:
        name = project_status.group("n1") or project_status.group("n2") or project_status.group("n3")
        return call("project_status", name=name)
    if _MODEL_STATUS.fullmatch(s):
        return call("model_status")
    meal_shop = _MEAL_SHOP.fullmatch(s)
    if meal_shop:
        return call("meal_shop", name=meal_shop.group("name") or meal_shop.group("name2") or "")
    meal_plan = _MEAL_PLAN.fullmatch(s)
    if meal_plan:
        return call("meal_plan")
    meal_idea = _MEAL_IDEA.fullmatch(s)
    if meal_idea:
        return call("meal_idea", tag=meal_idea.group("tag") or "")
    backup_now = _BACKUP_NOW.fullmatch(s)
    if backup_now:
        return call("backup_now")
    backup_last = _BACKUP_LAST.fullmatch(s)
    if backup_last:
        return call("backup_last")
    sort_preview = _SORT_PREVIEW.fullmatch(s)
    if sort_preview:
        spoken = sort_preview.group(1) or sort_preview.group(2) or ""
        if folder_key(spoken) is not None:
            return call("sort_folder_preview", folder=spoken)
    project_steps = _PROJECT_STEPS.fullmatch(s)
    if project_steps:
        goal = project_steps.group("g1") or project_steps.group("g2") or project_steps.group("g3") or project_steps.group("g4")
        return call("project_steps", goal=goal)
    trip_plan = _TRIP_PLAN.fullmatch(s)
    if trip_plan:
        days = (trip_plan.group("days") or "").lower()
        tail = trip_plan.group("tail")
        if not days:
            day_match = _TRIP_DAY_WORD.search(tail)
            if day_match:
                days = day_match.group(1).lower()
                tail = tail[: day_match.start()] + tail[day_match.end():]
        tail = re.sub(rf"\b(?:{_TRIP_NOUN})\b", "", tail, flags=re.IGNORECASE)
        tail = re.sub(r"^\s*days?\b", "", tail, flags=re.IGNORECASE)
        place = _TRIP_CONNECTOR.split(tail)[-1]
        place = re.sub(r"\s+", " ", place).strip()
        return call("trip_plan", place=place, days=days)
    trip_read = _TRIP_READ.fullmatch(s)
    if trip_read:
        return call("trip_read", place=trip_read.group("place").strip())
    trip_forget = _TRIP_FORGET.fullmatch(s)
    if trip_forget:
        return call("trip_forget", place=trip_forget.group("place").strip())
    if _CONVO_SUMMARY.fullmatch(s):
        return call("conversation_summary")
    doc_extract = _DOC_EXTRACT.fullmatch(s)
    if doc_extract:
        found = doc_extract.groupdict()
        name = next(found[key] for key in ("n1", "n2", "n3") if found[key])
        return call("doc_extract", name=name.strip())
    travel = _TRAVEL_OPTIONS.fullmatch(s)
    if travel:
        found = travel.groupdict()
        if found["what"]:
            return call("travel_options", what=found["what"], place=found["place"].strip(), problem=found["problem"])
        return call("travel_options", what=found["what2"], place=found["place2"].strip(), problem="missed")
    family_add = _FAMILY_ADD.fullmatch(s)
    if family_add and split_when(family_add.group("when"), date.today()) is not None:
        return call(
            "family_add",
            person=family_add.group("person").strip(),
            event=family_add.group("event").strip(),
            when=family_add.group("when").strip(),
        )
    if _FAMILY_WEEK.fullmatch(s):
        return call("family_week", person="")
    family_remove = _FAMILY_REMOVE.fullmatch(s)
    if family_remove:
        return call(
            "family_remove",
            person=family_remove.group("person").strip(),
            event=family_remove.group("event").strip(),
        )
    finance_sort = _FINANCE_SORT.fullmatch(s)
    if finance_sort:
        return call("finance_sort")
    finance_undo = _FINANCE_UNDO.fullmatch(s)
    if finance_undo:
        return call("finance_undo")
    finance_count = _FINANCE_COUNT.fullmatch(s)
    if finance_count:
        kind = finance_count.group("kind1") or finance_count.group("kind2")
        return call("finance_count", kind=kind)
    presentation = _PRESENTATION.fullmatch(s)
    if presentation:
        return call(
            "presentation_make",
            topic=presentation.group("topic").strip(),
            count=presentation.group("count") or "",
        )
    list_add = _LIST_ADD.fullmatch(s)
    if list_add:
        return call("list_add", item=list_add.group("item"), list=_list_word(list_add.group("list")))
    list_read = _LIST_READ.fullmatch(s)
    if list_read:
        return call("list_read", list=_list_word(list_read.group("a") or list_read.group("b") or list_read.group("c")))
    list_done = _LIST_DONE.fullmatch(s)
    if list_done:
        found = list_done.groupdict()
        return call(
            "list_done",
            item=next(found[key] for key in ("i1", "i2", "i3", "i4") if found[key]),
            list=_list_word(next((found[key] for key in ("l1", "l2", "l3", "l4") if found[key]), "")),
        )
    list_remove = _LIST_REMOVE.fullmatch(s)
    if list_remove:
        return call("list_remove", item=list_remove.group("item"), list=_list_word(list_remove.group("list")))
    list_clear = _LIST_CLEAR.fullmatch(s)
    if list_clear:
        return call("list_clear", list=_list_word(list_clear.group("list")))
    take_note = _TAKE_NOTE.fullmatch(s)
    if take_note:
        return call("take_note", text=take_note.group(1))
    note_says = _NOTE_SAYS.fullmatch(s)
    if note_says:
        return call("take_note", text=note_says.group(1))
    write_down = _WRITE_DOWN.fullmatch(s)
    if write_down:
        return call("take_note", text=write_down.group(1))
    if _READ_NOTES.fullmatch(s):
        return call("read_notes")
    if _DELETE_LAST_NOTE.fullmatch(s):
        return call("delete_last_note")
    if _OPEN_NOTES.fullmatch(s):
        return call("open_notes")
    if _TAKE_SCREENSHOT.fullmatch(s):
        return call("take_screenshot")
    if _TAKE_PICTURE.fullmatch(s):
        return call("take_picture")
    if _SHOW_PICTURE.fullmatch(s):
        return call("show_picture")

    browse_match = _BROWSE.fullmatch(s)
    if browse_match and "." in browse_match.group(1):
        return call("browse", url=browse_match.group(1))
    recite_by = _RECITE_BY.fullmatch(s)
    if recite_by:
        return call("recite_poem", author=recite_by.group(1))
    if re.fullmatch(r"(?:recite|read|say)\s+(?:me\s+)?(?:a|some|another)\s+poem", s, re.IGNORECASE):
        return call("recite_poem")
    recite_title = _RECITE_TITLE.fullmatch(s)
    if recite_title:
        if recite_title.group(2):
            return call("recite_poem", title=recite_title.group(1), author=recite_title.group(2))
        return call("recite_poem", title=recite_title.group(1))
    # Before read_page: "read this text file" must not read the web page.
    if _READ_THIS_FILE.fullmatch(s):
        return call("read_text_file")
    read_file_match = _READ_FILE.fullmatch(s)
    if read_file_match:
        return call("read_text_file", name=read_file_match.group(1))
    if _READ_PAGE.fullmatch(s):
        return call("read_page")
    if _READ_CLIPBOARD.fullmatch(s):
        return call("read_clipboard_aloud")
    ui_in_match = _UI_PRESS_IN.fullmatch(s)
    if ui_in_match:
        return call("press_button", label=ui_in_match.group(1), app=ui_in_match.group(2))
    click_probe = _CLICK.fullmatch(s)
    if not (click_probe and click_probe.group(2)):
        ui_button_match = _UI_PRESS_BUTTON.fullmatch(s)
        if ui_button_match:
            return call("press_button", label=ui_button_match.group(1))
    ui_list_match = _UI_LIST_BUTTONS.fullmatch(s)
    if ui_list_match:
        app = next((g for g in ui_list_match.groups() if g), "")
        return call("list_buttons", app=app)
    if _UI_READ_WINDOW.fullmatch(s):
        return call("read_window")
    click_match = _CLICK.fullmatch(s)
    if click_match:
        text_value = next(g for g in click_match.groups() if g is not None)
        return call("click_on", text=text_value)
    search_match = _SEARCH_SITE.fullmatch(s)
    if search_match:
        return call("search_site", query=search_match.group(1))
    scroll_match = _SCROLL.fullmatch(s)
    if scroll_match:
        if scroll_match.group(2) or scroll_match.group(3):
            amount = scroll_match.group(2) or scroll_match.group(3)
        elif "a little" in s:
            amount = "little"
        else:
            amount = "page"
        if amount == "top":
            direction = "up"
        elif amount == "bottom":
            direction = "down"
        else:
            direction = "up" if "up" in s else "down"
        _LAST_SCROLL_DIRECTION = direction
        return call("scroll", direction=direction, amount=amount)
    if _KEEP_SCROLLING.fullmatch(s):
        return call("scroll", direction=_LAST_SCROLL_DIRECTION, amount="page")
    if _GO_BACK.fullmatch(s):
        return call("go_back")

    if _TV_LIST.fullmatch(s):
        return call("list_tv")
    if _TV_PLAY.fullmatch(s):
        return call("play_tv", name="")
    tv_match = _TV_CHANNEL.fullmatch(s)
    if tv_match and livetv.is_channel(tv_match.group(1)) and re.search(
        r"\b(?:tv|television|channel|live)\b", s, re.IGNORECASE
    ) or (tv_match and livetv.clean_request(tv_match.group(1)) in _TV_NAMES):
        return call("play_tv", name=livetv.find(tv_match.group(1)).name)
    if _WHERE_AM_I.fullmatch(s):
        return call("where_am_i")
    if _LOCAL_NEWS.fullmatch(s):
        return call("local_news")
    events_match = _LOCAL_EVENTS.fullmatch(s)
    if events_match:
        when = (events_match.group(2) or events_match.group(3) or "").casefold()
        return call("local_events", what=" ".join(events_match.group(1).casefold().split()), when=" ".join(when.split()))
    weather_in = _WEATHER_IN.fullmatch(s)
    if weather_in and not re.fullmatch(r"my\s+area|here|town", weather_in.group(1), re.IGNORECASE):
        return call("local_weather", place=weather_in.group(1).strip())
    if _LOCAL_WEATHER.fullmatch(s) or weather_in:
        return call("local_weather")

    from_match = _DIRECTIONS_FROM.fullmatch(s)
    if from_match:
        return call("directions", origin=from_match.group(1).strip(), destination=from_match.group(2).strip())
    loose_match = _DIRECTIONS_LOOSE.fullmatch(s)
    if loose_match:
        return call("directions", destination=loose_match.group(1).strip())
    directions_match = _DIRECTIONS.fullmatch(s)
    if directions_match:
        destination = next(g for g in directions_match.groups() if g).strip()
        # "navigate to github.com" is a web address, not a place: leave it to the URL rules below.
        if not re.fullmatch(r"[\w-]+(?:\.[\w-]+)+(?:/\S*)?", destination):
            mode = re.search(r"\b(walking|walk|on foot|biking|bike|cycling|transit|bus|train)\b", s, re.IGNORECASE)
            if mode:
                return call("directions", destination=destination, mode=mode.group(1).casefold())
            return call("directions", destination=destination)
    nearby_match = _NEAREST.fullmatch(s) or _NEAR_ME.fullmatch(s)
    if nearby_match:
        return call("find_nearby", what=nearby_match.group(1).strip())

    for pattern in (_IMAGE_SEARCH_FOR, _IMAGE_SEARCH_DO, _IMAGE_LOOK_LIKE, _IMAGE_OF, _IMAGE_BARE, _IMAGE_KIND):
        image_match = pattern.fullmatch(s)
        if not image_match:
            continue
        query = image_match.group(1)
        if pattern is _IMAGE_KIND:
            if query.lower() in _IMAGE_OWN_WORDS:
                continue
            if image_match.group(0).lower().endswith("wallpapers"):
                query = f"{query} wallpapers"
        return call("image_search", query=query)

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

    set_email_match = _SET_EMAIL.fullmatch(s)
    if set_email_match:
        name = set_email_match.group(1)
        raw_email = set_email_match.group(2)
        spoken_email = spoken_address(raw_email)
        if not spoken_email and "@" not in raw_email and " at " not in raw_email.lower():
            low = raw_email.lower()
            nlow = name.lower()
            if nlow and low.startswith(nlow):
                rest = low[len(nlow):]
                for glue in ("at", "an"):
                    if rest.startswith(glue):
                        tail = rest[len(glue):]
                        if re.search(r"\.[a-z]{2,}$", tail):
                            spoken_email = f"{nlow}@{tail}"
                            break
        if spoken_email:
            address = spoken_email if " " not in raw_email else raw_email
            return call("set_contact_email", name=name, address=address)
    get_email_match = _GET_EMAIL.fullmatch(s)
    if get_email_match:
        return call("get_contact_email", name=get_email_match.group(1))
    forget_email_match = _FORGET_EMAIL.fullmatch(s)
    if forget_email_match:
        return call("forget_contact_email", name=forget_email_match.group(1))

    if _RADIO_LIST.fullmatch(s):
        return call("list_radio")
    if _RADIO_WHAT.fullmatch(s):
        return call("radio_genre")
    if _HELP_ASK.fullmatch(s):
        return call("voxa_help")
    help_topic = _HELP_TOPIC.fullmatch(s)
    if help_topic:
        topic = help_topic.group(1) or help_topic.group(2)
        if topic_for(topic) is not None:
            return call("voxa_help", topic=topic)

    world_time_match = _WORLD_TIME.fullmatch(s)
    if world_time_match and worldtime.zone_for(world_time_match.group(1)) is not None:
        return call("world_time", place=world_time_match.group(1))
    time_diff_match = _TIME_DIFF.fullmatch(s)
    if time_diff_match:
        place = time_diff_match.group(1) or time_diff_match.group(2) or time_diff_match.group(3)
        if worldtime.zone_for(place) is not None:
            return call("time_difference", place=place)
    if _TELL_TIME.fullmatch(s):
        return call("tell_time")
    if _TELL_DATE.fullmatch(s):
        return call("tell_date")
    if calc.answer(s) is not None:
        return call("calculate", expression=s)
    if _DISK_SPACE.fullmatch(s):
        return call("disk_space")
    if _MEMORY_FREE.fullmatch(s):
        return call("memory_free")
    if _GPU_STATUS.fullmatch(s):
        return call("gpu_status")
    if _CPU_STATUS.fullmatch(s):
        return call("cpu_status")
    if _SYSTEM_HEALTH.fullmatch(s):
        return call("system_health")
    if _BATTERY_LEVEL.fullmatch(s):
        return call("battery_level")

    if _WHICH_WINDOW.fullmatch(s):
        return call("active_window_name")
    if _WHATS_OPEN.fullmatch(s):
        return call("list_open_windows")

    if _CHECK_VERSION.fullmatch(s):
        return call("check_system_version")

    if _LIST_CHARACTERS.fullmatch(s):
        return call("list_characters")

    # Live quote questions must be handled before the general-question handoff below.
    exchange_match = _EXCHANGE_RATE.fullmatch(s)
    if exchange_match:
        base_text = exchange_match.group("base") or exchange_match.group("base2")
        quote_text = exchange_match.group("quote") or exchange_match.group("quote2")
        base = lookups.currency_code(base_text)
        quote = lookups.currency_code(quote_text)
        if base and quote:
            return call("exchange_rate", base=base, quote=quote)
    market_match = _MARKET.fullmatch(s)
    if market_match:
        return call("market", what=(market_match.group("what") or market_match.group("what2")
                                    or market_match.group("what3") or market_match.group("what4")
                                    or market_match.group("what5") or market_match.group("what6")
                                    or market_match.group("what7") or market_match.group("what8")
                                    or market_match.group("what9") or market_match.group("what_missing") or ""))

    if _QUESTION.match(s):
        return None

    if _OPEN_MAIL.fullmatch(s):
        return call("open_site", name="gmail")
    if _COMPOSE.fullmatch(s):
        return call("compose_gmail")

    set_channel_match = _SET_CHANNEL.fullmatch(s)
    if set_channel_match:
        return call("set_youtube_channel", name=set_channel_match.group(1))

    set_owner_match = _SET_GITHUB_OWNER.fullmatch(s)
    if set_owner_match:
        return call("set_github_owner", name=set_owner_match.group(1))

    if _NOISE_ON.fullmatch(s):
        return call("set_noise_cancellation", state="on")
    if _NOISE_OFF.fullmatch(s):
        return call("set_noise_cancellation", state="off")
    if _NOISE_GET.fullmatch(s):
        return call("get_noise_cancellation")

    on_app = _PLAY_ON_APP.fullmatch(s)
    if on_app and on_app.group(2).casefold() not in _NOT_AN_APP and len(on_app.group(2).split()) <= 3:
        return call("play_on_app", query=on_app.group(1).strip(), app=on_app.group(2).strip())

    theme_match = _CHANGE_THEME.fullmatch(s)
    if theme_match:
        return call("change_theme", name=next(g for g in theme_match.groups() if g).strip())

    radio_match = _RADIO_PLAY.fullmatch(s)
    if radio_match:
        return call("play_radio", name=radio.clean_request(radio_match.group(1) or ""))
    station_match = _PLAY_ANY.fullmatch(s)
    if station_match and radio.is_station(station_match.group(1)):
        return call("play_radio", name=radio.clean_request(station_match.group(1)))

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

    if _UPDATE_SYSTEM.fullmatch(s):
        return call("update_system")

    install_match = _INSTALL_APP.fullmatch(s)
    if install_match:
        captured = (install_match.group(1) or install_match.group(2) or "").strip()
        if captured and not any(
            captured == skip or captured.startswith(skip) for skip in _INSTALL_SKIP
        ):
            return call("install_app", name=captured)

    uninstall_match = _UNINSTALL_APP.fullmatch(s)
    if uninstall_match:
        return call("uninstall_app", name=uninstall_match.group(1).strip())

    open_file_folder_match = _OPEN_FILE_FOLDER.fullmatch(s)
    if open_file_folder_match and folder_key(open_file_folder_match.group(2)) is not None:
        return call(
            "open_file",
            name=open_file_folder_match.group(1),
            folder=open_file_folder_match.group(2),
        )
    open_file_match = _OPEN_FILE.fullmatch(s)
    if open_file_match and not (
        open_file_match.group(1).casefold() == "manager"
        and s.casefold() in {"open the file manager", "open my file manager"}
    ):
        return call("open_file", name=open_file_match.group(1), folder="")

    spoken_url = _SPOKEN_URL.fullmatch(s)
    # Only for an address said as words ("open air ships dot com"); a typed one is left to the rules below.
    if spoken_url and (" " in spoken_url.group(1).strip() or re.search(r"\sdot\s", s, re.IGNORECASE)):
        site = re.sub(r"[^a-z0-9-]", "", spoken_url.group(1).casefold())
        if site:
            return call("open_url", url=f"{site}.{spoken_url.group(2).casefold()}")
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
    if _CUBE_SPIN.fullmatch(s):
        return call("compiz_control", action="spin")
    if _CUBE_SPIN_STOP.fullmatch(s):
        return call("compiz_control", action="spin_stop")
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

    hidden_files_match = _HIDDEN_FILES.fullmatch(s)
    if hidden_files_match:
        verb = hidden_files_match.group(1).casefold()
        if verb in ("show", "check", "turn on"):
            return call("hidden_files", show="on")
        if verb in ("hide", "uncheck", "turn off"):
            return call("hidden_files", show="off")
        return call("hidden_files", show="toggle")

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

    if _CLOSE_YOURSELF.fullmatch(s) or _CLOSE_YOURSELF.fullmatch(_prepare(text)):
        return call("close_yourself")
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
        character = find_character(name)
        if character is not None:
            return call("change_character", name=character)
        if site_url(name) is not None and "." not in name:
            return call("open_site", name=name.casefold())
        return call("switch_to", name=name)
    character_match = _CHARACTER_CHANGE.fullmatch(s)
    if character_match:
        character = find_character(character_match.group(1))
        if character is not None:
            return call("change_character", name=character)
    if _LIST_CHARACTERS.fullmatch(s):
        return call("list_characters")
    if _MINIMIZE_ALL.fullmatch(s):
        return call("minimize_all")
    if _RESTORE_ALL.fullmatch(s):
        return call("restore_all")
    if _MINIMIZE_ACTIVE.fullmatch(s):
        return call("minimize_app", name="")
    if _MINIMIZE_SELF.fullmatch(s):
        return call("minimize_self")
    if _MAXIMIZE_SELF.fullmatch(s):
        return call("maximize_self")
    minimize_match = _MINIMIZE_APP.fullmatch(s)
    if minimize_match:
        return call("minimize_app", name=minimize_match.group(1))
    if _MAXIMIZE_ACTIVE.fullmatch(s):
        return call("maximize_app", name="")
    maximize_match = _MAXIMIZE_APP.fullmatch(s)
    if maximize_match:
        return call("maximize_app", name=maximize_match.group(1) or maximize_match.group(2))
    if _UNDO_FILE.fullmatch(s):
        if s.lower() == "undo that":
            return call("undo_file_action", otherwise="key")
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

    if _READ_SELECTION.fullmatch(s):
        return call("read_selection")
    if _SUMMARIZE_SELECTION.fullmatch(s):
        return call("summarize_selection")

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
    flight_match = _FLIGHT_STATUS.fullmatch(s)
    if flight_match:
        return call("flight_status", flight=flight_match.group("flight") or flight_match.group("flight2"))
    package_match = _TRACK_PACKAGE.fullmatch(s)
    if package_match:
        return call("track_package", tracking=package_match.group("tracking") or package_match.group("tracking2"))
    price_match = _COMPARE_PRICES.fullmatch(s)
    if price_match:
        return call("compare_prices",
                    item=price_match.group("item") or price_match.group("item2") or price_match.group("item3"))
    company_match = _LOOK_UP_COMPANY.fullmatch(s)
    if company_match:
        return call("look_up", name=company_match.group("name"), kind="company")
    wiki_match = _LOOK_UP_WIKI.fullmatch(s)
    if wiki_match:
        return call("look_up", name=wiki_match.group("name"), kind="person")
    person_match = _LOOK_UP_PERSON.fullmatch(s)
    if person_match and lookups.person_words_ok(person_match.group("name")):
        return call("look_up", name=person_match.group("name"), kind="person")
    fact_match = _FACT_CHECK.fullmatch(s)
    if fact_match:
        return call("fact_check",
                    claim=fact_match.group("claim") or fact_match.group("claim2") or fact_match.group("claim3"))
    app_match = _OPEN_APP.fullmatch(s)
    if app_match:
        return call("open_app", name=clean_spoken_app_name(app_match.group(1)))
    return None


def route_spoken(text: str) -> tuple[ToolCall | None, str]:
    """route(), forgiving of someone else talking in the room.

    The microphone sometimes delivers the request together with a sentence that was not meant for
    Voxa ("Add pay the rent to my to-do list. Dude, it's bothering you."). When the whole text is
    not a command but exactly one of its sentences is, that sentence is the request. Returns the
    call and the words it was made from.
    """
    call = route(text)
    if call is not None:
        return call, text
    parts = [part.strip(" ,") for part in re.split(r"(?<=[.!?])\s+|\s+,\s+", text) if len(part.split()) >= 2]
    if len(parts) < 2:
        return None, text
    found = [(route(part), part) for part in parts]
    found = [(call, part) for call, part in found if call is not None]
    return found[0] if len(found) == 1 else (None, text)
