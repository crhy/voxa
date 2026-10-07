from __future__ import annotations

TOPICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "apps": (
        "I can open, close, minimize and switch between your programs.",
        ("open LibreOffice", "close Brave", "minimize Pluma", "show the desktop"),
    ),
    "files": (
        "I can copy, move, rename, delete and find your files, and undo the last of those.",
        (
            "copy report from Downloads to Documents",
            "find the file called budget",
            "what's in my Documents",
            "empty trash",
        ),
    ),
    "music": (
        "I can play music and video without adverts and control the volume.",
        (
            "play Dirty Old Town by the Pogues",
            "pause the music",
            "volume up",
            "play the latest video from my channel",
        ),
    ),
    "web": (
        "I can search the web, open sites, and find the cheapest flights, hotels, cars and products.",
        (
            "what is the price of Bitcoin",
            "get me the cheapest ticket to Hawaii",
            "open my email",
            "directions to the zoo",
        ),
    ),
    "writing": (
        "I can type what you say, clean up your text, write an email you dictate, and post a GitHub issue.",
        (
            "start dictation",
            "clean up the text",
            "send an email to my mom",
            "post an issue to GitHub for Voxa",
        ),
    ),
    "time": (
        "I can set timers and reminders and run routines you teach me.",
        ("set a timer for five minutes", "remind me at three to call Bob", "cancel my reminders"),
    ),
    "me": (
        "You can pause me, interrupt me, ask for another language, or have me repeat a command.",
        ("Voxa, pause", "can I get that in German", "again", "say that again"),
    ),
}

ALIASES = {
    "programs": "apps",
    "applications": "apps",
    "windows": "apps",
    "file": "files",
    "folders": "files",
    "documents": "files",
    "video": "music",
    "videos": "music",
    "youtube": "music",
    "volume": "music",
    "sound": "music",
    "internet": "web",
    "browser": "web",
    "browsing": "web",
    "shopping": "web",
    "travel": "web",
    "search": "web",
    "email": "writing",
    "mail": "writing",
    "dictation": "writing",
    "typing": "writing",
    "text": "writing",
    "github": "writing",
    "timers": "time",
    "reminders": "time",
    "routines": "time",
    "you": "me",
    "yourself": "me",
    "talking": "me",
}


def topic_for(spoken: str) -> str | None:
    """Map a spoken topic word to a TOPICS key, or None when it is not one of ours."""
    word = spoken.strip().casefold()
    for prefix in ("the ", "my ", "your "):
        if word.startswith(prefix):
            word = word[len(prefix) :].strip()
    if word in TOPICS:
        return word
    if word in ALIASES:
        return ALIASES[word]
    if word.endswith("s") and (word[:-1] in TOPICS or word[:-1] in ALIASES):
        base = word[:-1]
        return base if base in TOPICS else ALIASES[base]
    if word + "s" in TOPICS:
        return word + "s"
    if word + "s" in ALIASES:
        return ALIASES[word + "s"]
    return None


def overview() -> str:
    return (
        "I can open and control your programs, manage files, play music and video, "
        "search and shop on the web, type and tidy what you write, and keep timers and "
        "reminders. Ask me, for example: what can you do with files?"
    )


def topic_help(key: str) -> str:
    summary, examples = TOPICS[key]
    tried = ", or ".join(examples[:3])
    text = f"{summary} Try saying: {tried}"
    if text[-1] not in ".?":
        text += "."
    return text
