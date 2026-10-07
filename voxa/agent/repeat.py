""""Do that again": which sentences mean it, and which commands are safe to run twice."""

NEVER_REPEAT = frozenset({
    "trash_file", "empty_trash", "send_gmail", "compose_gmail", "lock_screen", "type_text",
    "set_youtube_channel", "make_folder", "rename_file", "move_file", "cleanup_text",
    "close_app", "repeat_last",
})

REPEAT_PHRASES = frozenset({
    "again", "do that again", "do it again", "do that once more", "once more",
    "one more time", "same again", "and again", "repeat that action",
    "do the same again", "more",
})


def _normalize(text: str) -> str:
    t = text.lower()
    for ch in ".,!?\u2014":
        t = t.replace(ch, " ")
    words = t.split()
    if words and words[0] in ("voxa", "and"):
        words = words[1:]
    if words and words[-1] == "please":
        words = words[:-1]
    return " ".join(words)


def is_repeat_action(text: str) -> bool:
    return _normalize(text) in REPEAT_PHRASES


def repeatable(tool: str) -> bool:
    return tool not in NEVER_REPEAT
