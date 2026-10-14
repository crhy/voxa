from __future__ import annotations

import re
from datetime import datetime

BACKUP_FOLDER = "Voxa Backups"

_DIR_NAME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{4}$")


def backup_dir_name(now: datetime) -> str:
    return now.strftime("%Y-%m-%d %H%M")


def parse_listing(ls_text: str) -> list[str]:
    names = [line.strip() for line in ls_text.splitlines()]
    dated = [name for name in names if _DIR_NAME.match(name)]
    dated.sort()
    return dated


def _clock(when: datetime) -> str:
    hour = when.hour % 12
    if hour == 0:
        hour = 12
    period = "AM" if when.hour < 12 else "PM"
    return f"{hour}:{when.minute:02d} {period}"


def spoken_done(count: int, name: str) -> str:
    if count == 1:
        return f"I backed up 1 file into {name}."
    return f"I backed up {count} files into {name}."


def spoken_last(names: list[str], now: datetime) -> str:
    if not names:
        return "You have no backups yet."
    newest = names[-1]
    when = datetime.strptime(newest, "%Y-%m-%d %H%M")
    days = (now.date() - when.date()).days
    if days <= 0:
        return f"Your last backup was today at {_clock(when)}."
    if days == 1:
        return f"Your last backup was yesterday at {_clock(when)}."
    return f"Your last backup was {days} days ago."
