from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from voxa import documents
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.slides import parse_slides, write_odp

log = logging.getLogger(__name__)

ask_model: Callable[[list[dict]], str] | None = None

PROMPT = (
    "You make slide decks for the user to open and edit in LibreOffice Impress. "
    "Given a topic, write exactly {count} slides: for each slide put its title on "
    "its own line starting with a number and a period, then write two or three bullet "
    "points under it, each starting with a dash. No other text."
)

_NUMBER_WORDS = {
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


def _odp_path(topic: str) -> Path:
    """A never-colliding .odp path inside the user's Voxa Drafts folder."""
    base = Path(os.environ.get("XDG_DOCUMENTS_DIR") or (Path.home() / "Documents"))
    folder = base / "Voxa Drafts"
    folder.mkdir(parents=True, exist_ok=True)
    stem = documents._slug(topic)
    path = folder / f"{stem} slides.odp"
    number = 2
    while path.exists():
        path = folder / f"{stem} slides {number}.odp"
        number += 1
    return path


def _slide_count(spoken: str) -> int:
    try:
        count = int(spoken)
    except (TypeError, ValueError):
        count = _NUMBER_WORDS.get(spoken.strip().casefold(), 5)
    return max(3, min(8, count))


def _presentation_make_handler(args: dict[str, str]) -> ToolResult:
    topic = args["topic"].strip()
    if not topic:
        return ToolResult.failure("What should the presentation be about?")
    if ask_model is None:
        return ToolResult.failure("I can't reach my brain right now.")
    count = _slide_count(args.get("count", ""))
    try:
        reply = ask_model(
            [
                {"role": "system", "content": PROMPT.format(count=count)},
                {"role": "user", "content": f"Make a presentation about {topic}."},
            ]
        )
    except Exception:  # noqa: BLE001 - the model server is outside our control
        reply = ""
    slides = parse_slides(reply)[:count]
    if not slides:
        return ToolResult.failure(f"I couldn't make any slides about {topic}.")
    path = _odp_path(topic)
    write_odp(path, topic, slides)
    documents.open_in_libreoffice(path)
    return ToolResult.success(f"I made {len(slides)} slides about {topic} and opened them in LibreOffice.")


def slides_tools() -> list[Tool]:
    return [
        Tool(
            name="presentation_make",
            description="Make a slide deck about a topic and open it in LibreOffice Impress.",
            parameters={"topic": "what the presentation is about", "count": "how many slides, 3 to 8"},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=_presentation_make_handler,
            required=("topic",),
        ),
    ]
