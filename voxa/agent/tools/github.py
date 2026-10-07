from __future__ import annotations

import re
from collections.abc import Callable

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

set_owner: Callable[[str], None] | None = None


def clean_owner(spoken: str) -> str:
    text = spoken.strip()
    if text.startswith("@"):
        text = text[1:]
    text = re.sub(r"\b(?:dash|hyphen)\b", "-", text, flags=re.IGNORECASE)
    parts = re.split(r"[ -]+", text)
    if parts and all(len(part) == 1 and part.isalpha() for part in parts):
        text = "".join(parts)
    else:
        text = text.replace(" ", "")
    text = "".join(c for c in text.lower() if c.isalnum() or c == "-")
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,38})", text):
        return text
    return ""


def _set_github_owner_handler(args: dict[str, str]) -> ToolResult:
    owner = clean_owner(args["name"])
    if not owner:
        return ToolResult.failure("I did not catch the GitHub name.")
    if set_owner is None:
        return ToolResult.failure("I cannot save that here.")
    set_owner(owner)
    return ToolResult.success(f"Got it. Your GitHub name is {owner}.")


def github_tools() -> list[Tool]:
    return [
        Tool(
            name="set_github_owner",
            description="Remember the user's GitHub account name so project lookups search their own repositories.",
            parameters={"name": "the GitHub account name"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_github_owner_handler,
            required=("name",),
        ),
    ]
