from __future__ import annotations

from voxa.agent.helptext import overview, topic_for, topic_help
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult


def voxa_help(args: dict[str, str]) -> ToolResult:
    key = topic_for(args.get("topic", ""))
    if key is None:
        return ToolResult.success(overview())
    return ToolResult.success(topic_help(key))


def help_tools() -> list[Tool]:
    return [
        Tool(
            name="voxa_help",
            description="Answer what Voxa can do from a fixed list; with a topic, answer about that topic.",
            parameters={"topic": "what to explain, for example files, music, web or writing"},
            risk=RiskLevel.READ_ONLY,
            handler=voxa_help,
            required=(),
        ),
    ]
