from __future__ import annotations

from voxa.agent.calc import answer
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

_FAILURE = "I could not work that out."


def calculate(args: dict[str, str]) -> ToolResult:
    result = answer(args.get("expression", ""))
    if result is None:
        return ToolResult.failure(_FAILURE)
    return ToolResult.success(result)


def calc_tools() -> list[Tool]:
    return [
        Tool(
            name="calculate",
            description="Answer an arithmetic sum or unit conversion exactly, without the AI model.",
            parameters={"expression": "the sum or conversion to work out"},
            risk=RiskLevel.READ_ONLY,
            handler=calculate,
        ),
    ]
