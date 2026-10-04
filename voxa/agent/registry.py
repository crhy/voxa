from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voxa.agent.policy import RiskLevel
from voxa.agent.result import ToolResult

Handler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    risk: RiskLevel
    handler: Handler
    required: tuple[str, ...] = ()
    timeout_seconds: float = 15.0


class ToolError(Exception):
    """Raised when a tool cannot be registered, found, or validated."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        lines = []
        for name in self.names():
            tool = self._tools[name]
            args = ", ".join(tool.parameters)
            lines.append(f"{name}({args}): {tool.description} [risk {int(tool.risk)}]")
        return "\n".join(lines)

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, str]:
        tool = self.get(name)
        cleaned: dict[str, str] = {}
        for key, value in args.items():
            if key not in tool.parameters:
                raise ToolError(f"unknown argument for tool {name}: {key}")
            cleaned[key] = str(value).strip()
        for key in tool.required:
            if not cleaned.get(key):
                raise ToolError(f"missing required argument for tool {name}: {key}")
        return cleaned

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        cleaned = self.validate(name, args)
        try:
            return self.get(name).handler(cleaned)
        except Exception as exc:
            return ToolResult.failure("That didn't work.", detail=str(exc))
