from __future__ import annotations

from voxa.agent.registry import ToolRegistry
from voxa.agent.tools.applications import application_tools
from voxa.agent.tools.browser import browser_tools
from voxa.agent.tools.typing import typing_tools
from voxa.agent.tools.windows import window_tools


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [*browser_tools(), *application_tools(), *typing_tools(), *window_tools()]:
        registry.register(tool)
    return registry
