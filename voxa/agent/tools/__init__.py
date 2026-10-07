from __future__ import annotations

from voxa.agent.registry import ToolRegistry
from voxa.agent.tools.applications import application_tools
from voxa.agent.tools.browser import browser_tools
from voxa.agent.tools.compiz import compiz_tools
from voxa.agent.tools.deals import deal_tools
from voxa.agent.tools.files import file_dialog_tools
from voxa.agent.tools.home import home_tools
from voxa.agent.tools.media import media_tools
from voxa.agent.tools.reminders import reminders_tools
from voxa.agent.tools.typing import typing_tools
from voxa.agent.tools.volume import volume_tools
from voxa.agent.tools.web import web_tools
from voxa.agent.tools.windows import window_tools


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [
        *browser_tools(),
        *deal_tools(),
        *application_tools(),
        *media_tools(),
        *typing_tools(),
        *window_tools(),
        *web_tools(),
        *reminders_tools(),
        *home_tools(),
        *volume_tools(),
        *compiz_tools(),
        *file_dialog_tools(),
    ]:
        registry.register(tool)
    return registry
