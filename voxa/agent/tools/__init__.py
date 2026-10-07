from __future__ import annotations

from voxa.agent.registry import ToolRegistry
from voxa.agent.tools.applications import application_tools
from voxa.agent.tools.appstore import appstore_tools
from voxa.agent.tools.browser import browser_tools
from voxa.agent.tools.calc import calc_tools
from voxa.agent.tools.compiz import compiz_tools
from voxa.agent.tools.contacts import contact_tools
from voxa.agent.tools.deals import deal_tools
from voxa.agent.tools.filemanage import file_manage_tools
from voxa.agent.tools.files import file_dialog_tools
from voxa.agent.tools.github import github_tools
from voxa.agent.tools.help import help_tools
from voxa.agent.tools.home import home_tools
from voxa.agent.tools.media import media_tools
from voxa.agent.tools.notes import notes_tools
from voxa.agent.tools.reminders import reminders_tools
from voxa.agent.tools.screenshot import screenshot_tools
from voxa.agent.tools.system import system_tools
from voxa.agent.tools.sysupdate import sysupdate_tools
from voxa.agent.tools.textedit import textedit_tools
from voxa.agent.tools.typing import typing_tools
from voxa.agent.tools.uicontrol import uicontrol_tools
from voxa.agent.tools.volume import volume_tools
from voxa.agent.tools.web import web_tools
from voxa.agent.tools.windows import window_tools


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [
        *browser_tools(),
        *calc_tools(),
        *deal_tools(),
        *application_tools(),
        *contact_tools(),
        *media_tools(),
        *github_tools(),
        *help_tools(),
        *typing_tools(),
        *window_tools(),
        *web_tools(),
        *reminders_tools(),
        *notes_tools(),
        *screenshot_tools(),
        *home_tools(),
        *volume_tools(),
        *compiz_tools(),
        *file_dialog_tools(),
        *textedit_tools(),
        *file_manage_tools(),
        *system_tools(),
        *sysupdate_tools(),
        *appstore_tools(),
        *uicontrol_tools(),
    ]:
        registry.register(tool)
    return registry
