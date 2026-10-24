from __future__ import annotations

from voxa.agent.registry import ToolRegistry
from voxa.agent.tools.anc import anc_tools
from voxa.agent.tools.applications import application_tools
from voxa.agent.tools.appstore import appstore_tools
from voxa.agent.tools.briefing import briefing_tools
from voxa.agent.tools.browser import browser_tools
from voxa.agent.tools.calc import calc_tools
from voxa.agent.tools.calendar_local import calendar_local_tools
from voxa.agent.tools.camera import camera_tools
from voxa.agent.tools.compiz import compiz_tools
from voxa.agent.tools.contacts import contact_tools
from voxa.agent.tools.convert import convert_tools
from voxa.agent.tools.convosummary import convosummary_tools
from voxa.agent.tools.databackup import databackup_tools
from voxa.agent.tools.dates import date_tools
from voxa.agent.tools.deals import deal_tools
from voxa.agent.tools.desktoptheme import desktoptheme_tools
from voxa.agent.tools.devstatus import devstatus_tools
from voxa.agent.tools.events import event_tools
from voxa.agent.tools.expenses import expense_tools
from voxa.agent.tools.filemanage import file_manage_tools
from voxa.agent.tools.files import file_dialog_tools
from voxa.agent.tools.github import github_tools
from voxa.agent.tools.headsup import headsup_tools
from voxa.agent.tools.help import help_tools
from voxa.agent.tools.hidden_files import hidden_files_tools
from voxa.agent.tools.home import home_tools
from voxa.agent.tools.lists import list_tools
from voxa.agent.tools.livetv import livetv_tools
from voxa.agent.tools.local import local_tools
from voxa.agent.tools.localmusic import local_music_tools
from voxa.agent.tools.lookups import lookups_tools
from voxa.agent.tools.meals import meals_tools
from voxa.agent.tools.media import media_tools
from voxa.agent.tools.memory_facts import memory_facts_tools
from voxa.agent.tools.notes import notes_tools
from voxa.agent.tools.projectsteps import project_steps_tools
from voxa.agent.tools.radio import radio_tools
from voxa.agent.tools.readfile import readfile_tools
from voxa.agent.tools.recite import recite_tools
from voxa.agent.tools.reminders import reminders_tools
from voxa.agent.tools.screenshot import screenshot_tools
from voxa.agent.tools.system import system_tools
from voxa.agent.tools.sysupdate import sysupdate_tools
from voxa.agent.tools.textedit import textedit_tools
from voxa.agent.tools.transcribe_file import transcribe_file_tools
from voxa.agent.tools.translate import translate_tools
from voxa.agent.tools.traveldocs import travel_doc_tools
from voxa.agent.tools.trips import trip_tools
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
        *desktoptheme_tools(),
        *local_music_tools(),
        *contact_tools(),
        *media_tools(),
        *github_tools(),
        *help_tools(),
        *typing_tools(),
        *window_tools(),
        *web_tools(),
        *reminders_tools(),
        *notes_tools(),
        *list_tools(),
        *event_tools(),
        *convosummary_tools(),
        *trip_tools(),
        *travel_doc_tools(),
        *project_steps_tools(),
        *databackup_tools(),
        *meals_tools(),
        *devstatus_tools(),
        *headsup_tools(),
        *lookups_tools(),
        *transcribe_file_tools(),
        *convert_tools(),
        *memory_facts_tools(),
        *calendar_local_tools(),
        *expense_tools(),
        *date_tools(),
        *screenshot_tools(),
        *camera_tools(),
        *radio_tools(),
        *livetv_tools(),
        *local_tools(),
        *briefing_tools(),
        *translate_tools(),
        *home_tools(),
        *volume_tools(),
        *compiz_tools(),
        *file_dialog_tools(),
        *textedit_tools(),
        *readfile_tools(),
        *recite_tools(),
        *file_manage_tools(),
        *hidden_files_tools(),
        *system_tools(),
        *sysupdate_tools(),
        *appstore_tools(),
        *uicontrol_tools(),
        *anc_tools(),
    ]:
        registry.register(tool)
    return registry
