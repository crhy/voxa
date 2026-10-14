from __future__ import annotations

from datetime import date, datetime

from voxa.agent.headsup import forgotten, meeting_brief, suggest_next, today_summary
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import calendar_local as calendar_tools
from voxa.agent.tools import dates as dates_tools
from voxa.agent.tools import lists as lists_tools
from voxa.agent.tools import memory_facts as memory_tools

REACH_ERROR = "I could not reach your notes."


def _safe_read(read) -> str:
    try:
        return read()
    except Exception:
        return ""


def _heads_up_handler(args: dict[str, str]) -> ToolResult:
    now = datetime.now()
    calendar = _safe_read(calendar_tools._read)
    dates_text = _safe_read(dates_tools._read)
    lists_text = _safe_read(lists_tools._read)
    return ToolResult.success(today_summary(calendar, dates_text, lists_text, now))


def _what_did_i_forget_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    dates_text = _safe_read(dates_tools._read)
    lists_text = _safe_read(lists_tools._read)
    calendar = _safe_read(calendar_tools._read)
    return ToolResult.success(forgotten(dates_text, lists_text, calendar, today))


def _suggest_next_action_handler(args: dict[str, str]) -> ToolResult:
    now = datetime.now()
    dates_text = _safe_read(dates_tools._read)
    lists_text = _safe_read(lists_tools._read)
    calendar = _safe_read(calendar_tools._read)
    return ToolResult.success(suggest_next(dates_text, lists_text, calendar, now))


def _meeting_brief_handler(args: dict[str, str]) -> ToolResult:
    now = datetime.now()
    title = args["title"].strip()
    if not title:
        return ToolResult.failure("Which appointment do you mean?")
    calendar = _safe_read(calendar_tools._read)
    lists_text = _safe_read(lists_tools._read)
    facts_text = _safe_read(memory_tools._read)
    return ToolResult.success(meeting_brief(calendar, lists_text, facts_text, title, now))


def headsup_tools() -> list[Tool]:
    return [
        Tool(
            name="heads_up",
            description="Give a heads up on today: appointments, things due soon, and open to-dos.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_heads_up_handler,
        ),
        Tool(
            name="what_did_i_forget",
            description="Say what you have forgotten: overdue deadlines and open to-do items.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_what_did_i_forget_handler,
        ),
        Tool(
            name="suggest_next_action",
            description="Recommend the next action: the soonest appointment, an urgent item, or the first open to-do.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_suggest_next_action_handler,
        ),
        Tool(
            name="meeting_brief",
            description="Brief you on an appointment: when it is, related facts, and related open to-dos.",
            parameters={"title": "which appointment, e.g. dentist"},
            risk=RiskLevel.READ_ONLY,
            handler=_meeting_brief_handler,
            required=("title",),
        ),
    ]
