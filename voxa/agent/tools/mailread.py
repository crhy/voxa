"""Reading the inbox by voice (#56, part 1): unread count, senders, subjects, one message aloud.

Both tools are Level-0 READ_ONLY. They never send, delete, move or unsubscribe, and
the only connection they can make is the user's own IMAP account over TLS. With
``VOXA_SIMULATE_ACTIONS`` on, or with no mail account configured, nothing connects
at all: the simulated answer is spoken from the settings alone.
"""

from __future__ import annotations

import logging

from voxa.agent import mailread
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.simulation import actions_simulated

log = logging.getLogger("voxa.agent.tools.mailread")

SIMULATED_SUMMARY = "Mail reading is simulated: your unread mail is not being opened."
SIMULATED_READ = "Reading one email is simulated: nothing was fetched from your mailbox."


def _settings():
    from voxa.config import ConfigStore

    return ConfigStore().load()


def _configured(settings) -> bool:
    return bool(settings.mail_host and settings.mail_user and settings.mail_password)


def _mailbox(settings):
    """A connected mailbox, or None when mail is not set up or unreachable."""
    return mailread.connect(settings)


def mail_summary(args: dict[str, str]) -> ToolResult:
    settings = _settings()
    if not _configured(settings):
        return ToolResult.failure(mailread.NOT_CONFIGURED)
    if actions_simulated() and mailread.client_factory is None:
        return ToolResult.success(SIMULATED_SUMMARY)
    client = _mailbox(settings)
    if client is None:
        return ToolResult.failure(mailread.UNREADABLE)
    try:
        inbox = mailread.unread(client)
        if inbox is None:
            return ToolResult.failure(mailread.UNREADABLE)
        if inbox.count == 0:
            return ToolResult.success(mailread.spoken_none(), detail="0 unread")
        ranked = mailread.important(inbox.messages, settings.contacts)
        if inbox.count > mailread.SPOKEN_IMPORTANT and ranked:
            speech = (
                f"{mailread.spoken_summary(inbox.count, inbox.senders, inbox.subjects)} "
                f"{mailread.spoken_too_many(inbox.count, len(ranked))} "
                f"{mailread.spoken_important(ranked)}"
            )
        elif ranked:
            speech = (
                f"{mailread.spoken_summary(inbox.count, inbox.senders, inbox.subjects)} "
                f"{mailread.spoken_important(ranked)}"
            )
        else:
            speech = mailread.spoken_summary(inbox.count, inbox.senders, inbox.subjects)
        return ToolResult.success(speech, detail=f"{inbox.count} unread in the last {inbox.days:g} days")
    finally:
        _close(client)


def mail_read(args: dict[str, str]) -> ToolResult:
    settings = _settings()
    if not _configured(settings):
        return ToolResult.failure(mailread.NOT_CONFIGURED)
    if actions_simulated() and mailread.client_factory is None:
        return ToolResult.success(SIMULATED_READ)
    client = _mailbox(settings)
    if client is None:
        return ToolResult.failure(mailread.UNREADABLE)
    try:
        body = mailread.read_one(client, args["who"])
        if not body:
            return ToolResult.failure(f"I couldn't find an unread email from or about {args['who']}.")
        first = " ".join(body.split())
        return ToolResult.success(first[:600], detail=first)
    finally:
        _close(client)


def _close(client) -> None:
    """Hang up the mailbox; a fake test mailbox may not implement logout."""
    mailread.close(client)


def mail_tools() -> list[Tool]:
    return [
        Tool(
            name="mail_summary",
            description="Count the unread emails in the inbox and say who they are from and what they are about.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=mail_summary,
        ),
        Tool(
            name="mail_read",
            description="Read one email aloud, chosen by the person it is from or its subject.",
            parameters={"who": "the sender's name or a word from the subject"},
            risk=RiskLevel.READ_ONLY,
            handler=mail_read,
            required=("who",),
        ),
    ]
