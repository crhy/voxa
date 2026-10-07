from __future__ import annotations

import re
from collections.abc import Callable

from voxa.agent.mailflow import contact_key, spoken_address
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

get_contacts: Callable[[], dict[str, str]] | None = None
save_contacts: Callable[[dict[str, str]], None] | None = None


def contact_key_name(name: str) -> str:
    stripped = re.sub(r"'s$|s'$", "", name.strip())
    return contact_key(stripped)


def _set_contact_email_handler(args: dict[str, str]) -> ToolResult:
    key = contact_key_name(args["name"])
    addr = spoken_address(args["address"])
    if not key or not addr:
        return ToolResult.failure("I did not catch that email address.")
    if get_contacts is None or save_contacts is None:
        return ToolResult.failure("I cannot save that here.")
    contacts = dict(get_contacts())
    contacts[key] = addr
    save_contacts(contacts)
    return ToolResult.success(f"Saved. {key.title()}'s email is {addr}.")


def _get_contact_email_handler(args: dict[str, str]) -> ToolResult:
    key = contact_key_name(args["name"])
    if get_contacts is None:
        return ToolResult.failure(f"I do not have an email address for {key}.")
    addr = get_contacts().get(key, "")
    if not addr:
        return ToolResult.failure(f"I do not have an email address for {key}.")
    return ToolResult.success(f"{key.title()}'s email is {addr}.")


def _forget_contact_email_handler(args: dict[str, str]) -> ToolResult:
    key = contact_key_name(args["name"])
    if get_contacts is None or save_contacts is None:
        return ToolResult.failure(f"I do not have an email address for {key}.")
    contacts = dict(get_contacts())
    if key not in contacts:
        return ToolResult.failure(f"I do not have an email address for {key}.")
    del contacts[key]
    save_contacts(contacts)
    return ToolResult.success(f"Forgotten {key}'s email.")


def contact_tools() -> list[Tool]:
    return [
        Tool(
            name="set_contact_email",
            description="Remember a contact's email address from an address spoken out loud.",
            parameters={"name": "the contact's name", "address": "the email address said out loud"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_contact_email_handler,
            required=("name", "address"),
        ),
        Tool(
            name="get_contact_email",
            description="Read back a contact's saved email address.",
            parameters={"name": "the contact's name"},
            risk=RiskLevel.REVERSIBLE,
            handler=_get_contact_email_handler,
            required=("name",),
        ),
        Tool(
            name="forget_contact_email",
            description="Forget a contact's saved email address.",
            parameters={"name": "the contact's name"},
            risk=RiskLevel.REVERSIBLE,
            handler=_forget_contact_email_handler,
            required=("name",),
        ),
    ]
