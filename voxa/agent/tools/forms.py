"""Fill one web form's name, address and email fields from what Voxa already knows (issue #63).

The page half already exists in ``browser_session`` (``outline`` and ``type_into``); this is the
profile half: read the form's fields, fill only the ones Voxa has a value for, ask for the rest,
and never send the form without a spoken yes.
"""

from __future__ import annotations

import logging
import re
import subprocess

from voxa.agent.cdp import CdpError
from voxa.agent.mailflow import contact_key
from voxa.agent.memory_facts import facts
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import contacts as contact_store
from voxa.agent.tools import memory_facts as fact_store
from voxa.config import ConfigStore

log = logging.getLogger("voxa.agent.tools.forms")

# The only word that sends a form. Anything else is refused.
YES = "yes"

# Labels that ask for something other than the person's own name.
_NOT_A_NAME = (
    "file name",
    "filename",
    "user name",
    "username",
    "domain",
    "site name",
    "company name",
    "business name",
    "product name",
    "project name",
    "field name",
    "name and role",
)

_NAME_PATTERNS = (
    re.compile(r"^my name is (.+)$", re.I),
    re.compile(r"^my name[:\s]+(.+)$", re.I),
    re.compile(r"^call me (.+)$", re.I),
)
_ADDRESS_PATTERNS = (
    re.compile(r"^my (?:home )?address is (.+)$", re.I),
    re.compile(r"^my address[:\s]+(.+)$", re.I),
    re.compile(r"^i live at (.+)$", re.I),
    re.compile(r"^i live in (.+)$", re.I),
)
_EMAIL_PATTERNS = (
    re.compile(r"^my email(?: address)? is (.+)$", re.I),
    re.compile(r"^my email(?: address)?[:\s]+(.+)$", re.I),
)

# Button labels a form usually sends with.
_SUBMIT_LABELS = ("submit", "send", "save", "go", "ok", "yes")


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def form_fields(outline: list[dict]) -> list[str]:
    """The labels of the fillable fields in a page outline, in page order."""
    labels: list[str] = []
    for item in outline:
        if not isinstance(item, dict):
            continue
        if item.get("kind") not in ("field", "select"):
            continue
        label = str(item.get("text") or item.get("placeholder") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def field_key(label: str) -> str:
    """Which profile value a field label asks for: name, address, email, or ""."""
    text = " ".join(label.lower().split())
    if "email" in text or "e-mail" in text:
        return "email"
    if "address" in text or "where you live" in text:
        return "address"
    if "name" in text and not any(bad in text for bad in _NOT_A_NAME):
        return "name"
    return ""


def _read_memory() -> str:
    return fact_store._read()


def _fact_lines() -> list[str]:
    try:
        return facts(_read_memory())
    except (OSError, subprocess.SubprocessError):
        return []


def _first(patterns: tuple[re.Pattern[str], ...], lines: list[str]) -> str:
    for pattern in patterns:
        for line in lines:
            match = pattern.match(line)
            if match:
                return " ".join(match.group(1).split()).strip(".,!?")
    return ""


def _email_from_contacts(name: str) -> str:
    getter = contact_store.get_contacts
    if getter is None:
        return ""
    stored = getter() or {}
    keys = [contact_key(name)] if name else []
    keys += ["me", "myself", "self"]
    for key in keys:
        address = stored.get(key, "")
        if address:
            return address
    return ""


def _home_location() -> str:
    try:
        return ConfigStore().load().home_location.strip()
    except OSError:
        return ""


def profile_values() -> dict[str, str]:
    """What Voxa knows about the person: name, address, email; "" when unknown."""
    lines = _fact_lines()
    name = _first(_NAME_PATTERNS, lines)
    email = _first(_EMAIL_PATTERNS, lines) or _email_from_contacts(name)
    address = _first(_ADDRESS_PATTERNS, lines) or _home_location()
    return {"name": name, "address": address, "email": email}


def fill_form(session, values: dict[str, str], fields: list[str]) -> dict[str, list[str]]:
    """Type each known value into the field that asks for it; touch nothing else.

    Returns {"filled": [...], "unknown": [...], "failed": [...]}: the fields Voxa filled,
    the fields she has no value for (never guessed), and the fields the page would not take.
    Nothing is submitted here.
    """
    filled: list[str] = []
    unknown: list[str] = []
    failed: list[str] = []
    for field in fields:
        key = field_key(field)
        value = values.get(key, "") if key else ""
        if not value:
            unknown.append(field)
            continue
        if session.type_into(field, value):
            filled.append(field)
        else:
            failed.append(field)
    return {"filled": filled, "unknown": unknown, "failed": failed}


def _submit_button(outline: list[dict]) -> str:
    """The first button whose label usually sends a form."""
    for item in outline:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "button":
            continue
        label = str(item.get("text") or "").strip()
        if label.lower() in _SUBMIT_LABELS:
            return label
    return ""


def _form_read_handler(args: dict[str, str]) -> ToolResult:
    from voxa.agent.tools.web import get_session

    session = get_session()
    try:
        fields = form_fields(session.outline())
    except CdpError as exc:
        return ToolResult.failure("I could not read the page.", detail=str(exc))
    if not fields:
        return ToolResult.failure("There is no form on this page.")
    return ToolResult.success(
        f"This form has {len(fields)} fields: {_join(fields)}.",
        detail="\n".join(fields),
    )


def _form_fill_handler(args: dict[str, str]) -> ToolResult:
    from voxa.agent.tools.web import get_session

    session = get_session()
    try:
        fields = form_fields(session.outline())
        outcome = fill_form(session, profile_values(), fields)
    except CdpError as exc:
        return ToolResult.failure("I could not read the page.", detail=str(exc))
    filled, unknown, failed = outcome["filled"], outcome["unknown"], outcome["failed"]
    detail = "\n".join(
        [
            f"filled: {', '.join(filled)}",
            f"missing: {', '.join(unknown)}",
            f"refused by the page: {', '.join(failed)}",
        ]
    )
    if not fields:
        return ToolResult.failure("There is no form on this page.")
    if not filled:
        return ToolResult.failure(f"I don't have what this form asks for. What should go in {_join(unknown)}?")
    speech = f"Filled in {_join(filled)}."
    if unknown:
        speech += f" What should go in {_join(unknown)}?"
    if failed:
        speech += f" The page would not take {_join(failed)}."
    return ToolResult.success(f"{speech} I have not sent it.", detail=detail)


def _form_submit_handler(args: dict[str, str]) -> ToolResult:
    if args.get("confirm", "").strip().casefold() != YES:
        return ToolResult.failure(
            "I only send a form when you say yes. Say yes and I'll press the button."
        )
    from voxa.agent.tools.web import get_session

    session = get_session()
    button = args.get("button", "").strip()
    try:
        if not button:
            button = _submit_button(session.outline())
        if not button:
            return ToolResult.failure("I can't find a button to press on this page.")
        pressed = session.click(button)
        if not pressed:
            return ToolResult.failure(f"I couldn't find {button} on the page.")
        detail = f"{session.title()} - {session.url()}\nPressed: {pressed}"
    except CdpError as exc:
        return ToolResult.failure("I couldn't send the form.", detail=str(exc))
    return ToolResult.success(f"Pressed {pressed}.", detail=detail)


def form_tools() -> list[Tool]:
    return [
        Tool(
            name="form_read",
            description="List the fields of the form on the open page.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_form_read_handler,
        ),
        Tool(
            name="form_fill",
            description="Fill the form's name, address and email fields from what Voxa remembers, "
            "and ask for the fields she has no value for.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_form_fill_handler,
        ),
        Tool(
            name="form_submit",
            description="Press the form's send button. Only works when confirm is exactly yes.",
            parameters={"confirm": "yes, and only yes, to send the form", "button": "label of the button to press"},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=_form_submit_handler,
        ),
    ]
