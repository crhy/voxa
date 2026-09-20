"""Draft emails with the local model and hand them to the user's mail client.

Voxa never sends mail: it has no account and no credentials, and sending is a
Level-2 external action (see .voxa-spec/issue7-master-roadmap.md section 5). The
model writes the text, and `compose` opens the user's default mail client with
the draft already filled in, unsent, so the user reads it and sends it themselves.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

IN_FLATPAK = os.path.exists("/.flatpak-info")

_EMAIL = re.compile(
    r"^\s*(?:please\s+)?(?:draft|write|compose|send|create|make|email|mail)?\s*"
    r"(?:me\s+)?(?:an|the|a)?\s*email"
    r"(?:\s+to\s+(?P<to>[^,.!?]+?))?"
    r"(?:\s+(?:about|regarding|saying|telling\s+(?:them|him|her))\s+(?P<topic>.+?))?"
    r"\s*[.!?]*\s*$",
    re.I,
)

_EXTRA_COMMAND = re.compile(
    r"\b(?:then|after|and|also)\s+(?:open|launch|close|send|create|write|make|draft|compose|email|document|firefox|browser|app|program|file)\b",
    re.I,
)

DRAFT_SYSTEM_PROMPT = """You draft emails for the user, who reviews and sends them personally — you never send email
yourself, only draft it. Given a short request, write one clear, complete, ready-to-send email.
Reply with exactly this format and nothing else, no commentary before or after:
Subject: <subject line>
Body:
<the full email body, in complete sentences, signed off appropriately>"""


@dataclass(frozen=True, slots=True)
class EmailRequest:
    to: str
    topic: str


def parse_email_command(prompt: str) -> EmailRequest | None:
    """The recipient and topic from "draft an email to X about Y", or None."""
    if _EXTRA_COMMAND.search(prompt):
        return None
    match = _EMAIL.match(prompt)
    if match is None:
        return None
    return EmailRequest((match.group("to") or "").strip(), (match.group("topic") or "").strip())


def build_prompt(request: EmailRequest) -> str:
    if not request.to and not request.topic:
        return "Draft a short, generic email."
    parts = []
    if request.to:
        parts.append(f"to {request.to}")
    if request.topic:
        parts.append(f"about {request.topic}")
    return f"Draft an email {' '.join(parts)}."


def parse_draft(text: str) -> tuple[str, str]:
    """Split the model's reply into (subject, body), falling back to the raw text."""
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        if line.strip().casefold().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body = "\n".join(lines[index + 1 :]).strip()
            if body.casefold().startswith("body:"):
                body = body[5:].strip()
            return subject, body
    return "", text.strip()


def _host(command: list[str]) -> list[str]:
    return ["flatpak-spawn", "--host", *command] if IN_FLATPAK else command


def compose(to: str, subject: str, body: str) -> None:
    """Open the user's default mail client with an unsent draft prefilled."""
    command = ["xdg-email"]
    if subject:
        command += ["--subject", subject]
    if body:
        command += ["--body", body]
    if to:
        command.append(to)
    subprocess.Popen(  # noqa: S603 - fixed argv, the recipient comes from the user's own words
        _host(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
