"""Read the inbox over IMAP: how much is unread, who it is from, what it is about,
and one message read aloud. Part 1 of issue #56.

Strictly read-only. Nothing here sends, deletes, moves or unsubscribes: INBOX is
selected with ``readonly=True`` and every read uses ``BODY.PEEK``, which leaves the
server's flags, folders and messages exactly as they were. The only network
connection this module can make is the user's own IMAP account over TLS
(``imaplib.IMAP4_SSL`` on ``Settings.mail_host``); it imports no HTTP client, no
socket helper and no browser, and it never connects anywhere but that host.

Replying by dictation, unsubscribing, archiving in bulk, receipts and tracking
numbers, provider APIs and the Gmail-through-browser fallback are the next slices
of #56 and are not built here.
"""

from __future__ import annotations

import imaplib
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email import message_from_bytes, policy
from email.utils import parseaddr, parsedate_to_datetime

from voxa.agent.mailflow import _MAIL_DOMAINS, contact_key

log = logging.getLogger(__name__)

__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "MAX_MESSAGES",
    "SPOKEN_SENDERS",
    "Inbox",
    "Message",
    "NOT_CONFIGURED",
    "UNREADABLE",
    "client_factory",
    "connect",
    "unread",
    "close",
    "importance",
    "important",
    "score",
    "read_one",
    "spoken_summary",
    "spoken_important",
    "spoken_none",
    "spoken_too_many",
]

# Set by tests to a fake mailbox; None means use imaplib.IMAP4_SSL with the settings.
client_factory: Callable[[], object] | None = None

CONNECT_TIMEOUT_SECONDS = 8.0
# A 400-message inbox must not become a monologue: only the newest messages are
# fetched, and only a handful are ever spoken.
MAX_MESSAGES = 200
SPOKEN_SENDERS = 3
SPOKEN_IMPORTANT = 3

NOT_CONFIGURED = (
    "Mail isn't set up yet. Add your IMAP server, your account and its password in "
    "Preferences, and I can read your inbox from there."
)
UNREADABLE = "I couldn't read your inbox. Your mail server may be busy or the password wrong."

# Headers only: BODY.PEEK reads a body without marking it seen.
_HEADER_FETCH = "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
_BODY_FETCH = "(BODY.PEEK[])"

_IMPORTANT_WORDS = re.compile(
    r"\b(?:deliver(?:y|ies)?|invoice|receipt|tracking|shipment|package|parcel|ticket|"
    r"due|appointment|reservation|confirm(?:ed)?|payment|renewal)\b",
    re.IGNORECASE,
)
# A subject that already carries Re:/Fwd: means Voxa is replying into a thread
# the user started: someone is answering the user, so it is worth hearing.
_REPLY_SUBJECT = re.compile(r"^\s*(?:re|fwd|fw|aw)\s*:", re.IGNORECASE)
_TAGGED_SUBJECT = re.compile(r"^\s*(?:\[[^\]]{1,24}\]\s*)+")
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class Message:
    """One header read from INBOX; ``number`` is its IMAP sequence number."""

    number: int = 0
    sender: str = ""
    address: str = ""
    subject: str = ""
    when: datetime | None = None


@dataclass(frozen=True, slots=True)
class Inbox:
    """The unread mail in INBOX. ``days`` is how long the backlog has been sitting."""

    count: int = 0
    senders: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    days: float = 0.0
    messages: tuple[Message, ...] = ()


def _clean(value: object) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def _decode_when(raw: object) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(raw or ""))
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def close(client: object) -> None:
    """Hang up the mailbox. A fake test mailbox may not implement ``logout``."""
    if client is None:
        return
    try:
        client.logout()  # type: ignore[attr-defined]
    except (AttributeError, OSError, TimeoutError, imaplib.IMAP4.error):
        log.info("mail: logout did not complete cleanly")


def _search(client: object, criterion: str) -> list[int]:
    status, data = client.search(criterion)  # type: ignore[attr-defined]
    if status != "OK" or not data:
        return []
    return [int(number) for number in (data[0] or b"").split()]


def _fetch(client: object, number: int, query: str) -> bytes:
    status, data = client.fetch(number, query)  # type: ignore[attr-defined]
    if status != "OK":
        return b""
    for item in data or ():
        # imaplib hands back (b'<n> (BODY... {<len>}', b'<the literal>') pairs.
        if isinstance(item, tuple) and len(item) == 2 and item[1]:
            return item[1]
    return b""


def _header(client: object, number: int) -> Message:
    raw = _fetch(client, number, _HEADER_FETCH)
    if not raw:
        return Message(number=number)
    record = message_from_bytes(raw, policy=policy.default)
    name, address = parseaddr(_clean(record.get("from")))
    return Message(
        number=number,
        sender=name or address,
        address=address,
        subject=_clean(record.get("subject")),
        when=_decode_when(record.get("date")),
    )


def _plain_text(record) -> str:
    try:
        part = record.get_body(preferencelist=("plain",))
    except (AttributeError, ValueError):
        part = None
    if part is None:
        try:
            part = record.get_body(preferencelist=("html",))
        except (AttributeError, ValueError):
            part = None
        if part is None:
            payload = record.get_payload(decode=True)
            text = (payload or b"").decode("utf-8", "replace") if isinstance(payload, bytes) else str(payload or "")
        else:
            text = _HTML_TAG.sub(" ", str(part.get_content()))
    else:
        text = str(part.get_content())
    text = _BLANK_LINES.sub("\n\n", text).strip()
    return text


def connect(settings) -> object | None:
    """A logged-in IMAP client, or None.

    Returns None instead of raising when mail is not set up, when the server
    cannot be reached, or when the account refuses the login: a missing inbox is
    a thing to say, not a crash. The host always comes from ``settings.mail_host``.
    """
    if client_factory is not None:
        return client_factory()
    host = _clean(getattr(settings, "mail_host", ""))
    user = _clean(getattr(settings, "mail_user", ""))
    password = getattr(settings, "mail_password", "") or ""
    if not host or not user or not password:
        return None
    try:
        client = imaplib.IMAP4_SSL(host, timeout=CONNECT_TIMEOUT_SECONDS)
    except (OSError, TimeoutError, imaplib.IMAP4.error) as exc:
        log.info("mail: could not reach %s: %s", host, exc)
        return None
    try:
        client.login(user, password)
    except (OSError, TimeoutError, imaplib.IMAP4.error) as exc:
        log.info("mail: login for %s failed: %s", user, exc)
        try:
            client.logout()
        except (OSError, imaplib.IMAP4.error):
            pass
        return None
    return client


def unread(client: object) -> Inbox | None:
    """The unread mail in INBOX, or None when the mailbox cannot be read.

    ``readonly=True`` plus ``BODY.PEEK`` means the server is left untouched: no
    message is marked seen, nothing is moved or deleted.
    """
    if client is None:
        return None
    try:
        if client.select("INBOX", readonly=True)[0] != "OK":  # type: ignore[attr-defined]
            return None
        numbers = _search(client, "UNSEEN")
        return _inbox(client, numbers, len(numbers))
    except (imaplib.IMAP4.error, OSError, TimeoutError, ValueError) as exc:
        log.info("mail: reading the inbox failed: %s", exc)
        return None


def _inbox(client: object, numbers: list[int], total: int) -> Inbox:
    if len(numbers) > MAX_MESSAGES:
        numbers = numbers[-MAX_MESSAGES:]  # the newest messages in the mailbox
    messages = tuple(_header(client, number) for number in numbers)
    dated = [message.when for message in messages if message.when is not None]
    now = datetime.now(UTC)
    days = 0.0
    if dated:
        oldest = min(dated)
        if oldest <= now:
            days = round((now - oldest).total_seconds() / 86400.0, 1)
    senders: list[str] = []
    for message in messages:
        if message.sender and message.sender not in senders:
            senders.append(message.sender)
    subjects = tuple(dict.fromkeys(message.subject for message in messages if message.subject))
    return Inbox(
        count=total,
        senders=tuple(senders),
        subjects=subjects,
        days=days,
        messages=messages,
    )


def _known_contact(message: Message, contacts: dict[str, str]) -> bool:
    """True when the sender is one of the people in ``Settings.contacts``."""
    if message.address:
        for address in contacts.values():
            if address and address.casefold() == message.address.casefold():
                return True
    spoken = contact_key(message.sender)
    return bool(spoken) and spoken in {contact_key(name) for name in contacts}


def _company_sender(message: Message) -> bool:
    """A sender on something other than a personal mail domain: a shop, a courier."""
    address = message.address.casefold()
    if "@" not in address:
        return False
    domain = address.split("@", 1)[1]
    return not any(address.endswith(host) or domain == host for host in _MAIL_DOMAINS)


def _subject_core(subject: str) -> str:
    """Drop "[list name]" tags so a mailing-list reply still reads as a reply."""
    return _TAGGED_SUBJECT.sub("", subject).strip()


def score(message: Message, contacts: dict[str, str]) -> int:
    """How important one message looks: contact, business words, replies to the user."""
    text = f"{message.subject} {message.sender}"
    points = 0
    if _known_contact(message, contacts):
        points += 4
    if _IMPORTANT_WORDS.search(text):
        points += 3
    if _REPLY_SUBJECT.match(_subject_core(message.subject)):
        points += 2
    if _company_sender(message):
        points += 1
    return points


def importance(messages: Iterable[Message], contacts: dict[str, str]) -> list[Message]:
    """Rank messages most important first; a score of 0 means nothing stands out."""
    pool = list(messages)
    return sorted(pool, key=lambda message: (score(message, contacts), _stamp(message)), reverse=True)


def _stamp(message: Message) -> float:
    return message.when.timestamp() if message.when is not None else 0.0


def important(messages: Iterable[Message], contacts: dict[str, str]) -> list[Message]:
    """The messages that stand out, most important first, at most ``SPOKEN_IMPORTANT`` of them."""
    ranked = [message for message in importance(messages, contacts) if score(message, contacts) > 0]
    return ranked[:SPOKEN_IMPORTANT]


def _matches(message: Message, wanted: str) -> bool:
    needle = wanted.casefold()
    if not needle:
        return False
    return (
        needle in message.sender.casefold()
        or needle == message.address.casefold()
        or needle in message.subject.casefold()
    )


def read_one(client: object, who_or_subject: str) -> str:
    """The decoded body of the message from or about that person or subject, or "".

    Unread mail is searched first, the whole inbox only when nothing unread matches.
    """
    wanted = _clean(who_or_subject)
    if client is None or not wanted:
        return ""
    try:
        if client.select("INBOX", readonly=True)[0] != "OK":  # type: ignore[attr-defined]
            return ""
        for criterion in ("UNSEEN", "ALL"):
            for number in _search(client, criterion):
                message = _header(client, number)
                if _matches(message, wanted):
                    raw = _fetch(client, number, _BODY_FETCH)
                    if not raw:
                        continue
                    record = message_from_bytes(raw, policy=policy.default)
                    return _plain_text(record)
    except (imaplib.IMAP4.error, OSError, TimeoutError, ValueError) as exc:
        log.info("mail: reading one message failed: %s", exc)
        return ""
    return ""


def _spoken_list(items: Iterable[str]) -> str:
    words = [item for item in items if item]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} and {words[-1]}"


def spoken_none() -> str:
    return "You have no unread emails."


def spoken_summary(count: int, senders: Iterable[str], subjects: Iterable[str]) -> str:
    """One spoken sentence about the unread mail; at most three senders are named."""
    if count <= 0:
        return spoken_none()
    names = [sender for sender in senders if sender]
    shown = names[:SPOKEN_SENDERS]
    hidden = len(names) - len(shown)
    who = _spoken_list(shown)
    if not who:
        return f"You have {count} unread email{'' if count == 1 else 's'}."
    tail = f", and {hidden} more" if hidden > 0 else ""
    line = f"You have {count} unread email{'' if count == 1 else 's'}, from {who}{tail}."
    topics = [str(subject) for subject in subjects if str(subject)]
    if not topics:
        return line
    return f"{line} They are about {_spoken_list(topics[:SPOKEN_SENDERS])}."


def spoken_important(messages: Iterable[Message]) -> str:
    """The messages worth reading aloud first, or a plain summary line."""
    chosen = list(messages)
    if not chosen:
        return "Nothing in your unread mail looks important."
    parts = [
        f"{message.sender or 'someone'}: {message.subject or 'no subject'}"
        for message in chosen
    ]
    return "The important ones are " + "; ".join(parts) + "."


def spoken_too_many(count: int, shown: int) -> str:
    """Say the inbox is big without reading it all out."""
    if shown <= 0:
        return f"You have {count} unread emails and none of them looks important."
    return (
        f"You have {count} unread emails. I'll read the {shown} that look most important, "
        "and you can ask for the rest."
    )
