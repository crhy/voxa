"""A dictated email: who to, the subject, the body — then a draft the user sends himself."""

import re
import string

ASK_SUBJECT = "What is the subject?"
ASK_BODY = "What do you want to say? Say stop dictation when you are done."
CANCELLED = "Okay, no email."

_ADDRESS_RE = re.compile(r"^[a-z0-9._+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+$")

_WORD_MAP = {
    "at": "@",
    "dot": ".",
    "underscore": "_",
    "dash": "-",
    "hyphen": "-",
}

_CANCEL_WORDS = {"cancel", "cancel that", "never mind", "forget it"}
_NEW_REQUEST = re.compile(
    r"^(?:please\s+)?(?:open|launch|start|run|close|delete|remove|install|update|"
    r"change|set|order|buy|turn|switch|hack|search|find|play|stop|create|write|draft)\b",
    re.I,
)


_MAIL_DOMAINS = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
)


def spoken_address(text: str) -> str:
    """An email address said out loud, or ""."""
    joined = "".join(
        _WORD_MAP.get(word, word) for word in text.lower().split()
    )
    if _ADDRESS_RE.match(joined):
        return joined
    if "@" not in joined:
        for domain in _MAIL_DOMAINS:
            if joined.endswith(domain):
                rest = joined[: -len(domain)]
                for glue in ("at", "an"):
                    if rest.endswith(glue) and len(rest) > len(glue):
                        rebuilt = f"{rest[: -len(glue)]}@{domain}"
                        if _ADDRESS_RE.match(rebuilt):
                            return rebuilt
    return ""


def contact_key(spoken: str) -> str:
    """lower-case, a leading "my " removed, punctuation stripped: "My Mom." -> "mom"."""
    key = spoken.lower()
    if key.startswith("my "):
        key = key[len("my "):]
    return "".join(ch for ch in key if ch not in string.punctuation)


def resolve(spoken: str, contacts: dict[str, str]) -> str:
    """spoken_address(spoken) if non-empty; else contacts.get(contact_key(spoken), "")."""
    addr = spoken_address(spoken)
    if addr:
        return addr
    return contacts.get(contact_key(spoken), "")


def interrupts_address_prompt(text: str) -> bool:
    """True when an address reply is clearly a fresh, unrelated command."""
    cleaned = text.strip()
    return bool(_NEW_REQUEST.match(cleaned)) and not bool(spoken_address(cleaned))


class MailFlow:
    def __init__(self, to_spoken: str, contacts: dict[str, str] | None = None) -> None:
        self.to_spoken = to_spoken
        self.to = resolve(to_spoken, contacts or {})
        self.subject = ""
        self.parts: list[str] = []
        self.learned: tuple[str, str] | None = None
        self.step = "address" if to_spoken and self.to == "" else "subject"

    def first_question(self) -> str:
        if self.step == "address":
            return f"What is the email address for {self.to_spoken}? Say it, or say skip."
        return ASK_SUBJECT

    def feed(self, text: str, control: str | None = None) -> tuple[str, bool]:
        """Returns (what Voxa says, ready). ready is True exactly once: when the draft should be opened."""
        cleaned = text.lower().strip().strip(".!?")
        if cleaned in _CANCEL_WORDS:
            self.step = "done"
            return (CANCELLED, False)

        if self.step == "address":
            if cleaned in {"skip", "skip it", "no"}:
                self.step = "subject"
                return (ASK_SUBJECT, False)
            addr = spoken_address(text)
            if addr:
                self.to = addr
                self.learned = (contact_key(self.to_spoken), addr)
                self.step = "subject"
                return (f"Got it. {ASK_SUBJECT}", False)
            return (
                "I did not catch an email address. Say it again, or say skip.",
                False,
            )

        if self.step == "subject":
            if control == "stop":
                self.step = "done"
                return (CANCELLED, False)
            if cleaned in {"no subject", "skip"}:
                self.subject = ""
            else:
                subject = text.strip()
                if subject:
                    subject = subject[0].upper() + subject[1:]
                    if subject.endswith("."):
                        subject = subject[:-1]
                self.subject = subject
            self.step = "body"
            return (ASK_BODY, False)

        if self.step == "body":
            if control == "undo":
                if self.parts:
                    self.parts.pop()
                    return ("Removed.", False)
                return ("Nothing to remove.", False)
            if control in {"stop", "send"}:
                self.step = "done"
                if not self.parts:
                    return (CANCELLED, False)
                who = f" to {self.to_spoken}" if self.to_spoken else ""
                return (f"Your email{who} is ready. Check it and press Send.", True)
            self.parts.append(text)
            return ("", False)

        return ("", False)

    @property
    def body(self) -> str:
        processed = []
        for part in self.parts:
            p = part.strip()
            if not p:
                continue
            p = p[0].upper() + p[1:]
            if p[-1] not in ".!?":
                p += "."
            processed.append(p)
        return " ".join(processed)

    @property
    def active(self) -> bool:
        return self.step != "done"

    @property
    def caption(self) -> str:
        return {
            "address": "Email address",
            "subject": "Email subject",
            "body": "Email body",
        }.get(self.step, "")


def wants_dictated_email(to: str, topic: str) -> bool:
    """True when topic is empty: the user gave nothing to write about, so he dictates."""
    return topic == ""
