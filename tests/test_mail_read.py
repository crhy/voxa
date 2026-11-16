"""Reading the inbox hands-free (#56, part 1): parsing, ranking, spoken forms.

Every mailbox here is a fake that answers in imaplib's own reply shape, so the
whole file runs with no network and no mail account.
"""

from datetime import UTC, datetime, timedelta

from voxa.agent import mailread
from voxa.agent.mailread import Message
from voxa.config import Settings


def _raw(sender: str, subject: str, when: str, body: str = "") -> bytes:
    return (
        f"From: {sender}\r\nSubject: {subject}\r\nDate: {when}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Content-Transfer-Encoding: quoted-printable\r\n\r\n"
        f"{body}"
    ).encode()


def _header_bytes(raw: bytes) -> bytes:
    head = raw.split(b"\r\n\r\n", 1)[0]
    kept = [line for line in head.split(b"\r\n") if line.split(b":", 1)[0] in {b"From", b"Subject", b"Date"}]
    return b"\r\n".join(kept) + b"\r\n"


class FakeMailbox:
    """Answers like imaplib: (status, data) with RFC 2252 literals in the fetch pairs."""

    def __init__(self, messages: dict[int, bytes], unseen: str = "1 2 3") -> None:
        self.messages = messages
        self.unseen = unseen
        self.select_calls: list[tuple[str, bool]] = []
        self.queries: list[str] = []

    def select(self, mailbox: str, readonly: bool = False):
        self.select_calls.append((mailbox, readonly))
        return ("OK", b"3")

    def search(self, criterion: str):
        if criterion == "UNSEEN":
            return ("OK", [self.unseen.encode()])
        return ("OK", [" ".join(str(number) for number in self.messages).encode()])

    def fetch(self, number: int, query: str):
        self.queries.append(query)
        raw = self.messages.get(number)
        if raw is None:
            return ("BAD", None)
        body = raw if query == "(BODY.PEEK[])" else _header_bytes(raw)
        return ("OK", [(f"{number} (BODY.PEEK[...] {{{len(body)}}}".encode(), body)])


def _message(number: int, sender: str, address: str, subject: str, when: datetime) -> Message:
    return Message(number=number, sender=sender, address=address, subject=subject, when=when)


def _mailbox() -> FakeMailbox:
    return FakeMailbox(
        {
            1: _raw("Sam <sam@example.com>", "Thursday works", "Tue, 06 Oct 2026 09:30:00 +0000"),
            2: _raw("Alex <alex@gmail.com>", "Party plans", "Mon, 05 Oct 2026 09:00:00 +0000"),
            3: _raw(
                "Acme Receipts <billing@acme.co.uk>",
                "Invoice 9912 due",
                "Wed, 07 Oct 2026 09:00:00 +0000",
            ),
        },
        unseen="1 2 3",
    )


def test_unread_lists_count_senders_subjects_and_days() -> None:
    inbox = mailread.unread(_mailbox())
    assert inbox is not None
    assert inbox.count == 3
    assert inbox.senders == ("Sam", "Alex", "Acme Receipts")
    assert inbox.subjects == ("Thursday works", "Party plans", "Invoice 9912 due")
    assert inbox.days >= 2.0


def test_reading_the_inbox_leaves_the_server_read_only() -> None:
    mailbox = _mailbox()
    mailread.unread(mailbox)
    assert mailbox.select_calls and all(readonly for _, readonly in mailbox.select_calls)
    assert all("(BODY.PEEK" in query for query in mailbox.queries)
    assert not any("STORE" in query or query == "BODY[]" for query in mailbox.queries)


def test_unread_returns_none_when_the_mailbox_refuses() -> None:
    class Refusing:
        def select(self, mailbox: str, readonly: bool = False):
            return ("NO", b"no such folder")

    assert mailread.unread(Refusing()) is None
    assert mailread.unread(None) is None


def test_importance_puts_contacts_and_business_words_first() -> None:
    now = datetime.now(UTC)
    pool = [
        _message(1, "Sam", "sam@gmail.com", "Thursday works", now - timedelta(hours=1)),
        _message(2, "Alex", "alex@gmail.com", "Party plans", now - timedelta(hours=5)),
        _message(3, "Acme Receipts", "billing@acme.co.uk", "Invoice 9912 due", now - timedelta(hours=9)),
        _message(4, "Jordan", "jordan@gmail.com", "Dentist appointment", now - timedelta(hours=20)),
    ]
    contacts = {"sam": "sam@gmail.com"}
    ranked = mailread.importance(pool, contacts)
    assert [message.sender for message in ranked] == ["Sam", "Acme Receipts", "Jordan", "Alex"]
    assert mailread.score(pool[1], contacts) == 0
    # a reply into a thread the user started counts on its own, even behind a list tag
    assert mailread.score(_message(9, "Alex", "alex@gmail.com", "Re: Thursday", now), {}) == 2
    assert mailread.score(_message(9, "Alex", "alex@gmail.com", "[book-club] Re: Thursday", now), {}) == 2
    # a known contact is worth more than a shop on a company domain
    assert mailread.score(pool[0], contacts) == 4
    assert [message.sender for message in mailread.important(pool, contacts)] == [
        "Sam",
        "Acme Receipts",
        "Jordan",
    ]


def test_read_one_decodes_a_quoted_printable_body() -> None:
    mailbox = FakeMailbox(
        {
            1: _raw(
                "Sam <sam@example.com>",
                "Thursday works",
                "Tue, 06 Oct 2026 09:30:00 +0000",
                "Hi Alex, the meet=\r\nings on Thursday work=\r\ns for both of us.",
            )
        },
        unseen="1",
    )
    body = mailread.read_one(mailbox, "Sam")
    assert body == "Hi Alex, the meetings on Thursday works for both of us."


def test_read_one_falls_back_to_the_whole_inbox_and_misses_quietly() -> None:
    mailbox = FakeMailbox(
        {2: _raw("Pat <pat@example.com>", "Gas bill", "Tue, 06 Oct 2026 09:30:00 +0000")},
        unseen="",
    )
    assert "Gas bill" not in mailread.read_one(mailbox, "Pat")
    assert mailread.read_one(mailbox, "nobody") == ""
    assert mailread.read_one(None, "Sam") == ""


def test_spoken_summary_names_at_most_three_senders() -> None:
    speech = mailread.spoken_summary(
        7,
        ["Sam", "Alex", "Jordan", "Pat", "Tony"],
        ["Invoice", "Dentist", "Gas bill", "Holiday"],
    )
    assert speech == (
        "You have 7 unread emails, from Sam, Alex and Jordan, and 2 more. "
        "They are about Invoice, Dentist and Gas bill."
    )


def test_spoken_forms_for_empty_and_overwhelming_inboxes() -> None:
    assert mailread.spoken_none() == "You have no unread emails."
    assert mailread.spoken_summary(0, [], []) == "You have no unread emails."
    assert mailread.spoken_summary(1, ["Sam"], []) == "You have 1 unread email, from Sam."
    assert mailread.spoken_too_many(412, 3) == (
        "You have 412 unread emails. I'll read the 3 that look most important, "
        "and you can ask for the rest."
    )
    assert mailread.spoken_too_many(412, 0) == "You have 412 unread emails and none of them looks important."
    pool = [_message(1, "Sam", "sam@example.com", "Thursday works", datetime.now(UTC))]
    assert mailread.spoken_important(pool) == "The important ones are Sam: Thursday works."
    assert mailread.spoken_important([]) == "Nothing in your unread mail looks important."


def connect_attempts(monkeypatch) -> list[str]:
    """Replace the only connection mailread may make with a recorder that refuses."""
    seen: list[str] = []

    class Refusing:
        def __init__(self, host: str = "", timeout=None) -> None:
            seen.append(host)
            raise OSError("connection refused")

    monkeypatch.setattr(mailread.imaplib, "IMAP4_SSL", Refusing)
    return seen


def test_connect_never_reaches_for_a_missing_account(monkeypatch) -> None:
    seen = connect_attempts(monkeypatch)
    assert mailread.connect(Settings()) is None
    assert mailread.connect(Settings(mail_host="imap.example.org", mail_user="me")) is None
    assert seen == []


def test_connect_only_uses_the_host_from_the_settings(monkeypatch) -> None:
    seen = connect_attempts(monkeypatch)
    settings = Settings(mail_host="imap.example.org", mail_user="me", mail_password="hunter2")
    assert mailread.connect(settings) is None
    assert seen == ["imap.example.org"]
