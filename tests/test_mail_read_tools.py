"""The mail-reading tools (#56, part 1): routing, handlers, and the read-only guard.

The mailbox is a fake that answers in imaplib's reply shape; the one connection
mailread is allowed to make is replaced by a recorder that refuses, so a test that
reaches a real host fails instead of passing quietly.
"""

import ast
import imaplib
import inspect

from voxa.agent import mailread
from voxa.agent.intents import route
from voxa.agent.tools import default_registry
from voxa.agent.tools.mailread import SIMULATED_READ, SIMULATED_SUMMARY
from voxa.config import ConfigStore, Settings

CONFIGURED = Settings(mail_host="imap.example.org", mail_user="me", mail_password="hunter2")
BLANK = Settings()


def _raw(sender: str, subject: str) -> bytes:
    return (
        f"From: {sender}\r\nSubject: {subject}\r\n"
        "Date: Tue, 06 Oct 2026 09:30:00 +0000\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "Thursday works for both of us."
    ).encode()


class FakeMailbox:
    def __init__(self) -> None:
        self.messages = {
            1: _raw("Sam <sam@gmail.com>", "Thursday works"),
            2: _raw("Alex <alex@gmail.com>", "Party plans"),
            3: _raw("Acme Receipts <billing@acme.co.uk>", "Invoice 9912 due"),
        }
        self.select_calls_seen: list[str] = []
        self.queries: list[str] = []

    def select(self, mailbox: str, readonly: bool = False):
        assert readonly, "reading the inbox must not be able to change it"
        self.select_calls_seen.append(mailbox)
        return ("OK", b"3")

    def search(self, criterion: str):
        if criterion == "UNSEEN":
            return ("OK", [b"1 2 3"])
        return ("OK", [b" ".join(str(number).encode() for number in self.messages)])

    def fetch(self, number: int, query: str):
        self.queries.append(query)
        raw = self.messages[number]
        body = raw if query == "(BODY.PEEK[])" else b"\r\n".join(
            line for line in raw.split(b"\r\n\r\n")[0].split(b"\r\n")
            if line.split(b":", 1)[0] in {b"From", b"Subject", b"Date"}
        ) + b"\r\n"
        return ("OK", [(f"{number} (BODY.PEEK[...] {{{len(body)}}}".encode(), body)])


def use_settings(monkeypatch, settings: Settings = CONFIGURED) -> None:
    monkeypatch.setattr(ConfigStore, "load", lambda self, **_kwargs: settings)


def refusing_connection(monkeypatch) -> list[str]:
    """Replace imaplib's only entry point with a recorder that refuses to connect."""
    seen: list[str] = []

    class Refusing:
        def __init__(self, host: str = "", timeout=None) -> None:
            seen.append(host)
            raise OSError("connection refused")

    monkeypatch.setattr(imaplib, "IMAP4_SSL", Refusing)
    return seen


# --- routing ---


def test_reading_phrases_reach_the_new_tools():
    expected = {
        "any important email?": "mail_summary",
        "how many unread emails?": "mail_summary",
        "what's in my inbox?": "mail_summary",
        "read me the email from Sam": "mail_read",
    }
    for phrase, tool in expected.items():
        call = route(phrase)
        assert call is not None, phrase
        assert call.tool == tool, phrase
    assert route("read me the email from Sam").args == {"who": "Sam"}


def _registry():
    return default_registry()


def test_the_new_tools_are_registered_read_only():
    registry = _registry()
    for name in ("mail_summary", "mail_read"):
        assert name in registry.names()
        assert int(registry.get(name).risk) == 0


def test_opening_mail_still_opens_the_browser():
    for phrase in ("open gmail", "show my email", "check my inbox", "open my mail"):
        call = route(phrase)
        assert call is not None and call.tool == "open_site", phrase


# --- handlers ---


def test_mail_summary_reads_the_fake_inbox(monkeypatch):
    use_settings(monkeypatch)
    monkeypatch.setattr(mailread, "client_factory", FakeMailbox)
    result = _registry().call("mail_summary", {})
    assert result.ok, result.speech
    assert "3 unread emails" in result.speech
    assert "Sam" in result.speech and "Acme Receipts" in result.speech
    assert "Invoice 9912 due" in result.speech


def test_mail_read_speaks_one_body(monkeypatch):
    use_settings(monkeypatch)
    monkeypatch.setattr(mailread, "client_factory", FakeMailbox)
    result = _registry().call("mail_read", {"who": "Sam"})
    assert result.ok
    assert result.speech == "Thursday works for both of us."


def test_mail_read_says_when_nothing_matches(monkeypatch):
    use_settings(monkeypatch)
    monkeypatch.setattr(mailread, "client_factory", FakeMailbox)
    result = _registry().call("mail_read", {"who": "nobody"})
    assert not result.ok
    assert "nobody" in result.speech


def test_both_tools_refuse_without_an_account(monkeypatch):
    use_settings(monkeypatch, BLANK)
    seen = refusing_connection(monkeypatch)
    registry = _registry()
    for name, args in (("mail_summary", {}), ("mail_read", {"who": "Sam"})):
        result = registry.call(name, args)
        assert not result.ok
        assert result.speech == mailread.NOT_CONFIGURED
    assert seen == []


def test_an_unreachable_server_is_reported_not_raised(monkeypatch):
    use_settings(monkeypatch)
    seen = refusing_connection(monkeypatch)
    result = _registry().call("mail_summary", {})
    assert not result.ok
    assert result.speech == mailread.UNREADABLE
    assert seen == ["imap.example.org"]


def test_simulation_never_opens_the_mailbox(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    use_settings(monkeypatch)
    seen = refusing_connection(monkeypatch)
    registry = _registry()
    assert registry.call("mail_summary", {}).speech == SIMULATED_SUMMARY
    assert registry.call("mail_read", {"who": "Sam"}).speech == SIMULATED_READ
    assert seen == []


# --- the read-only guard ---


def test_mailread_imports_no_http_socket_or_shell_module():
    tree = ast.parse(inspect.getsource(mailread))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    banned = {"aiohttp", "requests", "urllib", "socket", "http", "ssl", "subprocess", "os", "shutil"}
    assert not imported & banned, sorted(imported & banned)
    assert imported <= {
        "__future__",
        "imaplib",
        "logging",
        "re",
        "collections.abc",
        "dataclasses",
        "datetime",
        "email",
        "email.utils",
        "voxa.agent.mailflow",
    }


def test_the_only_connection_uses_the_host_from_the_settings(monkeypatch):
    seen = refusing_connection(monkeypatch)
    assert mailread.connect(Settings(mail_host="imap.example.org", mail_user="me", mail_password="pw")) is None
    assert mailread.connect(Settings()) is None
    assert seen == ["imap.example.org"]
    source = inspect.getsource(mailread)
    assert "imaplib.IMAP4_SSL(host" in source
    assert "settings.mail_host" in source or 'getattr(settings, "mail_host"' in source
    assert "imaplib.IMAP4(" not in source  # never a plaintext connection


def test_nothing_in_the_read_path_writes(monkeypatch):
    mailbox = FakeMailbox()
    monkeypatch.setattr(mailread, "client_factory", lambda: mailbox)
    use_settings(monkeypatch)
    registry = _registry()
    assert registry.call("mail_summary", {}).ok
    assert registry.call("mail_read", {"who": "Sam"}).ok
    assert set(mailbox.select_calls_seen) == {"INBOX"}
    assert mailbox.queries and all("(BODY.PEEK" in query for query in mailbox.queries)
