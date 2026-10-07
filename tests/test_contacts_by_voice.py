from __future__ import annotations

import pytest

from voxa.agent.intents import route
from voxa.agent.tools.contacts import (
    _forget_contact_email_handler,
    _get_contact_email_handler,
    _set_contact_email_handler,
    get_contacts,
    save_contacts,
)

CONTACT_TOOLS = {"set_contact_email", "get_contact_email", "forget_contact_email"}


@pytest.fixture
def store(monkeypatch):
    data: dict[str, str] = {}
    monkeypatch.setattr("voxa.agent.tools.contacts.get_contacts", lambda: data)
    monkeypatch.setattr("voxa.agent.tools.contacts.save_contacts", lambda c: data.update(c))
    return data


def test_save(store):
    res = _set_contact_email_handler({"name": "Bob", "address": "bob at example dot com"})
    assert res.ok
    assert res.speech == "Saved. Bob's email is bob@example.com."


def test_read_back(store):
    _set_contact_email_handler({"name": "Bob", "address": "bob at example dot com"})
    res = _get_contact_email_handler({"name": "Bob"})
    assert res.ok
    assert res.speech == "Bob's email is bob@example.com."


def test_forget(store):
    _set_contact_email_handler({"name": "Bob", "address": "bob at example dot com"})
    res = _forget_contact_email_handler({"name": "Bob"})
    assert res.ok
    assert res.speech == "Forgotten bob's email."


def test_unknown_name(store):
    res = _get_contact_email_handler({"name": "Zoe"})
    assert not res.ok
    assert res.speech == "I do not have an email address for zoe."


def test_bad_address(store):
    res = _set_contact_email_handler({"name": "Bob", "address": "not an address"})
    assert not res.ok
    assert res.speech == "I did not catch that email address."


def test_hooks_missing():
    assert get_contacts is None
    assert save_contacts is None
    res = _set_contact_email_handler({"name": "Bob", "address": "bob at example dot com"})
    assert not res.ok
    assert res.speech == "I cannot save that here."


def test_possessive_keys(store):
    for spoken in ("Bob's", "my mom's", "moms'"):
        res = _set_contact_email_handler({"name": spoken, "address": "x at example dot com"})
        assert res.ok
        got = _get_contact_email_handler({"name": spoken})
        assert got.ok


def test_router_set():
    call = route("Mom's email is mom at example dot com")
    assert call is not None and call.tool == "set_contact_email"
    assert call.args["name"] == "Mom"
    assert call.args["address"] == "mom at example dot com"


def test_router_get():
    call = route("what is Bob's email?")
    assert call is not None and call.tool == "get_contact_email"
    assert call.args["name"] == "Bob"


def test_router_forget():
    call = route("forget Bob's email")
    assert call is not None and call.tool == "forget_contact_email"
    assert call.args["name"] == "Bob"


def test_router_not_email_tools():
    for phrase in ("my email is broken", "open my email", "check my email"):
        call = route(phrase)
        assert call is None or call.tool not in CONTACT_TOOLS
