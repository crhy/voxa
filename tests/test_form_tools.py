from __future__ import annotations

import pytest

from voxa.agent.tools import default_registry, forms
from voxa.agent.tools.forms import field_key, fill_form, form_fields, profile_values

OUTLINE = [
    {"n": 1, "kind": "link", "text": "Sign in", "placeholder": "", "href": "/login"},
    {"n": 2, "kind": "field", "text": "Your name", "placeholder": "", "href": ""},
    {"n": 3, "kind": "field", "text": "Street address", "placeholder": "", "href": ""},
    {"n": 4, "kind": "field", "text": "Email", "placeholder": "", "href": ""},
    {"n": 5, "kind": "field", "text": "Phone", "placeholder": "", "href": ""},
    {"n": 6, "kind": "button", "text": "Submit", "placeholder": "", "href": ""},
]


class FakeSession:
    """Records what the form tools ask for; no browser, no network, no sound."""

    def __init__(self, outline=None, takes=None):
        self._outline = list(outline if outline is not None else OUTLINE)
        self._takes = takes
        self.typed: list[tuple[str, str, bool]] = []
        self.clicked: list[str] = []

    def outline(self, limit=60):
        return list(self._outline)

    def type_into(self, field, value, submit=False):
        self.typed.append((field, value, submit))
        if self._takes is not None and field not in self._takes:
            return False
        return True

    def click(self, text, settle_ms=600):
        self.clicked.append(text)
        return text

    def title(self):
        return "Sign up - Jane Doe"

    def url(self):
        return "https://example.com/signup"


@pytest.fixture
def jane(monkeypatch):
    """A profile that knows a name and an email, and a saved location but no street address."""
    monkeypatch.setattr(
        forms,
        "_read_memory",
        lambda: (
            "- 2026-01-05 | My name is Jane Doe\n"
            "- 2026-01-06 | My email is jane.doe@example.com\n"
        ),
    )
    monkeypatch.setattr(forms.contact_store, "get_contacts", lambda: {"jane doe": "jane.doe@example.com"})
    monkeypatch.setattr(forms, "_home_location", lambda: "Portland, Oregon")
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: FakeSession())
    return {"name": "Jane Doe", "address": "Portland, Oregon", "email": "jane.doe@example.com"}


def test_form_fields_reads_labels_from_the_outline():
    assert form_fields(OUTLINE) == ["Your name", "Street address", "Email", "Phone"]
    assert form_fields([]) == []


def test_field_key_maps_labels_to_profile_values():
    assert field_key("Your name") == "name"
    assert field_key("Street address") == "address"
    assert field_key("Email") == "email"
    assert field_key("E-mail address") == "email"
    assert field_key("Phone") == ""
    assert field_key("File name") == ""


def test_profile_values_come_from_facts_contacts_and_location(jane):
    assert profile_values() == jane


def test_profile_values_are_empty_when_nothing_is_known(monkeypatch):
    monkeypatch.setattr(forms, "_read_memory", lambda: "")
    monkeypatch.setattr(forms.contact_store, "get_contacts", None)
    monkeypatch.setattr(forms, "_home_location", lambda: "")
    assert profile_values() == {"name": "", "address": "", "email": ""}


def test_profile_values_read_the_email_saved_for_the_person_themselves(monkeypatch):
    monkeypatch.setattr(forms, "_read_memory", lambda: "- 2026-01-05 | My name is Jane Doe")
    monkeypatch.setattr(
        forms.contact_store,
        "get_contacts",
        lambda: {"jane doe": "jane.doe@example.com", "bob": "bob@example.com"},
    )
    monkeypatch.setattr(forms, "_home_location", lambda: "")
    assert profile_values() == {"name": "Jane Doe", "address": "", "email": "jane.doe@example.com"}


def test_fill_form_types_only_the_fields_it_knows(jane):
    session = FakeSession()
    outcome = fill_form(session, jane, form_fields(OUTLINE))
    assert session.typed == [
        ("Your name", "Jane Doe", False),
        ("Street address", "Portland, Oregon", False),
        ("Email", "jane.doe@example.com", False),
    ]
    assert outcome["filled"] == ["Your name", "Street address", "Email"]
    assert outcome["unknown"] == ["Phone"]
    assert outcome["failed"] == []


def test_fill_form_never_submits(jane):
    session = FakeSession()
    fill_form(session, jane, form_fields(OUTLINE))
    assert [submit for _field, _value, submit in session.typed] == [False, False, False]
    assert session.clicked == []


def test_fill_form_names_the_fields_the_page_refused(jane):
    session = FakeSession(takes={"Email"})
    outcome = fill_form(session, jane, form_fields(OUTLINE))
    assert outcome["filled"] == ["Email"]
    assert outcome["failed"] == ["Your name", "Street address"]
    assert outcome["unknown"] == ["Phone"]


def test_form_read_tool_lists_the_fields(jane):
    result = default_registry().call("form_read", {})
    assert result.ok
    assert result.speech == "This form has 4 fields: Your name, Street address, Email and Phone."


def test_form_fill_tool_fills_from_the_profile_and_asks_for_the_rest(jane):
    result = default_registry().call("form_fill", {})
    assert result.ok
    assert result.speech == (
        "Filled in Your name, Street address and Email. What should go in Phone?"
        " I have not sent it."
    )


def test_form_fill_tool_never_submits(jane, monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    default_registry().call("form_fill", {})
    assert session.clicked == []
    assert [submit for _field, _value, submit in session.typed] == [False, False, False]


def test_form_fill_tool_without_a_form(monkeypatch):
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: FakeSession(outline=[]))
    result = default_registry().call("form_fill", {})
    assert not result.ok
    assert result.speech == "There is no form on this page."


def test_form_fill_tool_asks_when_the_profile_knows_nothing(monkeypatch):
    monkeypatch.setattr(forms, "_read_memory", lambda: "")
    monkeypatch.setattr(forms.contact_store, "get_contacts", None)
    monkeypatch.setattr(forms, "_home_location", lambda: "")
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    result = default_registry().call("form_fill", {})
    assert not result.ok
    assert result.speech == (
        "I don't have what this form asks for. What should go in "
        "Your name, Street address, Email and Phone?"
    )
    assert session.typed == []


def test_form_submit_refuses_without_the_yes(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    for spoken in ("", "yeah", "sure", "yes please", "no", "YES PLEASE"):
        result = default_registry().call("form_submit", {"confirm": spoken})
        assert not result.ok
        assert result.speech == "I only send a form when you say yes. Say yes and I'll press the button."
    assert session.clicked == []


def test_form_submit_refuses_without_the_parameter(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    result = default_registry().call("form_submit", {})
    assert not result.ok
    assert session.clicked == []


def test_form_submit_presses_the_button_after_yes(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    result = default_registry().call("form_submit", {"confirm": "yes"})
    assert result.ok
    assert result.speech == "Pressed Submit."
    assert session.clicked == ["Submit"]
    assert session.typed == []


def test_form_submit_presses_the_button_the_user_names(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    result = default_registry().call("form_submit", {"confirm": "yes", "button": "Send"})
    assert result.ok
    assert session.clicked == ["Send"]


def test_form_submit_without_a_button_to_press(monkeypatch):
    session = FakeSession(outline=[{"n": 1, "kind": "field", "text": "Your name", "placeholder": "", "href": ""}])
    monkeypatch.setattr("voxa.agent.tools.web.get_session", lambda: session)
    result = default_registry().call("form_submit", {"confirm": "yes"})
    assert not result.ok
    assert result.speech == "I can't find a button to press on this page."
    assert session.clicked == []


def test_form_tools_are_registered():
    names = set(default_registry().names())
    assert {"form_read", "form_fill", "form_submit"} <= names
