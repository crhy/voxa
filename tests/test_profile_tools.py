"""The personal profile by voice (#41): "my address is …", "what address do you have?".

The profile is the plain, non-secret half of #41: name, home address, phone
number. It lives in the settings file, is saved the way contacts are, and is
read back out loud.
"""

from __future__ import annotations

import pytest

from voxa.agent.intents import ROUTED_TOOLS, route
from voxa.agent.profile import (
    ADDRESS_FIELD,
    NAME_FIELD,
    PHONE_FIELD,
    clean_value,
    profile_forget,
    profile_get,
    profile_set,
    spoken_profile,
)
from voxa.agent.tools import default_registry
from voxa.agent.tools.profile import _forget_handler, _get_handler, _set_handler, profile_tools
from voxa.config import ConfigStore, Settings

PROFILE_TOOLS = {
    "set_name",
    "get_name",
    "forget_name",
    "set_home_address",
    "get_home_address",
    "forget_home_address",
    "set_phone_number",
    "get_phone_number",
    "forget_phone_number",
}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A settings file in a temporary directory, wired to the tools the way window.py does."""
    store = ConfigStore(tmp_path / "config.json")
    current = Settings()

    def save(field: str, value: str) -> None:
        if value:
            profile_set(current, field, value)
        else:
            profile_forget(current, field)
        store.save(current)

    monkeypatch.setattr("voxa.agent.tools.profile.get_profile", lambda field: profile_get(current, field))
    monkeypatch.setattr("voxa.agent.tools.profile.save_profile", save)
    return current


def test_field_names_are_the_settings_fields():
    assert (NAME_FIELD, ADDRESS_FIELD, PHONE_FIELD) == ("name", "home_address", "phone_number")
    assert spoken_profile(ADDRESS_FIELD, "12 oak street") == "Your home address is 12 oak street."
    assert spoken_profile(NAME_FIELD, "Rhy") == "Your name is Rhy."
    assert spoken_profile(PHONE_FIELD, "555 1212") == "Your phone number is 555 1212."


def test_profile_helpers_work_on_plain_strings(settings):
    assert profile_set(settings, ADDRESS_FIELD, "  12   oak  street  ")
    assert profile_get(settings, ADDRESS_FIELD) == "12 oak street"
    assert profile_forget(settings, ADDRESS_FIELD)
    assert profile_get(settings, ADDRESS_FIELD) == ""
    assert not profile_forget(settings, ADDRESS_FIELD)
    assert not profile_set(settings, "secret", "hunter2")
    assert not profile_set(settings, ADDRESS_FIELD, "   ")


def test_spoken_address_is_saved(settings):
    result = _set_handler(ADDRESS_FIELD, {"home_address": "12 oak street"})
    assert result.ok
    assert result.speech == "Your home address is 12 oak street."
    assert settings.home_address == "12 oak street"


def test_address_is_read_back(settings):
    _set_handler(ADDRESS_FIELD, {"home_address": "12 oak street"})
    result = _get_handler(ADDRESS_FIELD, {})
    assert result.ok
    assert result.speech == "Your home address is 12 oak street."


def test_address_is_forgotten(settings):
    _set_handler(ADDRESS_FIELD, {"home_address": "12 oak street"})
    result = _forget_handler(ADDRESS_FIELD, {})
    assert result.ok
    assert result.speech == "Forgotten your home address."
    assert settings.home_address == ""


def test_reading_back_something_not_saved(settings):
    result = _get_handler(ADDRESS_FIELD, {})
    assert not result.ok
    assert result.speech == "I do not have your home address."


def test_name_and_phone_use_the_same_shape(settings):
    assert _set_handler(NAME_FIELD, {"name": "Rhy"}).ok
    assert _set_handler(PHONE_FIELD, {"phone_number": "555 1212"}).ok
    assert _get_handler(NAME_FIELD, {}).speech == "Your name is Rhy."
    assert _get_handler(PHONE_FIELD, {}).speech == "Your phone number is 555 1212."
    assert _forget_handler(PHONE_FIELD, {}).speech == "Forgotten your phone number."


def test_missing_hooks_refuse():
    """Without the window's injection points nothing is saved, and she says so."""
    from voxa.agent.tools import profile as tools_profile

    assert tools_profile.get_profile is None
    assert tools_profile.save_profile is None
    assert _set_handler(ADDRESS_FIELD, {"home_address": "12 oak street"}).speech == "I cannot save that here."


def test_tools_are_registered_and_reversible():
    tools = {tool.name: tool for tool in profile_tools()}
    assert set(tools) == PROFILE_TOOLS
    for tool in tools.values():
        assert tool.risk.name == "REVERSIBLE"
    names = default_registry().names()
    assert PROFILE_TOOLS <= set(names)
    assert ROUTED_TOOLS <= set(names)


def test_the_registry_calls_them(settings):
    registry = default_registry()
    assert registry.call("set_home_address", {"home_address": "12 oak street"}).ok
    assert registry.call("get_home_address", {}).speech == "Your home address is 12 oak street."
    assert registry.call("forget_home_address", {}).ok
    assert not registry.call("get_home_address", {}).ok


def test_router_saves_an_address():
    call = route("my address is 12 oak street")
    assert call is not None
    assert call.tool == "set_home_address"
    assert call.args == {"home_address": "12 oak street"}
    assert route("my home address is 7 high street").args == {"home_address": "7 high street"}


def test_router_reads_the_address_back():
    for phrase in ("what address do you have?", "what is my address?", "what's your home address"):
        call = route(phrase)
        assert call is not None and call.tool == "get_home_address", phrase
        assert call.args == {}


def test_router_forgets_the_address():
    for phrase in ("forget my address", "forget the address", "forget my home address"):
        call = route(phrase)
        assert call is not None and call.tool == "forget_home_address", phrase


def test_router_name_and_phone():
    assert route("my name is Rhy").tool == "set_name"
    assert route("my name is Rhy").args == {"name": "Rhy"}
    assert route("what is my name?").tool == "get_name"
    assert route("forget my name").tool == "forget_name"
    assert route("my phone number is 555 1212").tool == "set_phone_number"
    assert route("my phone is 555 1212").args == {"phone_number": "555 1212"}
    assert route("what is my phone number?").tool == "get_phone_number"
    assert route("forget my phone number").tool == "forget_phone_number"


def test_the_router_keeps_untouched_phrases_elsewhere():
    for phrase in ("my email is bob at example dot com", "what is bob's email?", "open my email"):
        call = route(phrase)
        assert call is None or call.tool not in PROFILE_TOOLS, phrase
    assert route("what is the name of this window") is None


def test_the_profile_survives_a_settings_round_trip(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings()
    settings.name = "Rhy"
    settings.home_address = "12   oak\tstreet"
    settings.phone_number = "555 1212"
    store.save(settings)
    loaded = store.load()
    assert loaded.name == "Rhy"
    assert loaded.home_address == "12 oak street"
    assert loaded.phone_number == "555 1212"


def test_clean_value_caps_a_long_dictation():
    assert len(clean_value("word " * 300)) == 200
