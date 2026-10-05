from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

from voxa.agent.homeassistant import Entity, HomeAssistant, HomeAssistantError, match_entity
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.home import NOT_CONFIGURED, home_scene, home_set, home_status, home_turn


class FakeResponse(io.BytesIO):
    pass


STATES = [
    {"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "on"},
    {"entity_id": "light.living_room", "name": "Living Room Lamp", "state": "off"},
    {"entity_id": "switch.fan", "name": "Ceiling Fan", "state": "on"},
    {"entity_id": "climate.thermostat", "name": "Thermostat", "state": "68"},
    {"entity_id": "scene.movie_night", "name": "Movie Night", "state": "unknown"},
    {"entity_id": "garage_door.door", "name": "Garage Door", "state": "closed"},
]


def fake_open(request, timeout=None):
    url = getattr(request, "full_url", str(request))
    if "/api/states" in url:
        return FakeResponse(json.dumps(STATES).encode())
    return FakeResponse(b'{"ok": true}')


def configured():
    """Patch config so url/token are set and the opener returns STATES."""
    return patch("voxa.agent.homeassistant.open_url", side_effect=fake_open)


class FakeSettings:
    home_assistant_url = "http://127.0.0.1:8123"
    home_assistant_token = "secret-token"


def with_config():
    store = patch("voxa.config.ConfigStore")
    return store


# --- states parsing ---

def test_states_returns_list_of_dicts() -> None:
    client = HomeAssistant("http://x", "tok")
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(json.dumps(STATES).encode())):
        states = client.states()
    assert isinstance(states, list)
    assert states[0]["entity_id"] == "light.kitchen"


def test_states_skips_non_list_payload() -> None:
    client = HomeAssistant("http://x", "tok")
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(b'{"a": 1}')):
        assert client.states() == []


def test_states_raises_on_unreachable() -> None:
    client = HomeAssistant("http://x", "tok")
    with patch("voxa.agent.homeassistant.open_url", side_effect=urllib.error.URLError("down")):
        try:
            client.states()
        except HomeAssistantError:
            return
    raise AssertionError("expected HomeAssistantError")


# --- service call URL and JSON body ---

def test_call_posts_to_services_endpoint() -> None:
    client = HomeAssistant("http://base", "tok")
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(b'{"ok": true}')) as mocked:
        client.call("light", "turn_on", "light.kitchen")
        request = mocked.call_args[0][0]
    assert request.full_url == "http://base/api/services/light/turn_on"
    body = json.loads(request.data)
    assert body == {"entity_id": "light.kitchen"}


def test_call_sends_extra_data_and_auth_header() -> None:
    client = HomeAssistant("http://base", "tok")
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(b'{"ok": true}')) as mocked:
        client.call("light", "set_brightness", "light.kitchen", brightness_pct=50)
        request = mocked.call_args[0][0]
    assert json.loads(request.data) == {"entity_id": "light.kitchen", "brightness_pct": 50}
    assert request.headers["Authorization"] == "Bearer tok"


# --- entity matching table ---

def test_entities_builds_entity_records() -> None:
    client = HomeAssistant("http://x", "tok")
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(json.dumps(STATES).encode())):
        ents = client.entities()
    assert ents[0] == Entity("light.kitchen", "Kitchen Light", "light", "on")


def test_match_exact_name() -> None:
    ents = [Entity("light.kitchen", "Kitchen Light", "light", "on")]
    assert match_entity("Kitchen Light", ents).entity_id == "light.kitchen"


def test_match_kitchen_lights_soundalike() -> None:
    ents = [Entity("light.kitchen", "Kitchen Light", "light", "on")]
    assert match_entity("kitchen lights", ents).entity_id == "light.kitchen"


def test_match_word_containment() -> None:
    ents = [Entity("light.living_room", "Living Room Lamp", "light", "off")]
    assert match_entity("living room", ents).entity_id == "light.living_room"


def test_match_domain_filter() -> None:
    ents = [Entity("light.kitchen", "Kitchen Light", "light", "on")]
    assert match_entity("kitchen", ents, domains={"switch"}) is None


def test_match_no_result() -> None:
    ents = [Entity("light.kitchen", "Kitchen Light", "light", "on")]
    assert match_entity("garage", ents) is None


# --- each tool's speech and not-configured message ---

def test_home_turn_not_configured() -> None:
    result = home_turn({"name": "kitchen lights", "state": "on"})
    assert not result.ok
    assert result.speech == NOT_CONFIGURED


def test_home_turn_speech() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "kitchen lights", "state": "off"})
    assert result.ok
    assert result.speech == "Turning Kitchen Light off."


def test_home_set_brightness() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_set({"name": "living room lamp", "value": "50"})
    assert result.ok
    assert result.speech == "Setting Living Room Lamp to 50 percent."


def test_home_scene_speech() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_scene({"name": "movie night"})
    assert result.ok
    assert result.speech == "Running Movie Night."


def test_home_status_speech() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_status({"name": "garage door"})
    assert result.ok
    assert result.speech == "Garage Door is closed."


def test_token_never_in_result() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "kitchen lights", "state": "on"})
    assert "secret-token" not in result.speech
    assert "secret-token" not in result.detail


# --- router table ---

def test_router_turn_on() -> None:
    assert route("turn on the kitchen lights") == ToolCall("home_turn", {"name": "kitchen lights", "state": "on"})


def test_router_turn_suffix() -> None:
    assert route("turn the lights off").args == {"name": "lights", "state": "off"}


def test_router_dim() -> None:
    assert route("dim the lamp to 50 percent").tool == "home_set"


def test_router_thermostat() -> None:
    assert route("set the thermostat to 70").args == {"name": "thermostat", "value": "70"}


def test_router_scene() -> None:
    assert route("run the movie night scene").tool == "home_scene"


def test_router_status() -> None:
    assert route("is the garage door closed").tool == "home_status"


def test_router_computer_not_home() -> None:
    assert route("turn off the computer") is None
