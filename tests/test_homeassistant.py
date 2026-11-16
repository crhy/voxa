from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

from voxa.agent.homeassistant import (
    Entity,
    HomeAssistant,
    HomeAssistantError,
    closest_entities,
    colour_payload,
    domain_services,
    match_entity,
)
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.home import (
    NOT_CONFIGURED,
    home_cover,
    home_colour,
    home_devices,
    home_lock,
    home_scene,
    home_set,
    home_status,
    home_toggle,
    home_turn,
)


class FakeResponse(io.BytesIO):
    pass


STATES = [
    {"entity_id": "light.kitchen", "name": "Kitchen Light", "state": "on"},
    {"entity_id": "light.living_room", "name": "Living Room Lamp", "state": "off"},
    {"entity_id": "light.bedroom", "name": "Bedroom Light", "state": "on", "color_name": "pink"},
    {"entity_id": "switch.fan", "name": "Ceiling Fan", "state": "on"},
    {"entity_id": "climate.thermostat", "name": "Thermostat", "state": "68"},
    {"entity_id": "scene.movie_night", "name": "Movie Night", "state": "unknown"},
    {"entity_id": "garage_door.door", "name": "Garage Door", "state": "closed"},
    {"entity_id": "cover.blinds", "name": "Bedroom Blinds", "state": "unknown"},
    {"entity_id": "lock.front_door", "name": "Front Door", "state": "locked"},
]

REQUESTS: list[tuple[str, dict]] = []


def fake_open(request, timeout=None):
    url = getattr(request, "full_url", str(request))
    REQUESTS.append((url, json.loads(request.data) if request.data else {}))
    if "/api/states" in url:
        return FakeResponse(json.dumps(STATES).encode())
    return FakeResponse(b'{"ok": true}')


def configured():
    """Patch config so url/token are set and the opener returns STATES."""
    REQUESTS.clear()
    return patch("voxa.agent.homeassistant.open_url", side_effect=fake_open)


def posts() -> list[tuple[str, dict]]:
    """The service calls a command made: which service, and with what body."""
    return [(url.split("/api/services/")[-1], body) for url, body in REQUESTS if "/api/services/" in url]


def state_fetches() -> int:
    return sum(1 for url, _ in REQUESTS if "/api/states" in url)


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


# --- covers, locks, colours and the device list ---

def test_domain_services_maps_covers_and_locks() -> None:
    cover = domain_services(Entity("cover.blinds", "Blinds", "cover", "closed"))
    assert cover["open"] == ("open_cover",) and cover["close"] == ("close_cover",)
    garage = domain_services(Entity("garage_door.door", "Garage", "garage_door", "closed"))
    assert garage["open"] == ("open_garage_door",)
    lock = domain_services(Entity("lock.front_door", "Front Door", "lock", "locked"))
    assert lock["on"] == ("lock",) and lock["off"] == ("unlock",)


def test_colour_payload_reads_names_and_hues() -> None:
    assert colour_payload("pink") == {"color_name": "pink"}
    assert colour_payload("the warm white one") == {"color_name": "white"}
    assert colour_payload("warm") == {"hs_color": [30.0, 40.0]}
    assert colour_payload("200 degrees") == {"hs_color": [200.0, 100.0]}
    assert colour_payload("splendid") == {}


def test_closest_entities_ranks_by_sound() -> None:
    ents = [
        Entity("switch.fan", "Ceiling Fan", "switch", "on"),
        Entity("light.kitchen", "Kitchen Light", "light", "on"),
    ]
    assert [e.name for e in closest_entities("kitcen lites", ents)] == ["Kitchen Light"]


def test_closest_entities_caps_and_ignores_unrelated() -> None:
    lamps = [Entity(f"light.lamp{i}", f"Lamp {i}", "light", "on") for i in range(6)]
    assert [e.name for e in closest_entities("lamp", lamps)] == ["Lamp 0", "Lamp 1", "Lamp 2"]
    assert closest_entities("zebra", lamps) == []


# --- each new action's exact POST body ---

def test_home_cover_open_posts_the_garage_service() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "garage door", "action": "open"})
    assert result.ok
    assert result.speech == "Opening Garage Door."
    assert posts() == [("garage_door/open_garage_door", {"entity_id": "garage_door.door"})]


def test_home_cover_close_posts_the_cover_service() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "bedroom blinds", "action": "close"})
    assert result.ok
    assert result.speech == "Closing Bedroom Blinds."
    assert posts() == [("cover/close_cover", {"entity_id": "cover.blinds"})]


def test_home_cover_toggle_moves_from_the_cached_state() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "garage door", "action": "toggle"})
    assert result.speech == "Opening Garage Door."
    assert posts() == [("garage_door/open_garage_door", {"entity_id": "garage_door.door"})]


def test_home_cover_toggle_without_a_state_calls_both_services() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "bedroom blinds", "action": "toggle"})
    assert result.speech == "Toggling Bedroom Blinds."
    assert posts() == [
        ("cover/open_cover", {"entity_id": "cover.blinds"}),
        ("cover/close_cover", {"entity_id": "cover.blinds"}),
    ]


def test_home_lock_posts_lock() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_lock({"name": "front door", "action": "lock"})
    assert result.ok
    assert result.speech == "Locking Front Door."
    assert posts() == [("lock/lock", {"entity_id": "lock.front_door"})]


def test_home_lock_posts_unlock() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_lock({"name": "front door", "action": "unlock"})
    assert result.ok
    assert result.speech == "Unlocking Front Door."
    assert posts() == [("lock/unlock", {"entity_id": "lock.front_door"})]


def test_home_turn_on_a_cover_opens_it() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "garage door", "state": "on"})
    assert result.ok
    assert result.speech == "Opening Garage Door."
    assert posts() == [("garage_door/open_garage_door", {"entity_id": "garage_door.door"})]


def test_home_turn_off_a_lock_unlocks_it() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "front door", "state": "off"})
    assert result.ok
    assert result.speech == "Unlocking Front Door."
    assert posts() == [("lock/unlock", {"entity_id": "lock.front_door"})]


def test_home_toggle_posts_the_domain_toggle() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_toggle({"name": "ceiling fan"})
    assert result.ok
    assert result.speech == "Toggling Ceiling Fan."
    assert posts() == [("switch/toggle", {"entity_id": "switch.fan"})]


def test_home_colour_posts_color_name() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_colour({"name": "bedroom light", "colour": "pink"})
    assert result.ok
    assert result.speech == "Setting Bedroom Light to pink."
    assert posts() == [("light/turn_on", {"entity_id": "light.bedroom", "color_name": "pink"})]


def test_home_colour_posts_hs_color_for_warm() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_colour({"name": "kitchen light", "colour": "warm white"})
    assert result.ok
    assert posts() == [("light/turn_on", {"entity_id": "light.kitchen", "hs_color": [30.0, 40.0]})]


def test_home_colour_refuses_an_unreadable_colour() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_colour({"name": "bedroom light", "colour": "splendid"})
    assert not result.ok
    assert result.speech == "I couldn't read a colour from splendid."
    assert posts() == []


# --- a spoken name that misses says what Voxa can see ---

def test_misspoken_light_names_the_closest_device() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "kitcen lites", "state": "off"})
    assert not result.ok
    assert "closest I can see are Kitchen Light" in result.speech
    assert posts() == []


def test_misspoken_lock_names_the_closest_device() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_lock({"name": "the laundry door", "action": "lock"})
    assert not result.ok
    assert "closest I can see are Front Door" in result.speech


def test_a_cover_miss_names_the_nearest_cover() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "kithen blindz", "action": "open"})
    assert not result.ok
    assert result.speech == (
        "I couldn't find a door, gate or cover called kithen blindz."
        " The closest I can see are Bedroom Blinds."
    )


def test_a_miss_with_nothing_nearby_says_so() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_cover({"name": "kitchen light", "action": "open"})
    assert not result.ok
    assert result.speech == "I couldn't find a door, gate or cover called kitchen light."


def test_soundalike_name_still_resolves() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_turn({"name": "kithen lites", "state": "off"})
    assert result.ok
    assert result.speech == "Turning Kitchen Light off."


# --- the device list, and one state fetch per command ---

def test_home_devices_groups_by_domain() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_devices({})
    assert result.ok
    assert result.speech.startswith("I can see 9 devices:")
    assert "light Kitchen Light" in result.speech
    assert "garage_door Garage Door" in result.speech
    assert "lock Front Door" in result.speech
    assert state_fetches() == 1


def test_home_devices_caps_the_spoken_list() -> None:
    many = [{"entity_id": f"light.lamp{i}", "name": f"Lamp {i}", "state": "on"} for i in range(20)]
    with patch("voxa.agent.homeassistant.open_url", return_value=FakeResponse(json.dumps(many).encode())), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_devices({})
    assert "Lamp 11" in result.speech
    assert "Lamp 12" not in result.speech
    assert "8 more aren't listed." in result.speech


def test_one_state_fetch_per_command() -> None:
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        for command in (
            home_cover({"name": "garage door", "action": "open"}),
            home_lock({"name": "front door", "action": "lock"}),
            home_colour({"name": "bedroom light", "colour": "pink"}),
            home_toggle({"name": "ceiling fan"}),
            home_devices({}),
        ):
            assert command.ok
    assert state_fetches() == 5


def test_garage_door_question_answers_from_the_cached_state() -> None:
    assert route("is the garage door open") == ToolCall("home_status", {"name": "garage door"})
    with configured(), patch("voxa.config.ConfigStore") as store:
        store.return_value.load.return_value = FakeSettings
        result = home_status({"name": "garage door"})
    assert result.speech == "Garage Door is closed."
    assert posts() == []


# --- the new router rules ---

def test_router_open_cover() -> None:
    assert route("open the garage door") == ToolCall("home_cover", {"name": "garage door", "action": "open"})


def test_router_close_cover() -> None:
    assert route("close the blinds") == ToolCall("home_cover", {"name": "blinds", "action": "close"})


def test_router_lock_and_unlock() -> None:
    assert route("lock the front door") == ToolCall("home_lock", {"name": "front door", "action": "lock"})
    assert route("unlock the door") == ToolCall("home_lock", {"name": "door", "action": "unlock"})


def test_router_colour() -> None:
    assert route("set the kitchen lights to pink") == ToolCall("home_colour", {"name": "kitchen lights", "colour": "pink"})


def test_router_toggle() -> None:
    assert route("toggle the ceiling fan") == ToolCall("home_toggle", {"name": "ceiling fan"})


def test_router_home_devices() -> None:
    assert route("list my home devices") == ToolCall("home_devices", {})


def test_router_open_an_app_is_still_an_app() -> None:
    assert route("open the browser").tool == "open_app"
    assert route("close the text editor").tool == "close_app"


def test_router_lock_the_screen_is_still_the_screen() -> None:
    assert route("lock the screen").tool == "lock_screen"


def test_router_a_colour_word_on_another_rule_is_not_home() -> None:
    assert route("change the character to grace") == ToolCall("change_character", {"name": "grace"})
    assert route("set the thermostat to 70").tool == "home_set"
