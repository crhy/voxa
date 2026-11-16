from __future__ import annotations

import logging

from voxa.agent.homeassistant import (
    Entity,
    HomeAssistant,
    closest_entities,
    colour_payload,
    domain_services,
    match_entity,
    toggle_services,
)
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.home")

NOT_CONFIGURED = (
    "Home control isn't set up yet. Add your Home Assistant address and token in Preferences."
)

_TURN_DOMAINS = frozenset({"light", "switch", "fan", "plug"})
_COVER_DOMAINS = frozenset({"cover", "garage_door"})
_LOCK_DOMAINS = frozenset({"lock"})
_COLOUR_DOMAINS = frozenset({"light"})
# "turn on the garage door" and "turn the front door off" are still turns: the
# domain decides whether that reads as turning on, opening or locking.
_TURNABLE_DOMAINS = _TURN_DOMAINS | _COVER_DOMAINS | _LOCK_DOMAINS

# How the services read out loud, so a cover, a lock and a light each get
# said in the way a person would say them.
_VERBS = {
    "open_cover": "Opening",
    "close_cover": "Closing",
    "open_garage_door": "Opening",
    "close_garage_door": "Closing",
    "lock": "Locking",
    "unlock": "Unlocking",
    "toggle": "Toggling",
}

# One spoken list per command: the entity list is fetched once and reused for
# the match, the nearest-match suggestions and the reply.
_DEVICE_CAP = 12


def _settings():
    from voxa.config import ConfigStore

    return ConfigStore().load()


def _client():
    """A configured HomeAssistant client, or None when url/token are missing."""
    settings = _settings()
    if not settings.home_assistant_url or not settings.home_assistant_token:
        return None
    return HomeAssistant(settings.home_assistant_url, settings.home_assistant_token)


def _spoken_list(entities: list[Entity], conj: str = "or") -> str:
    names = [entity.name for entity in entities]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" {conj} " + names[-1]


def _not_found(spoken: str, entities: list[Entity], domains: set[str] | None, kind: str) -> str:
    """Say a spoken device name missed, and name the closest ones Voxa can see."""
    near = closest_entities(spoken, entities, domains=domains)
    if not near:
        return f"I couldn't find a {kind} called {spoken}."
    return f"I couldn't find a {kind} called {spoken}. The closest I can see are {_spoken_list(near)}."


def home_turn(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities, domains=_TURNABLE_DOMAINS)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, _TURNABLE_DOMAINS, "light, switch, fan, plug, door or lock"))
    state = args["state"].casefold()
    service = domain_services(entity)["off" if state == "off" else "on"][0]
    client.call(entity.domain, service, entity.entity_id)
    if entity.domain in _TURN_DOMAINS:
        return ToolResult.success(f"Turning {entity.name} {state}.")
    return ToolResult.success(f"{_VERBS[service]} {entity.name}.")


def home_set(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, None, "thing to set"))
    value = args["value"]
    if entity.domain == "light":
        try:
            percent = max(0, min(100, int(float(value))))
        except ValueError:
            return ToolResult.failure(f"I couldn't read a brightness from {value}.")
        client.call("light", "set_brightness", entity.entity_id, brightness_pct=percent)
        return ToolResult.success(f"Setting {entity.name} to {percent} percent.")
    if entity.domain == "climate":
        try:
            temp = float(value)
        except ValueError:
            return ToolResult.failure(f"I couldn't read a temperature from {value}.")
        client.call("climate", "set_temperature", entity.entity_id, temperature=temp)
        return ToolResult.success(f"Setting {entity.name} to {value}.")
    return ToolResult.failure(f"I can't set {entity.name}.")


def home_scene(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities, domains={"scene"})
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, {"scene"}, "scene"))
    client.call("scene", "turn_on", entity.entity_id)
    return ToolResult.success(f"Running {entity.name}.")


def home_status(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, None, "device"))
    state = entity.state or "unknown"
    return ToolResult.success(f"{entity.name} is {state}.", detail=entity.entity_id)


def _move(client: HomeAssistant, entity: Entity, action: str) -> ToolResult:
    """Run an open, close, lock, unlock or toggle action against one entity.

    A cover or lock toggles through its pair: toggle_services picks the one its
    cached state does not already show, so a known device moves with a single
    call.
    """
    if action == "toggle":
        services = toggle_services(entity)
    else:
        services = domain_services(entity).get(action, ())
    if not services:
        return ToolResult.failure(f"I can't {action} {entity.name}.")
    for service in services:
        client.call(entity.domain, service, entity.entity_id)
    if len(services) == 1:
        return ToolResult.success(f"{_VERBS[services[0]]} {entity.name}.", detail=entity.entity_id)
    return ToolResult.success(f"Toggling {entity.name}.", detail=entity.entity_id)


def home_cover(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities, domains=_COVER_DOMAINS)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, _COVER_DOMAINS, "door, gate or cover"))
    return _move(client, entity, args["action"].casefold())


def home_lock(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities, domains=_LOCK_DOMAINS)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, _LOCK_DOMAINS, "lock"))
    return _move(client, entity, args["action"].casefold())


def home_toggle(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, None, "device"))
    return _move(client, entity, "toggle")


def home_colour(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    entity = match_entity(args["name"], entities, domains=_COLOUR_DOMAINS)
    if entity is None:
        return ToolResult.failure(_not_found(args["name"], entities, _COLOUR_DOMAINS, "light"))
    data = colour_payload(args["colour"])
    if not data:
        return ToolResult.failure(f"I couldn't read a colour from {args['colour']}.")
    client.call("light", "turn_on", entity.entity_id, **data)
    return ToolResult.success(f"Setting {entity.name} to {args['colour']}.", detail=entity.entity_id)


def home_devices(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entities = client.entities()
    if not entities:
        return ToolResult.success("I can't see any devices on your Home Assistant.")
    groups: dict[str, list[Entity]] = {}
    for entity in entities:
        groups.setdefault(entity.domain, []).append(entity)
    parts: list[str] = []
    shown = 0
    for domain in sorted(groups):
        take = groups[domain][: _DEVICE_CAP - shown]
        if not take:
            break
        shown += len(take)
        parts.append(f"{domain} {_spoken_list(take, 'and')}")
    speech = f"I can see {len(entities)} devices: " + "; ".join(parts)
    if shown < len(entities):
        speech += f". {len(entities) - shown} more aren't listed."
    return ToolResult.success(speech, detail=", ".join(entity.entity_id for entity in entities))


def home_tools() -> list[Tool]:
    return [
        Tool(
            name="home_turn",
            description="Turn a light, switch, fan or plug on or off.",
            parameters={"name": "the entity name", "state": "on or off"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_turn,
            required=("name", "state"),
        ),
        Tool(
            name="home_set",
            description="Set a light's brightness or a thermostat's temperature.",
            parameters={"name": "the entity name", "value": "the value"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_set,
            required=("name", "value"),
        ),
        Tool(
            name="home_scene",
            description="Run a named Home Assistant scene.",
            parameters={"name": "the scene name"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_scene,
            required=("name",),
        ),
        Tool(
            name="home_status",
            description="Report whether an entity is on, off, open, closed or locked.",
            parameters={"name": "the entity name"},
            risk=RiskLevel.READ_ONLY,
            handler=home_status,
            required=("name",),
        ),
        Tool(
            name="home_cover",
            description="Open, close or toggle a door, gate or cover.",
            parameters={"name": "the entity name", "action": "open, close or toggle"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_cover,
            required=("name", "action"),
        ),
        Tool(
            name="home_lock",
            description="Lock, unlock or toggle a door lock.",
            parameters={"name": "the entity name", "action": "lock, unlock or toggle"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_lock,
            required=("name", "action"),
        ),
        Tool(
            name="home_toggle",
            description="Toggle a light, switch, fan, plug, door or lock.",
            parameters={"name": "the entity name"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_toggle,
            required=("name",),
        ),
        Tool(
            name="home_colour",
            description="Set a light's colour by name, warm or cool, or by hue.",
            parameters={"name": "the entity name", "colour": "the colour"},
            risk=RiskLevel.REVERSIBLE,
            handler=home_colour,
            required=("name", "colour"),
        ),
        Tool(
            name="home_devices",
            description="List the devices Voxa can see at home, grouped by type.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=home_devices,
            required=(),
        ),
    ]
