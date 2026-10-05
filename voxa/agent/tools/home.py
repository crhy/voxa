from __future__ import annotations

import logging

from voxa.agent.homeassistant import Entity, HomeAssistant, match_entity
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.home")

NOT_CONFIGURED = (
    "Home control isn't set up yet. Add your Home Assistant address and token in Preferences."
)

_TURN_DOMAINS = frozenset({"light", "switch", "fan", "plug"})


def _settings():
    from voxa.config import ConfigStore

    return ConfigStore().load()


def _client():
    """A configured HomeAssistant client, or None when url/token are missing."""
    settings = _settings()
    if not settings.home_assistant_url or not settings.home_assistant_token:
        return None
    return HomeAssistant(settings.home_assistant_url, settings.home_assistant_token)


def home_turn(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entity = match_entity(args["name"], client.entities(), domains=_TURN_DOMAINS)
    if entity is None:
        return ToolResult.failure(f"I couldn't find a light, switch, fan or plug called {args['name']}.")
    state = args["state"].casefold()
    if state == "off":
        client.call(entity.domain, "turn_off", entity.entity_id)
        return ToolResult.success(f"Turning {entity.name} off.")
    client.call(entity.domain, "turn_on", entity.entity_id)
    return ToolResult.success(f"Turning {entity.name} on.")


def home_set(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entity = match_entity(args["name"], client.entities())
    if entity is None:
        return ToolResult.failure(f"I couldn't find something to set called {args['name']}.")
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
    wanted = args["name"].casefold()
    for entity in client.entities():
        if entity.domain == "scene" and _name_matches(entity, wanted):
            client.call("scene", "turn_on", entity.entity_id)
            return ToolResult.success(f"Running {entity.name}.")
    return ToolResult.failure(f"I couldn't find a scene called {args['name']}.")


def home_status(args: dict[str, str]) -> ToolResult:
    client = _client()
    if client is None:
        return ToolResult.failure(NOT_CONFIGURED)
    entity = match_entity(args["name"], client.entities())
    if entity is None:
        return ToolResult.failure(f"I couldn't find {args['name']}.")
    state = entity.state or "unknown"
    return ToolResult.success(f"{entity.name} is {state}.", detail=entity.entity_id)


def _name_matches(entity: Entity, wanted: str) -> bool:
    return match_entity(wanted, [entity]) is not None


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
    ]
