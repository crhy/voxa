from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher

from voxa import apps

__all__ = [
    "Entity",
    "HomeAssistant",
    "HomeAssistantError",
    "closest_entities",
    "toggle_services",
    "colour_payload",
    "domain_services",
    "match_entity",
    "open_url",
]


class HomeAssistantError(RuntimeError):
    pass


# Home Assistant usually runs on a LAN address, but a loopback server must
# never be routed through an HTTP proxy either, so we keep a proxy-free opener
# for the loopback case exactly like the Ollama client does.
_local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def open_url(request: urllib.request.Request, timeout: float):
    """Open ``request``, bypassing any proxy for loopback URLs."""
    url = getattr(request, "full_url", str(request))
    host = urllib.parse.urlparse(url).hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"} or host.startswith("127."):
        return _local_opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    name: str
    domain: str
    state: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


_LEADING_WORDS = re.compile(r"^(?:the|a|an|some|my)\s+", re.IGNORECASE)


class HomeAssistant:
    def __init__(self, url: str = "", token: str = "", timeout: float = 4.0) -> None:
        self.base_url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def states(self) -> list[dict]:
        """GET /api/states: the full state list, one dict per entity."""
        request = urllib.request.Request(f"{self.base_url}/api/states", method="GET", headers=self._headers())
        try:
            with open_url(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise HomeAssistantError(f"Could not reach Home Assistant: {exc}") from exc
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def call(self, domain: str, service: str, entity_id: str, **data) -> dict:
        """POST /api/services/<domain>/<service> with the entity and extra data."""
        body = {"entity_id": entity_id}
        body.update(data)
        payload = json.dumps(body).encode("utf-8")
        endpoint = f"{self.base_url}/api/services/{domain}/{service}"
        request = urllib.request.Request(endpoint, data=payload, headers=self._headers(), method="POST")
        try:
            with open_url(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise HomeAssistantError(f"Home Assistant call failed: {exc}") from exc
        return result if isinstance(result, dict) else {}

    def entities(self) -> list[Entity]:
        """The configured entities as Entity records, skipping malformed entries."""
        out: list[Entity] = []
        for item in self.states():
            entity_id = str(item.get("entity_id", ""))
            if not entity_id:
                continue
            domain = entity_id.split(".", 1)[0]
            name = str(item.get("name", "")) or entity_id.replace(".", " ")
            state = item.get("state")
            out.append(Entity(entity_id=entity_id, name=name, domain=domain, state=str(state) if state is not None else ""))
        return out


# A spoken colour reaches Home Assistant either as a CSS colour name or as a
# hue/saturation pair; these are the names people actually say to a light.
_COLOUR_NAMES = frozenset(
    {
        "red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple",
        "pink", "magenta", "brown", "black", "white", "gray", "grey", "silver",
        "gold", "cyan", "teal", "turquoise", "beige", "cream", "ivory", "charcoal",
        "maroon", "navy", "olive", "lime", "khaki", "lavender", "coral", "salmon",
        "crimson", "scarlet", "amber", "peach", "mint", "rose", "sand", "aqua",
    }
)
_WARM_COLOURS = frozenset({"warm", "warm white", "warm light", "soft white", "candle"})
_COOL_COLOURS = frozenset({"cool", "cool white", "cool light", "daylight", "day light"})
_HUE = re.compile(r"(?:hue\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|°)?", re.IGNORECASE)


def _ranked(spoken: str, entities: list[Entity], domains: set[str] | None = None, floor: float = 0.82) -> list[tuple[float, Entity]]:
    """Score every entity against a spoken phrase, best first.

    An exact friendly-name or entity_id scores 1.0, word containment 0.9, and
    anything else is judged by how close its phonetic_key is to the phrase's.
    A candidate is returned only when it reaches ``floor``.
    """
    wanted = _LEADING_WORDS.sub("", _normalize(spoken), count=1)
    if not wanted:
        return []
    pool = [e for e in entities if domains is None or e.domain in domains]
    if not pool:
        return []

    wanted_words = set(wanted.split())
    qkey = apps.phonetic_key(wanted)
    scored: list[tuple[float, Entity]] = []
    for entity in pool:
        id_words = entity.entity_id.replace(".", " ")
        if _normalize(entity.name) == wanted or _normalize(entity.entity_id) == wanted:
            scored.append((1.0, entity))
            continue
        name_words = set(_normalize(entity.name).split())
        id_set = set(_normalize(id_words).split())
        if (wanted_words <= name_words or name_words <= wanted_words
                or wanted_words <= id_set or id_set <= wanted_words):
            scored.append((0.9, entity))
            continue
        if len(qkey) < 2:
            continue
        score = max(
            SequenceMatcher(None, qkey, apps.phonetic_key(candidate)).ratio()
            for candidate in (entity.name, id_words)
        )
        if score >= floor:
            scored.append((score, entity))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def match_entity(spoken: str, entities: list[Entity], domains: set[str] | None = None) -> Entity | None:
    """Find the entity a spoken phrase refers to, or None.

    Tries an exact friendly-name match, then word containment, then a
    phonetic_key similarity of at least 0.82 (so "kitchen lights" still
    matches "Kitchen Light" / "light.kitchen").
    """
    ranked = _ranked(spoken, entities, domains)
    return ranked[0][1] if ranked else None


def closest_entities(spoken: str, entities: list[Entity], limit: int = 3, domains: set[str] | None = None) -> list[Entity]:
    """The ``limit`` entities that sound nearest to a spoken phrase.

    The same scoring as match_entity with a 0.5 floor: when a spoken name misses
    outright the assistant has to name the devices it can see that sound closest,
    and 0.5 is what brings "kitcen lites" back to "Kitchen Light" without dragging
    unrelated devices into the sentence.
    """
    return [entity for _, entity in _ranked(spoken, entities, domains, floor=0.5)][:limit]


def domain_services(entity: Entity) -> dict[str, tuple[str, ...]]:
    """Which Home Assistant services carry out each action for this entity.

    A cover or garage door has no single toggle service, so its toggle is the
    pair open_cover/close_cover (open_garage_door/close_garage_door for a
    garage_door domain) and the caller picks the one its state does not already
    have. A lock's "on" is locking it; a light, switch, fan or plug turns on,
    off and toggles.
    """
    if entity.domain in {"cover", "garage_door"}:
        stem = "garage_door" if entity.domain == "garage_door" else "cover"
        open_service = f"open_{stem}"
        close_service = f"close_{stem}"
        return {
            "open": (open_service,),
            "close": (close_service,),
            "on": (open_service,),
            "off": (close_service,),
            "toggle": (open_service, close_service),
        }
    if entity.domain == "lock":
        return {
            "lock": ("lock",),
            "unlock": ("unlock",),
            "on": ("lock",),
            "off": ("unlock",),
            "toggle": ("lock", "unlock"),
        }
    return {"on": ("turn_on",), "off": ("turn_off",), "toggle": ("toggle",)}


def toggle_services(entity: Entity) -> tuple[str, ...]:
    """The service calls that toggle this entity, given the state Voxa holds.

    Covers and locks toggle through their pair: the one that moves them is the
    one their cached state is not already showing. With no state to go on, both
    are called and the hub decides.
    """
    services = domain_services(entity)["toggle"]
    if len(services) == 2:
        state = entity.state.casefold()
        if state in {"open", "locked", "on"}:
            return services[1:]
        if state in {"closed", "unlocked", "off"}:
            return services[:1]
    return services


def colour_payload(spoken: str) -> dict[str, object]:
    """The Home Assistant data for a spoken colour, or {} when it is unreadable.

    A colour Home Assistant names goes through color_name; warm and cool go
    through hs_color, as does a hue in degrees, which is what a colour slider
    reports.
    """
    wanted = _normalize(spoken)
    if wanted in _COLOUR_NAMES:
        return {"color_name": wanted}
    if wanted in _WARM_COLOURS:
        return {"hs_color": [30.0, 40.0]}
    if wanted in _COOL_COLOURS:
        return {"hs_color": [210.0, 30.0]}
    for name in sorted(_COLOUR_NAMES, key=len, reverse=True):
        if re.search(rf"\b{name}\b", wanted):
            return {"color_name": name}
    hue_match = _HUE.fullmatch(wanted)
    if hue_match:
        degrees = float(hue_match.group(1))
        if 0.0 <= degrees <= 360.0:
            return {"hs_color": [degrees, 100.0]}
    return {}
