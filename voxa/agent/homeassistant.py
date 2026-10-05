from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher

from voxa import apps

__all__ = ["Entity", "HomeAssistant", "HomeAssistantError", "match_entity", "open_url"]


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


def match_entity(spoken: str, entities: list[Entity], domains: set[str] | None = None) -> Entity | None:
    """Find the entity a spoken phrase refers to, or None.

    Tries an exact friendly-name match, then word containment, then a
    phonetic_key similarity of at least 0.82 (so "kitchen lights" still
    matches "Kitchen Light" / "light.kitchen").
    """
    wanted = _LEADING_WORDS.sub("", _normalize(spoken), count=1)
    if not wanted:
        return None
    pool = [e for e in entities if domains is None or e.domain in domains]
    if not pool:
        return None

    for entity in pool:
        if _normalize(entity.name) == wanted or _normalize(entity.entity_id) == wanted:
            return entity

    for entity in pool:
        name_words = set(_normalize(entity.name).split())
        id_words = set(_normalize(entity.entity_id.replace(".", " ")).split())
        wanted_words = set(wanted.split())
        if wanted_words <= name_words or name_words <= wanted_words:
            return entity
        if wanted_words <= id_words or id_words <= wanted_words:
            return entity

    qkey = apps.phonetic_key(wanted)
    if len(qkey) < 2:
        return None
    best: tuple[float, Entity] | None = None
    for entity in pool:
        for candidate in (entity.name, entity.entity_id.replace(".", " ")):
            score = SequenceMatcher(None, qkey, apps.phonetic_key(candidate)).ratio()
            if score >= 0.82 and (best is None or score > best[0]):
                best = (score, entity)
    return best[1] if best is not None else None
