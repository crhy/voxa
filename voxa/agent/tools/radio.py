from __future__ import annotations

import urllib.request

from voxa.agent import radio
from voxa.agent.player import VlcPlayer
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.ytdlp import Media

_stations: list[radio.Station] | None = None
now_playing: radio.Station | None = None


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Voxa"})
    with urllib.request.urlopen(request, timeout=6) as response:  # noqa: S310 - fixed https address
        return response.read().decode("utf-8", "replace")


def stations() -> list[radio.Station]:
    global _stations
    if _stations is None:
        _stations = radio.catalogue(_fetch)
    return _stations


def play_radio(args: dict[str, str]) -> ToolResult:
    global now_playing
    wanted = args.get("name", "")
    station = radio.find(wanted, stations())
    if station is None:
        return ToolResult.failure(f"I do not have a station for {radio.clean_request(wanted)}. {radio.spoken_list(stations())}")
    try:
        VlcPlayer().play(Media(station.name, "radio", 0, station.stream, None, station.stream, False))
    except Exception as exc:  # noqa: BLE001 - the player is outside our control
        return ToolResult.failure("I could not start the radio.", detail=str(exc))
    now_playing = station
    return ToolResult.success(f"Playing {station.name}.", detail=station.stream)


def list_radio(args: dict[str, str]) -> ToolResult:
    return ToolResult.success(radio.spoken_list(stations()))


def radio_genre(args: dict[str, str]) -> ToolResult:
    if now_playing is None:
        return ToolResult.failure("No radio station is playing.")
    return ToolResult.success(radio.describe(now_playing))


def radio_tools() -> list[Tool]:
    return [
        Tool(name="play_radio", description="Play an internet radio station by name or genre.",
             parameters={"name": "station name or genre; empty for the most popular"},
             risk=RiskLevel.REVERSIBLE, handler=play_radio),
        Tool(name="list_radio", description="Name the radio stations Voxa knows.", parameters={},
             risk=RiskLevel.READ_ONLY, handler=list_radio),
        Tool(name="radio_genre", description="Say which station is playing and its genre.", parameters={},
             risk=RiskLevel.READ_ONLY, handler=radio_genre),
    ]
