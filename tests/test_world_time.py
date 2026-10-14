from __future__ import annotations

from datetime import UTC, datetime

from voxa.agent import intents, worldtime
from voxa.agent.tools import default_registry

NOW = datetime(2026, 10, 10, 18, 41, tzinfo=UTC)


def test_zone_for_known():
    assert worldtime.zone_for("tokyo") == "Asia/Tokyo"
    assert worldtime.zone_for("  New York ") == "America/New_York"


def test_zone_for_the_prefix():
    assert worldtime.zone_for("the UK") == "Europe/London"


def test_zone_for_unknown():
    assert worldtime.zone_for("atlantis") is None


def test_time_in_tokyo():
    assert worldtime.time_in("tokyo", NOW) == (3, 41, "Sunday")


def test_spoken_time_in_tokyo():
    spoken = worldtime.spoken_time_in("tokyo", NOW, "Saturday")
    assert spoken == "In Tokyo it's 3:41 in the morning on Sunday."


def test_spoken_difference_tokyo():
    assert worldtime.spoken_difference("tokyo", NOW, -7.0) == "Tokyo is 16 hours ahead of you."


def test_spoken_difference_honolulu():
    assert worldtime.spoken_difference("honolulu", NOW, -7.0) == "Honolulu is 3 hours behind you."


def test_spoken_difference_same_time():
    assert worldtime.spoken_difference("seattle", NOW, -7.0) == "Seattle is on the same time as you."


def test_route_world_time():
    call = intents.route("what time is it in Tokyo")
    assert call.tool == "world_time"
    assert call.args == {"place": "Tokyo"}


def test_route_time_difference():
    call = intents.route("what's the time difference between here and London")
    assert call.tool == "time_difference"
    assert call.args == {"place": "London"}


def test_route_plain_time_still_tell_time():
    assert intents.route("what time is it").tool == "tell_time"


def test_route_kitchen_not_world_time():
    call = intents.route("what time is it in the kitchen")
    assert call is None or call.tool != "world_time"


def test_handler_world_time_ok():
    result = default_registry().call("world_time", {"place": "tokyo"})
    assert result.ok


def test_handler_time_difference_ok():
    result = default_registry().call("time_difference", {"place": "tokyo"})
    assert result.ok


def test_handler_unknown_place():
    result = default_registry().call("world_time", {"place": "atlantis"})
    assert not result.ok
    assert result.speech == "I don't know the time zone for atlantis."


def test_registry_has_both():
    names = default_registry().names()
    assert "world_time" in names
    assert "time_difference" in names
