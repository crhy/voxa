import pytest

from voxa.agent import livetv, radio
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import browser, default_registry, local
from voxa.agent.tools import livetv as livetv_tools

ROUTES = [
    ("open live TV", ToolCall("play_tv", {"name": ""})),
    ("what channels do you have", ToolCall("list_tv", {})),
    ("watch NASA TV", ToolCall("play_tv", {"name": "NASA TV"})),
    ("put on France 24", ToolCall("play_tv", {"name": "France 24"})),
    ("watch Al Jazeera", ToolCall("play_tv", {"name": "Al Jazeera English"})),
    ("watch Russia Today", ToolCall("play_tv", {"name": "RT News"})),
    ("watch classic movies", ToolCall("play_tv", {"name": "RetroStrange TV"})),
    ("put on Bloomberg TV", ToolCall("play_tv", {"name": "Bloomberg Television"})),
    ("watch NHK World", ToolCall("play_tv", {"name": "NHK World-Japan"})),
    ("watch a video about cats", ToolCall("play_video", {"query": "a video about cats"})),
    ("play the local news", ToolCall("local_news", {})),
    ("show the local weather", ToolCall("local_weather", {})),
    ("what's the weather in Paris", ToolCall("local_weather", {"place": "Paris"})),
    ("live music tonight in my area", ToolCall("local_events", {"what": "live music", "when": "tonight"})),
    ("concerts near me this weekend", ToolCall("local_events", {"what": "concerts", "when": "this weekend"})),
    ("directions from Portland to Seattle", ToolCall("directions", {"origin": "Portland", "destination": "Seattle"})),
    ("navigate from the airport to downtown", ToolCall("directions", {"origin": "airport", "destination": "downtown"})),
    ("show me the way to the beach", ToolCall("directions", {"destination": "beach"})),
    ("route to Denver", ToolCall("directions", {"destination": "Denver"})),
    ("directions to the public library", ToolCall("directions", {"destination": "the public library"})),
    ("navigate to github.com", ToolCall("open_url", {"url": "github.com"})),
    ("play classical radio", ToolCall("play_radio", {"name": "classical"})),
]


@pytest.mark.parametrize(("said", "expected"), ROUTES)
def test_routes(said, expected):
    assert route(said) == expected


def test_every_new_tool_is_registered():
    names = set(default_registry().names())
    assert {"play_tv", "list_tv", "local_weather", "local_news", "local_events"} <= names


def test_classical_and_lounge_stations_are_always_offered():
    names = [station.name for station in radio.catalogue(lambda _url: "{}")]
    assert "HPR-2 Classical" in names and "Illinois Street Lounge" in names
    assert radio.find("classical", list(radio.BUILTIN)).name == "HPR-2 Classical"
    assert radio.find("ska", list(radio.BUILTIN)).name == "Bob's Ska Radio"


def test_tv_channel_lookup():
    assert livetv.find("the NASA TV channel").name == "NASA TV"
    assert livetv.find("deutsche welle").name == "DW News"
    assert livetv.find("Russia Today").name == "RT News"
    assert livetv.find("classic movie channel").name == "RetroStrange TV"
    assert livetv.find("Bloomberg TV").name == "Bloomberg Television"
    assert livetv.find("NHK World").name == "NHK World-Japan"
    assert livetv.find("cartoon network") is None
    assert not livetv.is_channel("a video about cats")


def test_play_tv_asks_when_no_channel_is_named(monkeypatch):
    played = []
    monkeypatch.setattr(livetv_tools.VlcPlayer, "play", lambda _self, media: played.append(media))
    asked = livetv_tools.play_tv({"name": ""})
    assert asked.speech == "Which channel would you like?" and asked.detail.startswith(livetv.ASK_TV)
    result = livetv_tools.play_tv({"name": "NASA TV"})
    assert result.ok and result.speech == "Playing NASA TV."
    assert played[0].video_url.endswith(".m3u8")


@pytest.fixture
def opened(monkeypatch):
    urls = []
    monkeypatch.setattr(local.host, "spawn", lambda command: urls.append(command[-1]) or True)
    monkeypatch.setattr(browser.host, "spawn", lambda command: urls.append(command[-1]) or True)
    monkeypatch.setattr(local, "locate", lambda: "Portland, Oregon")
    monkeypatch.setattr(local, "fetch", lambda _url: "not json")
    return urls


def test_local_tools_use_the_users_place(opened):
    assert local.local_weather({}).speech == "Here is the weather for Portland."
    assert local.local_news({}).speech == "Here is the local news for Portland."
    assert local.local_events({"what": "live music", "when": "tonight"}).speech == "Here is live music tonight near Portland."
    assert "weather%20Portland" in opened[0] and "Portland%20local%20news%20live" in opened[1]
    assert local.local_weather({"place": "Tokyo"}).detail.endswith("weather%20Tokyo")


def test_local_tools_say_so_when_the_place_is_unknown(opened, monkeypatch):
    monkeypatch.setattr(local, "locate", lambda: "")
    for tool in (local.local_weather, local.local_news, local.local_events):
        result = tool({})
        assert not result.ok and result.speech == local.NO_LOCATION
    assert opened == []


def test_directions_with_a_starting_point(opened):
    result = browser.directions({"origin": "Portland", "destination": "Seattle"})
    assert result.speech == "Here are directions from Portland to Seattle."
    assert "origin=Portland" in result.detail and "destination=Seattle" in result.detail
