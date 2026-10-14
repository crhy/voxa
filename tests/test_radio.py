import json

from voxa.agent import intents, radio
from voxa.agent.tools import radio as radio_tools


def test_catalogue_and_find():
    built = radio.catalogue()
    assert len(built) == 16 and built[0].name == "Groove Salad"
    assert radio.find("groove salad", built).name == "Groove Salad"
    assert radio.find("some ska radio", built).name == "Bob's Ska Radio"
    assert radio.find("bobs ska", built).name == "Bob's Ska Radio"
    assert radio.find("cowpoke radio", built).name == "Cowpoke Radio"
    assert radio.find("country", built).name in ("Boot Liquor", "Cowpoke Radio")
    assert radio.find("", built).name == "Groove Salad"
    assert radio.find("polka", built) is None
    assert radio.is_station("groove salad") and not radio.is_station("dirty old town by the pogues")
    assert radio.spoken_list(built).startswith("I have 16 stations: Groove Salad, ")
    assert radio.describe(built[0]).startswith("This is Groove Salad: ambient, electronic, chill.")


def test_soma_list_is_used_when_reachable():
    text = json.dumps({"channels": [
        {"id": "a", "title": "Quiet One", "genre": "ambient", "description": "d", "listeners": "5"},
        {"id": "b", "title": "Loud One", "genre": "rock|metal", "description": "d", "listeners": "900"},
    ]})
    stations = radio.catalogue(lambda url: text)
    assert [s.name for s in stations] == [
        "Loud One", "Quiet One", "Illinois Street Lounge", "Bossa Beyond", "Christmas Lounge",
        "Bob's Ska Radio", "Cowpoke Radio", "HPR-2 Classical",
    ]
    assert stations[0].stream == "https://ice1.somafm.com/b-128-mp3" and stations[0].genre == "rock metal"
    assert radio.catalogue(lambda url: "not json")[0].name == "Groove Salad"

    def broken(url):
        raise OSError("offline")

    assert len(radio.catalogue(broken)) == 16


def test_permanent_soma_favourites_are_not_duplicated_in_live_catalogue():
    text = json.dumps({"channels": [
        {"id": "bossa", "title": "Bossa Beyond", "genre": "bossa", "description": "live", "listeners": "9"},
        {"id": "other", "title": "Other", "genre": "other", "description": "d", "listeners": "8"},
    ]})
    stations = radio.catalogue(lambda url: text)
    assert [s.name for s in stations].count("Bossa Beyond") == 1
    assert "Christmas Lounge" in [s.name for s in stations]


def test_tools(monkeypatch):
    played = []

    class FakeVlc:
        def play(self, media, fullscreen=False):
            played.append(media)

    monkeypatch.setattr(radio_tools, "VlcPlayer", FakeVlc)
    monkeypatch.setattr(radio_tools, "_stations", radio.catalogue())
    monkeypatch.setattr(radio_tools, "now_playing", None)
    assert not radio_tools.radio_genre({}).ok
    result = radio_tools.play_radio({"name": "ska"})
    assert result.ok and result.speech == "Playing Bob's Ska Radio."
    assert played[0].audio_url == "http://51.255.235.165:5528/stream" and played[0].video_url is None
    assert radio_tools.radio_genre({}).speech.startswith("This is Bob's Ska Radio")
    assert "16 stations" in radio_tools.list_radio({}).speech
    missing = radio_tools.play_radio({"name": "polka"})
    assert missing.ok and "polka" in missing.speech and "Which station" in missing.speech


def test_router():
    assert intents.route("play groove salad").tool == "play_radio"
    assert intents.route("play some ska radio").args == {"name": "ska"}
    assert intents.route("put on the radio").args == {"name": ""}
    assert intents.route("play cowpoke radio").args == {"name": "cowpoke"}
    assert intents.route("what stations do you have").tool == "list_radio"
    assert intents.route("what station is this").tool == "radio_genre"
    other = intents.route("play Dirty Old Town by the Pogues")
    assert other is not None and other.tool != "play_radio"
