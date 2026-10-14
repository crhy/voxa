from voxa.agent import intents, lists, radio
from voxa.agent.tools import radio as radio_tools


def test_list_done_accepts_is_done():
    call = intents.route("Mark call the vet as done")
    assert call is not None and call.tool == "list_done" and call.args["item"] == "call the vet"
    call = intents.route("Mark call the vet is done")
    assert call is not None and call.tool == "list_done" and call.args["item"] == "call the vet"


def test_find_drops_filler_words():
    items = [(False, "Call the vet")]
    assert lists.find(items, "called a vet") == 0
    assert lists.find(items, "milk") is None


def test_ska_mishearing_routes_to_play_radio():
    call = intents.route("Play Scott Radio")
    assert call is not None and call.tool == "play_radio" and call.args == {"name": "ska"}


def test_missing_station_asks_instead_of_reading_list():
    radio_tools._stations = radio.catalogue()
    result = radio_tools.play_radio({"name": "polka"})
    assert result.ok and "polka" in result.speech and "Which station" in result.speech
    assert result.detail.startswith(radio.ASK_RADIO)
