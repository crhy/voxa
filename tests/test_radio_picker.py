from voxa.agent import radio
from voxa.agent.tools import radio as radio_tools


def test_numbered():
    built = radio.catalogue()
    listed = radio.numbered(built)
    assert listed[0] == "1. Groove Salad" and listed[1] == "2. " + built[1].name
    assert len(listed) == len(built)


def test_pick_by_number():
    built = radio.catalogue()
    assert radio.pick("3", built) is built[2]
    assert radio.pick("number three", built) is built[2]
    assert radio.pick("the third one", built) is built[2]
    assert radio.pick("fourteen", built) is built[13]
    assert radio.pick("fifteen", built) is built[14]


def test_pick_falls_back_to_find():
    built = radio.catalogue()
    assert radio.pick("groove salad", built).name == "Groove Salad"
    assert radio.pick("ska", built).name == "Bob's Ska Radio"
    assert radio.pick("polka", built) is None


def test_no_station_offers_choices():
    radio_tools._stations = radio.catalogue()
    result = radio_tools.play_radio({"name": ""})
    assert result.ok and result.speech == "Which station would you like?"
    assert result.detail.startswith(radio.ASK_RADIO)
    lines = [line for line in result.detail[len(radio.ASK_RADIO):].split("\n") if line]
    assert lines == radio.numbered(radio_tools._stations)
    assert lines[0] == "1. Groove Salad"


def test_list_offers_choices():
    radio_tools._stations = radio.catalogue()
    result = radio_tools.list_radio({})
    assert result.ok and "15 stations" in result.speech
    assert result.detail.startswith(radio.ASK_RADIO)
    lines = [line for line in result.detail[len(radio.ASK_RADIO):].split("\n") if line]
    assert lines == radio.numbered(radio_tools._stations)
