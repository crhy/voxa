from types import SimpleNamespace

import pytest

pytest.importorskip("gi")  # the picker module needs GTK; machines without it skip this file

from voxa.agent import radio  # noqa: E402
from voxa.ui.radio_picker import station_rows  # noqa: E402

STATIONS = list(radio.BUILTIN)


def test_station_rows_are_numbered_with_genre_and_description():
    rows = station_rows([SimpleNamespace(name="Groove Salad", genre="ambient downtempo chill electronica", description="A plate of beats.")])
    assert rows == [("1", "Groove Salad", "ambient, downtempo, chill — A plate of beats.")]
    assert station_rows([SimpleNamespace(name="Plain", genre="", description="")]) == [("1", "Plain", "")]


def test_a_number_or_a_name_answers_the_station_list():
    assert radio.chosen("three", STATIONS) is STATIONS[2]
    assert radio.chosen("number 2 please", STATIONS) is STATIONS[1]
    assert radio.chosen("the first one", STATIONS) is STATIONS[0]
    assert radio.chosen("play ska radio", STATIONS).name == "Bob's Ska Radio"
    assert radio.chosen("Groove Salad", STATIONS).name == "Groove Salad"


def test_a_new_request_is_not_taken_as_a_station():
    for text in ("what is seven times eight", "what time is it", "open the calculator", "set a timer for 3 minutes", ""):
        assert radio.chosen(text, STATIONS) is None, text
