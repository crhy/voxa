from voxa.endpoint import is_complete


def test_play_ufo_waits_for_full_pause():
    assert is_complete("play UFO") is False


def test_play_some_jazz_waits_for_full_pause():
    assert is_complete("play some jazz") is False


def test_open_audacious_is_complete():
    assert is_complete("open audacious") is True


def test_stop_is_complete():
    assert is_complete("stop") is True
