"""Tests for mis-heard wake words and noise in voxa/agent/hearing.py."""

from voxa.agent.hearing import direct_command, is_noise, should_drop_noise

STOP_MUSIC = [
    "Boxes, top music",
    "Fox, stop music",
    "So it's top music",
    "top music",
    "it's top music",
    "stop usic",
    "stopped music",
]

PAUSE = [
    "Xa, pause",
    "Step, pause",
    "Set, pause",
    "So pause",
]

NOISE = ["Music", "As", "Up", "Out", "Peace", "Pag", "Yeah"]


def test_direct_command_stop_music():
    for text in STOP_MUSIC:
        assert direct_command(text) == "stop music"


def test_direct_command_pause():
    for text in PAUSE:
        assert direct_command(text) == "pause"


def test_direct_command_rejects_long_sentences():
    assert direct_command("so pause the video when I say") is None


def test_direct_command_rejects_requests():
    assert direct_command("play jazz") is None
    assert direct_command("what is the capital of France") is None


def test_is_noise_drops_filler():
    for text in NOISE:
        assert is_noise(text)


def test_is_noise_drops_hallucination():
    assert is_noise("Blender, Blender, Blender, Blender, Blender, Blender, Blender")


def test_is_noise_keeps_requests():
    assert not is_noise("play jazz")
    assert not is_noise("what is the capital of France")


def test_control_words_are_not_noise():
    for word in ("stop", "play", "next", "skip", "mute"):
        assert not is_noise(word)


def test_followup_predicate_keeps_yes():
    assert should_drop_noise("Yes", followup_active=True) is False
    assert should_drop_noise("Yes", followup_active=False) is True
