from __future__ import annotations

import voxa.agent.tools.media as media
from voxa.agent.intents import route
from voxa.agent.tools.media import (
    _play_latest_handler,
    _set_youtube_channel_handler,
    clean_channel,
    is_my_channel,
)


def test_is_my_channel():
    assert is_my_channel("")
    assert is_my_channel("my channel")
    assert is_my_channel("my youtube channel")
    assert is_my_channel("my own channel")
    assert is_my_channel("mychannel")
    assert is_my_channel("my")
    assert not is_my_channel("Veritasium")
    assert not is_my_channel("rhykhanz")


def test_clean_channel():
    assert clean_channel("R H Y K H A N Z") == "rhykhanz"
    assert clean_channel("@RhyKhanz") == "rhykhanz"
    assert clean_channel("rhy khanz") == "rhykhanz"
    assert clean_channel("r-h-y") == "rhy"
    assert clean_channel("  ") == ""


def test_play_latest_without_channel(monkeypatch):
    calls = []

    def recorder(channel, *args, **kwargs):
        calls.append(channel)
        return []

    monkeypatch.setattr(media, "latest_from_channel", recorder)
    monkeypatch.setattr(media, "get_my_channel", None)
    monkeypatch.delenv("VOXA_YOUTUBE_CHANNEL", raising=False)
    result = _play_latest_handler({"channel": "my youtube channel"})
    assert not result.ok
    assert "I don't know your channel yet" in result.speech
    assert calls == []


def test_play_latest_with_saved_channel(monkeypatch):
    calls = []

    def recorder(channel, *args, **kwargs):
        calls.append(channel)
        return ["https://some.url/watch?v=1"]

    class FakeMedia:
        title = "New Upload"
        webpage_url = "https://some.url/watch?v=1"

    class FakePlayer:
        def play(self, media):
            return None

    monkeypatch.setattr(media, "latest_from_channel", recorder)
    monkeypatch.setattr(media, "get_my_channel", lambda: "rhykhanz")
    monkeypatch.setattr(media, "resolve", lambda url: FakeMedia())
    monkeypatch.setattr(media, "VlcPlayer", lambda: FakePlayer())
    result = _play_latest_handler({"channel": "my channel"})
    assert calls == ["@rhykhanz"]
    assert result.ok
    assert "New Upload" in result.speech


def test_play_latest_named_channel_passthrough(monkeypatch):
    calls = []

    def recorder(channel, *args, **kwargs):
        calls.append(channel)
        return []

    monkeypatch.setattr(media, "latest_from_channel", recorder)
    _play_latest_handler({"channel": "Veritasium"})
    assert calls == ["Veritasium"]


def test_set_youtube_channel_saves(monkeypatch):
    saved = []
    monkeypatch.setattr(media, "set_my_channel", lambda handle: saved.append(handle))
    result = _set_youtube_channel_handler({"name": "R H Y K H A N Z"})
    assert saved == ["rhykhanz"]
    assert result.ok
    assert "rhykhanz" in result.speech


def test_router_set_channel():
    call = route("my youtube channel is R H Y K H A N Z")
    assert call is not None
    assert call.tool == "set_youtube_channel"
    assert call.args["name"] == "R H Y K H A N Z"


def test_router_play_latest_my_channel():
    call = route("play the latest video from my youtube channel")
    assert call is not None
    assert call.tool == "play_latest"
    assert is_my_channel(call.args["channel"])
