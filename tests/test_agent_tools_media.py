from __future__ import annotations

import os

from voxa.agent.tools import default_registry
from voxa.agent.tools.media import (
    _media_control_handler,
    _play_latest_handler,
    _play_music_handler,
    _play_video_handler,
    media_tools,
)
from voxa.agent.ytdlp import Media, ResolveError


def _make_media(title: str) -> Media:
    return Media(title, "uploader", 100, f"https://example/{title}", "v", "a", False)


class FakePlayer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def play(self, media: Media, fullscreen: bool = False) -> None:
        self.calls.append(("play", media))

    def pause(self) -> None:
        self.calls.append(("pause", None))

    def resume(self) -> None:
        self.calls.append(("resume", None))

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def next(self) -> None:
        self.calls.append(("next", None))

    def previous(self) -> None:
        self.calls.append(("previous", None))

    def seek(self, seconds: float) -> None:
        self.calls.append(("seek", seconds))

    def set_volume(self, level: float) -> None:
        self.calls.append(("set_volume", level))


def test_play_video_speech(monkeypatch):
    played = FakePlayer()
    monkeypatch.setattr("voxa.agent.tools.media.resolve", lambda q, **kw: _make_media("Lo-fi Beats"))
    monkeypatch.setattr("voxa.agent.tools.media.shutil.which", lambda name: "/usr/bin/vlc")
    monkeypatch.setattr("voxa.agent.tools.media.VlcPlayer", lambda: played)
    result = _play_video_handler({"query": "lo-fi beats"})
    assert result.ok
    assert result.speech == "Playing Lo-fi Beats."
    assert played.calls[0][0] == "play"


def test_play_video_falls_back_on_resolve_error(monkeypatch):
    def boom(query, **kw):
        raise ResolveError("nothing playable")

    fallback_calls: list[dict] = []
    monkeypatch.setattr("voxa.agent.tools.media.resolve", boom)
    monkeypatch.setattr(
        "voxa.agent.tools.media.play_youtube",
        lambda args: fallback_calls.append(args)
        or __import__("voxa.agent.result", fromlist=["ToolResult"]).ToolResult(True, "Playing lo-fi beats on YouTube.", "browser"),
    )
    result = _play_video_handler({"query": "lo-fi beats"})
    assert result.ok
    assert fallback_calls == [{"query": "lo-fi beats"}]
    assert "resolve failed" in result.detail


def test_play_video_falls_back_when_vlc_missing(monkeypatch):
    monkeypatch.setattr("voxa.agent.tools.media.resolve", lambda q, **kw: _make_media("Lo-fi Beats"))
    monkeypatch.setattr("voxa.agent.tools.media.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "voxa.agent.tools.media.play_youtube",
        lambda args: __import__("voxa.agent.result", fromlist=["ToolResult"]).ToolResult(True, "Playing lo-fi beats on YouTube.", "browser"),
    )
    result = _play_video_handler({"query": "lo-fi beats"})
    assert result.ok
    assert "VLC not installed" in result.detail


def test_play_music_speech(monkeypatch):
    played = FakePlayer()
    monkeypatch.setattr("voxa.agent.tools.media.resolve", lambda q, **kw: _make_media("Jazz"))
    monkeypatch.setattr("voxa.agent.tools.media.music_player", lambda: played)
    result = _play_music_handler({"query": "jazz"})
    assert result.ok
    assert result.speech == "Playing Jazz."
    assert played.calls[0][0] == "play"


def test_play_latest_uses_env_channel(monkeypatch):
    monkeypatch.setenv("VOXA_YOUTUBE_CHANNEL", "TechLinked")
    monkeypatch.setattr("voxa.agent.tools.media.latest_from_channel", lambda channel, **kw: ["https://yt/watch/1"])
    monkeypatch.setattr("voxa.agent.tools.media.resolve", lambda q, **kw: _make_media("New Upload"))
    monkeypatch.setattr("voxa.agent.tools.media.VlcPlayer", lambda: FakePlayer())
    result = _play_latest_handler({"channel": "my channel"})
    assert result.ok
    assert result.speech == "Playing New Upload."


def test_play_latest_without_channel_fails():
    os.environ.pop("VOXA_YOUTUBE_CHANNEL", None)
    result = _play_latest_handler({"channel": ""})
    assert not result.ok
    assert result.speech == "I don't know which channel to play."


def test_media_control_uses_active_player(monkeypatch):
    active = FakePlayer()
    monkeypatch.setattr("voxa.agent.tools.media.active_player", lambda: active)
    result = _media_control_handler({"action": "pause"})
    assert result.ok
    assert active.calls == [("pause", None)]
    assert result.detail == "player:pause"


def test_media_control_seek_actions(monkeypatch):
    active = FakePlayer()
    monkeypatch.setattr("voxa.agent.tools.media.active_player", lambda: active)
    _media_control_handler({"action": "forward"})
    _media_control_handler({"action": "back"})
    assert active.calls == [("seek", 30), ("seek", -30)]


def test_media_control_volume_actions(monkeypatch):
    active = FakePlayer()
    monkeypatch.setattr("voxa.agent.tools.media.active_player", lambda: active)
    _media_control_handler({"action": "louder"})
    _media_control_handler({"action": "quieter"})
    assert active.calls == [("set_volume", 0.1), ("set_volume", 0.0)]


def test_media_control_falls_back_to_key_press(monkeypatch):
    monkeypatch.setattr("voxa.agent.tools.media.active_player", lambda: None)
    pressed: list[dict] = []
    monkeypatch.setattr(
        "voxa.agent.tools.media._press_key_handler",
        lambda args: pressed.append(args)
        or __import__("voxa.agent.result", fromlist=["ToolResult"]).ToolResult(True, "", "key"),
    )
    result = _media_control_handler({"action": "next"})
    assert pressed == [{"key": "next track"}]
    assert result.ok


def test_media_control_unknown_action():
    result = _media_control_handler({"action": "teleport"})
    assert not result.ok
    assert result.speech == "I don't know that control."


def test_media_tools_registered():
    names = set(default_registry().names())
    for tool in media_tools():
        assert tool.name in names
