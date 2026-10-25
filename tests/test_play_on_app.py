from __future__ import annotations

from voxa import apps
from voxa.agent.intents import route
from voxa.agent.localmusic import pick_tracks
from voxa.agent.tools import localmusic as lm


def test_route_play_on_app():
    result = route("play boz skaggs on my audacious")
    assert result is not None
    assert result.tool == "play_on_app"
    assert result.args["query"] == "boz skaggs"
    assert result.args["app"] == "audacious"


def test_route_play_on_app_caps():
    result = route("Play UFO on my Audacious player")
    assert result is not None
    assert result.tool == "play_on_app"
    assert result.args["query"] == "UFO"
    assert result.args["app"] == "Audacious"


def test_route_play_music_in_app():
    result = route("play music in audacious")
    assert result is not None
    assert result.tool == "play_on_app"
    assert result.args["query"] == "music"
    assert result.args["app"] == "audacious"


def test_route_not_play_on_app():
    result = route("play despacito on youtube")
    assert result is None or result.tool != "play_on_app"


def test_pick_tracks_folder():
    files = [
        "/m/Boz Skaggs/Silk Degrees/track1.mp3",
        "/m/Boz Skaggs/Silk Degrees/track2.flac",
        "/m/Other/track.mp3",
    ]
    result = pick_tracks("boz skaggs", "/m", files)
    assert result == ["/m/Boz Skaggs/Silk Degrees/track1.mp3", "/m/Boz Skaggs/Silk Degrees/track2.flac"]


def test_pick_tracks_filename():
    files = ["/m/despacito.mp3", "/m/other.flac"]
    result = pick_tracks("despacito", "/m", files)
    assert result == ["/m/despacito.mp3"]


def test_pick_tracks_all_music():
    files = ["/m/a.mp3", "/m/b.flac", "/m/c.txt"]
    result = pick_tracks("music", "/m", files)
    assert result == ["/m/a.mp3", "/m/b.flac"]


def test_pick_tracks_no_match():
    files = ["/m/a.mp3"]
    result = pick_tracks("nonexistent", "/m", files)
    assert result == []


def test_pick_tracks_non_audio():
    files = ["/m/song.txt", "/m/song.mp3"]
    result = pick_tracks("song", "/m", files)
    assert result == ["/m/song.mp3"]


def test_play_on_app_success(monkeypatch):
    fake_app = apps.DesktopApp(name="Audacious", path="/fake.desktop")
    monkeypatch.setattr(apps, "list_apps", lambda: [fake_app])
    monkeypatch.setattr(apps, "match_app", lambda q, a: fake_app)
    launched: list[tuple[apps.DesktopApp, tuple[str, ...]]] = []

    def fake_launch(app: apps.DesktopApp, files=()):
        launched.append((app, tuple(files)))
        return True

    monkeypatch.setattr(apps, "launch", fake_launch)
    monkeypatch.setattr(lm, "_music_files", lambda: ("/m", ["/m/Boz Skaggs/track.mp3"]))
    result = lm.play_on_app({"query": "boz skaggs", "app": "audacious"})
    assert result.ok
    assert len(launched) == 1
    assert launched[0][1] == ("/m/Boz Skaggs/track.mp3",)


def test_play_on_app_unknown_app(monkeypatch):
    monkeypatch.setattr(apps, "list_apps", lambda: [])
    monkeypatch.setattr(apps, "match_app", lambda q, a: None)
    launched: list[tuple[apps.DesktopApp, tuple[str, ...]]] = []

    def fake_launch(app: apps.DesktopApp, files=()):
        launched.append((app, tuple(files)))
        return True

    monkeypatch.setattr(apps, "launch", fake_launch)
    monkeypatch.setattr(lm, "_music_files", lambda: ("/m", ["/m/track.mp3"]))
    result = lm.play_on_app({"query": "boz skaggs", "app": "nonexistent"})
    assert not result.ok
    assert len(launched) == 0


def test_play_on_app_no_music(monkeypatch):
    fake_app = apps.DesktopApp(name="Audacious", path="/fake.desktop")
    monkeypatch.setattr(apps, "list_apps", lambda: [fake_app])
    monkeypatch.setattr(apps, "match_app", lambda q, a: fake_app)
    launched: list[tuple[apps.DesktopApp, tuple[str, ...]]] = []

    def fake_launch(app: apps.DesktopApp, files=()):
        launched.append((app, tuple(files)))
        return True

    monkeypatch.setattr(apps, "launch", fake_launch)
    monkeypatch.setattr(lm, "_music_files", lambda: ("/m", ["/m/other.mp3"]))
    result = lm.play_on_app({"query": "nonexistent", "app": "audacious"})
    assert not result.ok
    assert len(launched) == 0
