from __future__ import annotations

import pytest

from voxa.agent.player import AudaciousPlayer, AudioPlayer, VlcPlayer, active_player, music_player


@pytest.fixture
def captured(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.spawn", lambda argv: calls.append(list(argv)))
    return calls


def _vlc_argv(media, fullscreen=False):
    argv = ["flatpak", "run", "org.videolan.VLC", "--one-instance", "--meta-title", media.title, media.video_url]
    if media.audio_url:
        argv += ["--input-slave", media.audio_url]
    if fullscreen:
        argv += ["--fullscreen"]
    return argv


def test_vlc_argv_with_audio_companion(captured):
    media = _media("V", "vid", "aud")
    VlcPlayer().play(media)
    assert captured[0] == _vlc_argv(media)


def test_vlc_argv_without_audio_companion(captured):
    media = _media("V", "comb", None)
    VlcPlayer().play(media)
    assert captured[0] == _vlc_argv(media)


def test_vlc_argv_fullscreen(captured):
    media = _media("V", "vid", "aud")
    VlcPlayer().play(media, fullscreen=True)
    assert captured[0] == _vlc_argv(media, fullscreen=True)


def test_audacious_argv_flatpak(captured, monkeypatch):
    monkeypatch.setattr("voxa.agent.player.shutil.which", lambda _b: None)
    media = _media("M", None, "aud")
    AudaciousPlayer().play(media)
    assert captured[0] == ["flatpak", "run", "org.atheme.audacious", "aud"]


def test_audacious_argv_binary_present(captured, monkeypatch):
    monkeypatch.setattr("voxa.agent.player.shutil.which", lambda b: "/usr/bin/audacious" if b == "audacious" else None)
    media = _media("M", "vid", None)
    AudaciousPlayer().play(media)
    assert captured[0] == ["audacious", "vid"]


def test_mpris_pause_argv(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.subprocess.run", lambda *a, **k: seen.append(a[0]) or _completed(a[0]))
    VlcPlayer().pause()
    assert seen[0] == [
        "gdbus", "call", "--session", "--dest", "org.mpris.MediaPlayer2.vlc",
        "--object-path", "/org/mpris/MediaPlayer2", "--method", "org.mpris.MediaPlayer2.Player.Pause",
    ]


def test_mpris_seek_argv(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.subprocess.run", lambda *a, **k: seen.append(a[0]) or _completed(a[0]))
    VlcPlayer().seek(5.0)
    assert seen[1] == [
        "gdbus", "call", "--session", "--dest", "org.mpris.MediaPlayer2.vlc",
        "--object-path", "/org/mpris/MediaPlayer2", "--method", "org.mpris.MediaPlayer2.Player.Seek",
        "--params", "<int64 5000000>",
    ]


def test_mpris_set_volume_argv(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.subprocess.run", lambda *a, **k: seen.append(a[0]) or _completed(a[0]))
    VlcPlayer().set_volume(0.5)
    assert seen[0] == [
        "gdbus", "call", "--session", "--dest", "org.mpris.MediaPlayer2.vlc",
        "--object-path", "/org/mpris/MediaPlayer2", "--method", "org.mpris.MediaPlayer2.Player.SetProperty",
        "--params", '"org.mpris.MediaPlayer2.Player", "Volume", <uint64 50>',
    ]


def test_mpris_status_parses(monkeypatch):
    monkeypatch.setattr(
        "voxa.agent.player.subprocess.run",
        lambda *a, **k: _completed(a[0], stdout="variant  string 'Playing'\n"),
    )
    assert VlcPlayer().status() == "Playing"


def test_music_player_default_vlc(monkeypatch):
    monkeypatch.setattr("voxa.agent.player._installed", lambda app_id, binary: app_id == "org.videolan.VLC")
    assert music_player().bus == "org.mpris.MediaPlayer2.vlc"


def test_music_player_audacious(monkeypatch):
    monkeypatch.setattr("voxa.agent.player._installed", lambda app_id, binary: app_id == "org.atheme.audacious")
    assert music_player("audacious").bus == "org.mpris.MediaPlayer2.audacious"


def test_music_player_builtin(monkeypatch):
    monkeypatch.setattr("voxa.agent.player._installed", lambda app_id, binary: False)
    assert isinstance(music_player("builtin"), AudioPlayer)


def test_music_player_unknown_falls_back_to_vlc(monkeypatch):
    monkeypatch.setattr("voxa.agent.player._installed", lambda app_id, binary: app_id == "org.videolan.VLC")
    assert music_player("mpv").bus == "org.mpris.MediaPlayer2.vlc"


def test_music_player_not_installed_falls_back_to_builtin(monkeypatch):
    monkeypatch.setattr("voxa.agent.player._installed", lambda app_id, binary: False)
    assert isinstance(music_player("audacious"), AudioPlayer)


def test_active_player_choice(monkeypatch):
    monkeypatch.setattr("voxa.agent.player.VlcPlayer.status", lambda _self: "")
    monkeypatch.setattr("voxa.agent.player.AudaciousPlayer.status", lambda _self: "Paused")
    assert active_player().bus == "org.mpris.MediaPlayer2.audacious"


def test_active_player_builtin(monkeypatch):
    monkeypatch.setattr("voxa.agent.player.VlcPlayer.status", lambda _self: "")
    monkeypatch.setattr("voxa.agent.player.AudaciousPlayer.status", lambda _self: "")
    monkeypatch.setattr("voxa.agent.player.AudioPlayer.status", lambda _self: "Playing")
    assert isinstance(active_player(), AudioPlayer)


def _media(title, video_url, audio_url):
    from voxa.agent.ytdlp import Media

    return Media(title, "U", 10, "https://x/y", video_url, audio_url, audio_url is None)


def _completed(argv, stdout=""):
    import subprocess

    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
