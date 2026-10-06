from __future__ import annotations

import pytest

from voxa.agent.player import AudaciousPlayer, AudioPlayer, VlcPlayer, active_player, mpris_bus_names, music_player


@pytest.fixture
def captured(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.spawn", lambda argv: calls.append(list(argv)))
    return calls


def _vlc_argv(media, fullscreen=False):
    main = media.video_url or media.audio_url
    argv = ["flatpak", "run", "org.videolan.VLC", "--one-instance", "--control", "dbus", "--meta-title", media.title, main]
    if media.video_url and media.audio_url:
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
    # gdbus takes each argument on its own (there is no --params flag); MPRIS seeks in microseconds.
    assert seen[0] == [
        "gdbus", "call", "--session", "--dest", "org.mpris.MediaPlayer2.vlc",
        "--object-path", "/org/mpris/MediaPlayer2", "--method", "org.mpris.MediaPlayer2.Player.Seek", "5000000",
    ]


def test_mpris_set_volume_argv(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.subprocess.run", lambda *a, **k: seen.append(a[0]) or _completed(a[0]))
    VlcPlayer().set_volume(0.5)
    assert seen[0] == [
        "gdbus", "call", "--session", "--dest", "org.mpris.MediaPlayer2.vlc",
        "--object-path", "/org/mpris/MediaPlayer2", "--method", "org.freedesktop.DBus.Properties.Set",
        "org.mpris.MediaPlayer2.Player", "Volume", "<0.50>",
    ]


def test_mpris_stop_also_closes_the_player(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.subprocess.run", lambda *a, **k: seen.append(a[0]) or _completed(a[0]))
    VlcPlayer().stop()
    assert [argv[-1] for argv in seen] == [
        "org.freedesktop.DBus.ListNames",
        "org.mpris.MediaPlayer2.Player.Stop",
        "org.mpris.MediaPlayer2.Quit",
        "PlaybackStatus",
    ]


def test_vlc_plays_music_that_has_only_an_audio_stream(captured):
    media = _media("Song", None, "aud")
    VlcPlayer().play(media)
    assert "aud" in captured[0] and captured[0][-1] == "--no-video" and None not in captured[0]


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


def test_mpris_bus_names_two_instances(monkeypatch):
    monkeypatch.setattr(
        "voxa.agent.player.subprocess.run",
        lambda *a, **k: _completed(
            a[0],
            stdout="['org.mpris.MediaPlayer2.mpv', 'org.mpris.MediaPlayer2.vlc', "
            "'org.mpris.MediaPlayer2.vlc.instance1234', 'org.freedesktop.DBus']\n",
        ),
    )
    assert mpris_bus_names("vlc") == ["org.mpris.MediaPlayer2.vlc", "org.mpris.MediaPlayer2.vlc.instance1234"]


def test_mpris_stop_reaches_every_instance_and_kills(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(argv, **_k):
        seen.append(argv)
        if argv[-1] == "org.freedesktop.DBus.ListNames":
            stdout = "['org.mpris.MediaPlayer2.vlc', 'org.mpris.MediaPlayer2.vlc.instance1234']\n"
        elif "org.freedesktop.DBus.Properties.Get" in argv:
            stdout = "variant  string 'Playing'\n"
        else:
            stdout = ""
        return _completed(argv, stdout=stdout)

    monkeypatch.setattr("voxa.agent.player.subprocess.run", fake_run)
    VlcPlayer().stop()
    controls = [(argv[argv.index("--dest") + 1], argv[-1]) for argv in seen if "--dest" in argv]
    for bus in ("org.mpris.MediaPlayer2.vlc", "org.mpris.MediaPlayer2.vlc.instance1234"):
        assert (bus, "org.mpris.MediaPlayer2.Player.Stop") in controls
        assert (bus, "org.mpris.MediaPlayer2.Quit") in controls
    assert "org.videolan.VLC" in [argv[-1] for argv in seen]


def test_mpris_status_first_non_empty_over_instances(monkeypatch):
    def fake_run(argv, **_k):
        if argv[-1] == "org.freedesktop.DBus.ListNames":
            stdout = "['org.mpris.MediaPlayer2.vlc.instance1234', 'org.mpris.MediaPlayer2.vlc']\n"
        elif "org.freedesktop.DBus.Properties.Get" in argv:
            dest = argv[argv.index("--dest") + 1]
            stdout = "variant  string ''\n" if dest.endswith(".instance1234") else "variant  string 'Paused'\n"
        else:
            stdout = ""
        return _completed(argv, stdout=stdout)

    monkeypatch.setattr("voxa.agent.player.subprocess.run", fake_run)
    assert VlcPlayer().status() == "Paused"
