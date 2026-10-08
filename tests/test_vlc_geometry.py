from __future__ import annotations

from voxa.agent.tools.windows import vlc_geometry


def test_vlc_geometry_seventy_five_percent_of_voxa():
    assert vlc_geometry((100, 100, 1200, 760), (2560, 1440)) == (1300, 100, 900, 570)


def test_vlc_geometry_right_of_voxa_when_screen_is_wide_enough():
    # Voxa spans x=100..1300; the screen (2560) fits 100+1200+900=2200, so VLC goes at x=1300.
    x, y, _w, _h = vlc_geometry((100, 100, 1200, 760), (2560, 1440))
    assert x == 1300 and y == 100


def test_vlc_geometry_centred_when_screen_is_not_wide_enough():
    # 1920 < 100+1200+900: centred on the screen instead.
    assert vlc_geometry((100, 100, 1200, 760), (1920, 1080)) == (510, 255, 900, 570)


def test_vlc_geometry_falls_back_to_900x570():
    # Unknown Voxa geometry and screen: 75% of the 1200x760 default, at the origin.
    assert vlc_geometry((0, 0, 1200, 760), (0, 0)) == (0, 0, 900, 570)


def test_vlc_geometry_never_bigger_than_voxa():
    for voxa_geom, screen in [
        ((0, 0, 1200, 760), (1920, 1080)),
        ((100, 100, 1200, 760), (2560, 1440)),
        ((0, 0, 800, 500), (1024, 768)),
    ]:
        _x, _y, w, h = vlc_geometry(voxa_geom, screen)
        assert w < voxa_geom[2] and h < voxa_geom[3]


def test_vlc_argv_has_no_fullscreen_option(monkeypatch):
    from voxa.agent.player import VlcPlayer
    from voxa.agent.ytdlp import Media

    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.spawn", lambda argv: calls.append(list(argv)))
    monkeypatch.setattr("voxa.agent.tools.windows.frame_vlc_window", lambda: None)
    media = Media("T", "U", 10, "https://x/y", "vid", "aud", False)
    VlcPlayer().play(media)
    assert "--no-fullscreen" in calls[0]
    assert "--fullscreen" not in calls[0]


def test_vlc_frames_only_video(monkeypatch):
    from voxa.agent.player import VlcPlayer
    from voxa.agent.ytdlp import Media

    calls: list[list[str]] = []
    monkeypatch.setattr("voxa.agent.player.spawn", lambda argv: calls.append(list(argv)))
    monkeypatch.setattr("voxa.agent.tools.windows.frame_vlc_window", lambda: calls.append(["frame"]))
    audio_only = Media("Song", "U", 10, "https://x/y", None, "aud", True)
    VlcPlayer().play(audio_only)
    assert calls[0][-1] == "--no-video"
    assert calls[-1] != ["frame"]
