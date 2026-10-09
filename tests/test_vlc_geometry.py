from __future__ import annotations

from voxa.agent.tools.windows import vlc_geometry


def test_vlc_geometry_is_16_9_on_1920x1080():
    for vw, vh in [(1200, 760), (900, 570), (1920, 1040)]:
        _x, _y, w, h = vlc_geometry((100, 100, vw, vh), (1920, 1080))
        assert abs(w - round(h * 16 / 9)) <= 1


def test_vlc_geometry_is_16_9_on_1366x768():
    for vw, vh in [(1200, 760), (900, 570), (1920, 1040)]:
        _x, _y, w, h = vlc_geometry((100, 100, vw, vh), (1366, 768))
        assert abs(w - round(h * 16 / 9)) <= 1


def test_vlc_geometry_height_matches_old_rule_when_screen_is_wide_enough():
    # The 16:9 width fits in screen minus 40 px, so the height stays at the
    # old 75% rule: round(760 * 0.75) = 570 and round(570 * 0.75) = 428.
    assert vlc_geometry((100, 100, 1200, 760), (1920, 1080)) == (453, 255, 1013, 570)
    assert vlc_geometry((100, 100, 900, 570), (1920, 1080)) == (1000, 100, 761, 428)


def test_vlc_geometry_narrows_to_fit_on_small_screen():
    # 1920x1040 on 1366x768: 75% height is 780, clamped to 768, whose 16:9
    # width (1365) exceeds 1366-40, so width becomes 1326 and height 746.
    assert vlc_geometry((100, 100, 1920, 1040), (1366, 768)) == (20, 11, 1326, 746)


def test_vlc_geometry_falls_back_when_screen_unknown():
    # Unknown screen: no fit rule applies, 16:9 width at 75% height, at origin.
    assert vlc_geometry((0, 0, 1200, 760), (0, 0)) == (0, 0, 1013, 570)


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
