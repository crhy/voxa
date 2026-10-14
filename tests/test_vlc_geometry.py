from __future__ import annotations

import subprocess

from voxa.agent.tools.windows import monitor_at, parse_monitors, vlc_geometry


def test_vlc_geometry_is_16_9_on_1920x1080():
    _x, _y, w, h = vlc_geometry((0, 0, 1920, 1080))
    assert abs(w - round(h * 16 / 9)) <= 1


def test_vlc_geometry_is_16_9_on_1366x768():
    _x, _y, w, h = vlc_geometry((0, 0, 1366, 768))
    assert abs(w - round(h * 16 / 9)) <= 1


def test_vlc_geometry_is_smaller_than_voxa_minimum():
    _x, _y, width, height = vlc_geometry((0, 0, 1920, 1080))
    assert (width, height) == (640, 360)
    assert width < 850
    assert height < 600


def test_vlc_geometry_is_centred_on_current_monitor():
    assert vlc_geometry((1920, 0, 1366, 768)) == (2283, 204, 640, 360)


def test_vlc_geometry_narrows_to_fit_on_small_screen():
    assert vlc_geometry((-600, 40, 600, 320)) == (-549, 60, 498, 280)


def test_vlc_geometry_falls_back_when_screen_unknown():
    assert vlc_geometry((0, 0, 0, 0)) == (0, 0, 640, 360)


def test_parse_monitors_and_choose_monitor_under_pointer():
    listing = """DP-1 connected primary 1920x1080+0+0
HDMI-1 connected 1366x768+1920+120
DP-2 connected 1280x1024-1280+0
"""
    monitors = parse_monitors(listing)
    assert monitors == [(0, 0, 1920, 1080), (1920, 120, 1366, 768), (-1280, 0, 1280, 1024)]
    assert monitor_at((2300, 400), monitors) == (1920, 120, 1366, 768)
    assert monitor_at((-400, 500), monitors) == (-1280, 0, 1280, 1024)


def test_frame_vlc_restores_centres_and_raises(monkeypatch):
    from voxa.agent.tools import windows

    calls: list[list[str]] = []
    monkeypatch.setattr(windows.simulation, "actions_simulated", lambda: False)
    monkeypatch.setattr(
        windows,
        "list_windows",
        lambda: [windows.Window("0x1", "vlc.Vlc", "RT News - VLC")],
    )
    monkeypatch.setattr(windows, "_current_monitor_geometry", lambda: (1920, 0, 1366, 768))
    monkeypatch.setattr(windows.time, "sleep", lambda _seconds: None)

    def run(command, check=True):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(windows, "_run", run)
    windows.frame_vlc_window()

    assert ["xdotool", "windowmap", "0x1"] in calls
    assert [
        "wmctrl", "-i", "-r", "0x1", "-b",
        "remove,hidden,maximized_vert,maximized_horz,fullscreen",
    ] in calls
    assert ["wmctrl", "-i", "-r", "0x1", "-e", "0,2283,204,640,360"] in calls
    assert calls[-2:] == [
        ["xdotool", "windowraise", "0x1"],
        ["wmctrl", "-i", "-a", "0x1"],
    ]


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
