"""Media players: VLC (video), Audacious (music) and an in-app GStreamer playbin."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any
from urllib import parse

from .host import host_command, spawn
from .ytdlp import Media

Gst: Any = None  # Initialized lazily so importing this module needs no GStreamer or PyGObject.
GLib: Any = None


def _ensure_gstreamer() -> Any:
    global Gst, GLib
    if Gst is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib as _GLib
        from gi.repository import Gst as _Gst

        _Gst.init(None)
        Gst = _Gst
        GLib = _GLib
    return Gst


class MprisPlayer:
    """Generic control of any player that speaks MPRIS over the session bus."""

    def __init__(self, bus_suffix: str) -> None:
        self.bus_suffix = bus_suffix
        self.bus = f"org.mpris.MediaPlayer2.{bus_suffix}"
        self.object_path = "/org/mpris/MediaPlayer2"
        self.iface = "org.mpris.MediaPlayer2.Player"

    def _call(self, method: str, params: str | None = None) -> str:
        argv = [
            "gdbus",
            "call",
            "--session",
            "--dest",
            self.bus,
            "--object-path",
            self.object_path,
            "--method",
            f"{self.iface}.{method}",
        ]
        if params is not None:
            argv += ["--params", params]
        return self._run(argv)

    def _run(self, argv: list[str]) -> str:
        proc = subprocess.run(host_command(argv), capture_output=True, text=True, check=False)
        return proc.stdout

    def pause(self) -> None:
        self._call("Pause")

    def resume(self) -> None:
        self._call("Resume")

    def toggle(self) -> None:
        if self.status() == "Playing":
            self.pause()
        else:
            self.resume()

    def stop(self) -> None:
        self._call("Stop")

    def next(self) -> None:
        self._call("Next")

    def previous(self) -> None:
        self._call("Previous")

    def seek(self, seconds: float) -> None:
        self._call("SetProperty", '"org.mpris.MediaPlayer2.Player", "Relative", <true>')
        self._call("Seek", f"<int64 {int(seconds * 1_000_000)}>")

    def set_volume(self, level: float) -> None:
        volume = max(0, min(100, int(round(level * 100))))
        self._call("SetProperty", f'"org.mpris.MediaPlayer2.Player", "Volume", <uint64 {volume}>')

    def status(self) -> str:
        out = self._call("GetProperty", '"org.mpris.MediaPlayer2.Player", "PlaybackStatus"')
        match = re.search(r"'([^']*)'", out)
        return match.group(1) if match else ""


class VlcPlayer(MprisPlayer):
    """Video playback through the VLC Flatpak."""

    def __init__(self) -> None:
        super().__init__("vlc")

    def play(self, media: Media, fullscreen: bool = False) -> None:
        argv = [
            "flatpak",
            "run",
            "org.videolan.VLC",
            "--one-instance",
            "--meta-title",
            media.title,
            media.video_url,
        ]
        if media.audio_url:
            argv += ["--input-slave", media.audio_url]
        if fullscreen:
            argv += ["--fullscreen"]
        spawn(argv)


class AudaciousPlayer(MprisPlayer):
    """Music playback through Audacious (the org.atheme.audacious Flatpak)."""

    def __init__(self) -> None:
        super().__init__("audacious")

    def play(self, media: Media) -> None:
        url = media.audio_url or media.video_url
        if shutil.which("audacious"):
            argv = ["audacious", url]
        else:
            argv = ["flatpak", "run", "org.atheme.audacious", url]
        spawn(argv)


class AudioPlayer:
    """Music inside Voxa with no window: a GStreamer playbin."""

    def __init__(self) -> None:
        self.pipeline = None

    def _uri(self, media: Media) -> str:
        url = media.audio_url or media.video_url
        if parse.urlparse(url).scheme:
            return url
        from pathlib import Path

        return Path(url).as_uri()

    def play(self, media: Media) -> None:
        gst = _ensure_gstreamer()
        pipeline = gst.ElementFactory.make("playbin")
        if pipeline is None:
            return
        pipeline.set_property("uri", self._uri(media))
        self.pipeline = pipeline
        pipeline.set_state(gst.State.PLAYING)

    def pause(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PAUSED)

    def resume(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.STOPPED)
            self.pipeline = None

    def set_volume(self, level: float) -> None:
        if self.pipeline is not None:
            self.pipeline.set_property("volume", max(0.0, min(1.0, level)))

    def status(self) -> str:
        if self.pipeline is None:
            return ""
        state = self.pipeline.get_state()[0]
        if state == Gst.State.PLAYING:
            return "Playing"
        if state == Gst.State.PAUSED:
            return "Paused"
        return "Stopped"


def _installed(app_id: str, binary: str) -> bool:
    if shutil.which(binary):
        return True
    try:
        proc = subprocess.run(
            host_command(["flatpak", "--list", "applications", "--flatpak-paths", ""]),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return app_id in proc.stdout


def music_player(name: str | None = None):
    """Pick the music player app: vlc (default), audacious, or builtin."""
    choice = (name or os.environ.get("VOXA_MUSIC_PLAYER") or "vlc").strip().lower()
    if choice == "builtin":
        return AudioPlayer()
    if choice == "audacious" and _installed("org.atheme.audacious", "audacious"):
        return AudaciousPlayer()
    if _installed("org.videolan.VLC", "vlc"):
        return VlcPlayer()
    return AudioPlayer()


def active_player():
    """The first player that is actually running, else the builtin if active."""
    for player in (VlcPlayer(), AudaciousPlayer()):
        if player.status() in ("Playing", "Paused"):
            return player
    builtin = AudioPlayer()
    if builtin.status() in ("Playing", "Paused"):
        return builtin
    return None
