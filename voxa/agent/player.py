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


def mpris_bus_names(prefix: str) -> list[str]:
    """Every session-bus name that is org.mpris.MediaPlayer2.<prefix> or starts with that plus ".".

    Extra instances of a player register as org.mpris.MediaPlayer2.<prefix>.instanceNNNN, so a
    single fixed bus name does not reach them; list the bus names instead.
    """
    argv = [
        "gdbus",
        "call",
        "--session",
        "--dest",
        "org.freedesktop.DBus",
        "--object-path",
        "/org/freedesktop/DBus",
        "--method",
        "org.freedesktop.DBus.ListNames",
    ]
    try:
        proc = subprocess.run(host_command(argv), capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    base = f"org.mpris.MediaPlayer2.{prefix}"
    names: list[str] = []
    for match in re.finditer(r"'([^']*)'", proc.stdout):
        name = match.group(1)
        if name == base or name.startswith(f"{base}."):
            names.append(name)
    return names


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

    def __init__(self, bus_suffix: str, kill_cmds: list[list[str]] | None = None) -> None:
        self.bus_suffix = bus_suffix
        self.bus = f"org.mpris.MediaPlayer2.{bus_suffix}"
        self.object_path = "/org/mpris/MediaPlayer2"
        self.iface = "org.mpris.MediaPlayer2.Player"
        self.kill_cmds = kill_cmds or []

    def _gdbus(self, method: str, *args: str) -> str:
        """`gdbus call` on the player; arguments are GVariant text, one per argument (gdbus has no --params)."""
        return self._gdbus_on(self.bus, method, *args)

    def _gdbus_on(self, bus: str, method: str, *args: str) -> str:
        argv = [
            "gdbus",
            "call",
            "--session",
            "--dest",
            bus,
            "--object-path",
            self.object_path,
            "--method",
            method,
            *args,
        ]
        return self._run(argv)

    def _call(self, method: str, *args: str) -> str:
        return self._gdbus(f"{self.iface}.{method}", *args)

    def _get(self, name: str) -> str:
        return self._gdbus("org.freedesktop.DBus.Properties.Get", self.iface, name)

    def _run(self, argv: list[str]) -> str:
        try:
            proc = subprocess.run(host_command(argv), capture_output=True, text=True, check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return ""
        return proc.stdout

    def pause(self) -> None:
        self._call("Pause")

    def resume(self) -> None:
        self._call("Play")

    def toggle(self) -> None:
        self._call("PlayPause")

    def _buses(self) -> list[str]:
        """Every bus name this player may have registered, falling back to the primary one."""
        return mpris_bus_names(self.bus_suffix) or [self.bus]

    def stop(self) -> None:
        """Stop playback and close the player: "stop the music" means the player goes away.

        Reaches every instance (org.mpris.MediaPlayer2.vlc.instanceNNNN included), and if any
        still reports a status after Stop+Quit, kills the app as a last resort.
        """
        buses = self._buses()
        for bus in buses:
            self._gdbus_on(bus, f"{self.iface}.Stop")
            self._gdbus_on(bus, "org.mpris.MediaPlayer2.Quit")
        if any(self._status_on(bus) for bus in buses):
            for cmd in self.kill_cmds:
                if self._run_ok(cmd):
                    break

    def _run_ok(self, argv: list[str]) -> bool:
        try:
            proc = subprocess.run(host_command(argv), capture_output=True, text=True, check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def next(self) -> None:
        self._call("Next")

    def previous(self) -> None:
        self._call("Previous")

    def seek(self, seconds: float) -> None:
        """Relative seek; MPRIS takes microseconds."""
        self._call("Seek", str(int(seconds * 1_000_000)))

    def volume(self) -> float | None:
        match = re.search(r"<([0-9.eE+-]+)>", self._get("Volume"))
        return float(match.group(1)) if match else None

    def set_volume(self, level: float) -> None:
        """Absolute volume, 0.0 to 1.0."""
        level = max(0.0, min(1.0, float(level)))
        self._gdbus("org.freedesktop.DBus.Properties.Set", self.iface, "Volume", f"<{level:.2f}>")

    def change_volume(self, delta: float) -> None:
        current = self.volume()
        self.set_volume((current if current is not None else 0.5) + delta)

    def _status_on(self, bus: str) -> str:
        match = re.search(r"'([^']*)'", self._gdbus_on(bus, "org.freedesktop.DBus.Properties.Get", self.iface, "PlaybackStatus"))
        return match.group(1) if match else ""

    def status(self) -> str:
        """First non-empty playback status over every bus this player registered."""
        for bus in self._buses():
            value = self._status_on(bus)
            if value:
                return value
        return ""


class VlcPlayer(MprisPlayer):
    """Video playback through the VLC Flatpak."""

    def __init__(self) -> None:
        super().__init__("vlc", [["flatpak", "kill", "org.videolan.VLC"]])

    def play(self, media: Media, fullscreen: bool = False) -> None:
        # Music has only an audio stream: play that on its own. Video gets the audio as a companion stream.
        main = media.video_url or media.audio_url
        if not main:
            raise ValueError("nothing to play: the stream has no audio or video address")
        argv = [
            "flatpak",
            "run",
            "org.videolan.VLC",
            "--one-instance",
            "--control",
            "dbus",
            "--meta-title",
            media.title,
            main,
        ]
        if media.video_url and media.audio_url:
            argv += ["--input-slave", media.audio_url]
        if fullscreen:
            argv += ["--fullscreen"]
        if not media.video_url:
            argv += ["--no-video"]
        spawn(argv)


class AudaciousPlayer(MprisPlayer):
    """Music playback through Audacious (the org.atheme.audacious Flatpak)."""

    def __init__(self) -> None:
        super().__init__("audacious", [["flatpak", "kill", "org.atheme.audacious"], ["pkill", "-x", "audacious"]])

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


_INSTALLED_CACHE: dict[tuple[str, str], bool] = {}


def _installed(app_id: str, binary: str) -> bool:
    """True when the player exists on the HOST, as a plain binary or as a Flatpak.

    Inside Voxa's own Flatpak neither is visible directly, so the host is asked. A positive answer is
    remembered; a negative one is asked again next time (the user may have just installed it).
    """
    key = (app_id, binary)
    if _INSTALLED_CACHE.get(key):
        return True
    found = False
    try:
        proc = subprocess.run(
            host_command(["sh", "-c", f"command -v {binary} >/dev/null 2>&1 || flatpak info {app_id} >/dev/null 2>&1"]),
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        found = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        found = shutil.which(binary) is not None
    if found:
        _INSTALLED_CACHE[key] = True
    return found


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
