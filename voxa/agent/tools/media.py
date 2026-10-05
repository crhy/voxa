from __future__ import annotations

import os
import shutil

from voxa.agent.player import VlcPlayer, active_player, music_player
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.browser import play_youtube
from voxa.agent.tools.typing import _press_key_handler
from voxa.agent.ytdlp import ResolveError, latest_from_channel, resolve

_CONTROL_KEYS = {
    "pause": "play pause",
    "resume": "play pause",
    "stop": "play pause",
    "next": "next track",
    "previous": "previous track",
    "louder": "volume up",
    "quieter": "volume down",
}


def _vlc_installed() -> bool:
    return shutil.which("vlc") is not None


def _apply_to_player(player, action: str) -> None:
    if action == "pause":
        player.pause()
    elif action == "resume":
        player.resume()
    elif action == "stop":
        player.stop()
    elif action == "next":
        player.next()
    elif action == "previous":
        player.previous()
    elif action == "forward":
        player.seek(30)
    elif action == "back":
        player.seek(-30)
    elif action == "louder":
        player.set_volume(0.1)
    elif action == "quieter":
        player.set_volume(0.0)


def _play_video_handler(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    try:
        media = resolve(query)
    except ResolveError as exc:
        fallback = play_youtube({"query": query})
        return ToolResult(fallback.ok, fallback.speech, detail=f"resolve failed ({exc}); browser fallback: {fallback.detail}")
    if not _vlc_installed():
        fallback = play_youtube({"query": query})
        return ToolResult(fallback.ok, fallback.speech, detail=f"VLC not installed; browser fallback: {fallback.detail}")
    VlcPlayer().play(media)
    return ToolResult.success(f"Playing {media.title}.", detail=media.webpage_url)


def _play_music_handler(args: dict[str, str]) -> ToolResult:
    query = args["query"]
    media = resolve(query, audio_only=True)
    music_player().play(media)
    return ToolResult.success(f"Playing {media.title}.", detail=media.webpage_url)


def _play_latest_handler(args: dict[str, str]) -> ToolResult:
    channel = args.get("channel", "")
    if not channel or channel in ("my channel", "mychannel", "my"):
        channel = os.environ.get("VOXA_YOUTUBE_CHANNEL", "")
    if not channel:
        return ToolResult.failure("I don't know which channel to play.")
    urls = latest_from_channel(channel)
    if not urls:
        return ToolResult.failure("That channel has no recent uploads.")
    media = resolve(urls[0])
    VlcPlayer().play(media)
    return ToolResult.success(f"Playing {media.title}.", detail=media.webpage_url)


def _media_control_handler(args: dict[str, str]) -> ToolResult:
    action = args["action"]
    player = active_player()
    if player is not None:
        _apply_to_player(player, action)
        return ToolResult.success("", detail=f"player:{action}")
    key = _CONTROL_KEYS.get(action)
    if key is None:
        return ToolResult.failure("I don't know that control.")
    return _press_key_handler({"key": key})


def media_tools() -> list[Tool]:
    return [
        Tool(
            name="play_video",
            description="Play a video by resolving it with yt-dlp and starting it in VLC.",
            parameters={"query": "what to play"},
            risk=RiskLevel.REVERSIBLE,
            handler=_play_video_handler,
            required=("query",),
        ),
        Tool(
            name="play_music",
            description="Play music by resolving an audio-only stream and starting the music player.",
            parameters={"query": "what music to play"},
            risk=RiskLevel.REVERSIBLE,
            handler=_play_music_handler,
            required=("query",),
        ),
        Tool(
            name="play_latest",
            description="Play the newest upload of a YouTube channel in VLC.",
            parameters={"channel": "the channel name, or empty for the saved channel"},
            risk=RiskLevel.REVERSIBLE,
            handler=_play_latest_handler,
        ),
        Tool(
            name="media_control",
            description="Control the running media: pause, resume, stop, next, previous, forward, back, louder, quieter.",
            parameters={"action": "one of pause, resume, stop, next, previous, forward, back, louder, quieter"},
            risk=RiskLevel.REVERSIBLE,
            handler=_media_control_handler,
            required=("action",),
        ),
    ]
