from __future__ import annotations

import os
import re
from collections.abc import Callable

from voxa.agent.player import AudaciousPlayer, AudioPlayer, VlcPlayer, active_player, music_player
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.browser import play_youtube
from voxa.agent.tools.typing import _press_key_handler
from voxa.agent.ytdlp import ResolveError, latest_from_channel, resolve

get_my_channel: Callable[[], str] | None = None
set_my_channel: Callable[[str], None] | None = None


def is_my_channel(channel: str) -> bool:
    if channel == "":
        return True
    text = channel.lower()
    for token in ("youtube", "you tube", "own", "channel", "the"):
        text = text.replace(token, "")
    text = text.replace(" ", "")
    return text in ("", "my", "mine")


def clean_channel(spoken: str) -> str:
    text = spoken.strip()
    if text.startswith("@"):
        text = text[1:]
    parts = re.split(r"[ -]+", text)
    if parts and all(len(part) == 1 and part.isalpha() for part in parts):
        text = "".join(parts)
    else:
        text = text.replace(" ", "")
    text = "".join(c for c in text if c.isalnum() or c in "_-.")
    return text.lower()


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
    from ..player import _installed

    return _installed("org.videolan.VLC", "vlc")


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
        player.change_volume(0.1) if hasattr(player, "change_volume") else player.set_volume(0.8)
    elif action == "quieter":
        player.change_volume(-0.1) if hasattr(player, "change_volume") else player.set_volume(0.3)


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
    if is_my_channel(channel):
        handle = get_my_channel() if get_my_channel is not None else ""
        if not handle:
            handle = os.environ.get("VOXA_YOUTUBE_CHANNEL", "")
        if not handle:
            return ToolResult.failure("I don't know your channel yet. Say: my YouTube channel is, and then its name.")
        if not (handle.startswith("@") or handle.startswith("http") or handle.startswith("UC")):
            handle = "@" + handle
        channel = handle
    urls = latest_from_channel(channel)
    if not urls:
        return ToolResult.failure("That channel has no recent uploads.")
    media = resolve(urls[0])
    VlcPlayer().play(media)
    return ToolResult.success(f"Playing {media.title}.", detail=media.webpage_url)


def _set_youtube_channel_handler(args: dict[str, str]) -> ToolResult:
    handle = clean_channel(args["name"])
    if not handle:
        return ToolResult.failure("I did not catch the channel name.")
    if set_my_channel is None:
        return ToolResult.failure("I cannot save that here.")
    set_my_channel(handle)
    return ToolResult.success(
        f"Got it. Your YouTube channel is {handle}. Say play the latest video from my channel."
    )


def _media_control_handler(args: dict[str, str]) -> ToolResult:
    action = args["action"]
    if action == "stop":
        stopped = False
        for player in (VlcPlayer(), AudaciousPlayer(), AudioPlayer()):
            if player.status() in ("Playing", "Paused"):
                _apply_to_player(player, "stop")
                stopped = True
        if stopped:
            return ToolResult.success("Stopped.", detail="player:stop")
        return _press_key_handler({"key": _CONTROL_KEYS["stop"]})
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
            name="set_youtube_channel",
            description="Remember the user's YouTube channel handle so 'my channel' resolves to it.",
            parameters={"name": "the channel name or handle"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_youtube_channel_handler,
            required=("name",),
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
