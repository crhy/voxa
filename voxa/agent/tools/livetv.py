from __future__ import annotations

from voxa.agent import livetv, player
from voxa.agent.player import VlcPlayer
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.ytdlp import Media

now_watching: livetv.Channel | None = None


def _ask(speech: str) -> ToolResult:
    return ToolResult.success(speech, detail=livetv.ASK_TV + "\n" + "\n".join(livetv.numbered()))


def play_tv(args: dict[str, str]) -> ToolResult:
    global now_watching
    wanted = args.get("name", "")
    if not livetv.clean_request(wanted):
        return _ask("Which channel would you like?")
    channel = livetv.find(wanted)
    if channel is None:
        return _ask(f"I do not have a channel called {livetv.clean_request(wanted)}. Which one would you like?")
    media = Media(channel.name, "live tv", 0, channel.stream, channel.stream, None, True)
    # Her "Playing …" comes first: a stream that starts at once plays over her own sentence.
    player.start_after(player.START_DELAY_SECONDS, lambda: VlcPlayer().play(media))
    now_watching = channel
    return ToolResult.success(f"Playing {channel.name}.", detail=channel.stream)


def list_tv(args: dict[str, str]) -> ToolResult:
    return _ask(f"I have {len(livetv.CHANNELS)} live channels. Which one would you like?")


def livetv_tools() -> list[Tool]:
    return [
        Tool(name="play_tv", description="Play a live internet television channel by name.",
             parameters={"name": "channel name; empty to be asked which one"},
             risk=RiskLevel.REVERSIBLE, handler=play_tv),
        Tool(name="list_tv", description="Show the live television channels Voxa knows.", parameters={},
             risk=RiskLevel.READ_ONLY, handler=list_tv),
    ]
