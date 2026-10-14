"""Live internet television: free, official round-the-clock streams (issue #100)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from voxa.agent.radio import norm

ASK_TV = "ask-tv"  # ToolResult.detail prefix: the numbered channel list follows, one per line


@dataclass(frozen=True, slots=True)
class Channel:
    name: str
    genre: str
    description: str
    stream: str
    aliases: tuple[str, ...] = ()


# Each address is the broadcaster's own public stream and was checked to answer on 2026-10-13.
CHANNELS: tuple[Channel, ...] = (
    Channel("NASA TV", "space science", "NASA's own channel: launches, the space station and science. No adverts.",
            "https://ntv1.akamaized.net/hls/live/2014075/NASA-NTV1-HLS/master.m3u8", ("nasa", "nasa television", "space")),
    Channel("DW News", "news world", "Deutsche Welle: world news in English from Germany's public broadcaster.",
            "https://dwamdstream102.akamaized.net/hls/live/2015525/dwstream102/index.m3u8",
            ("dw", "deutsche welle", "d w", "d w news", "world news", "news")),
    Channel("France 24", "news world", "France 24 in English: international news around the clock.",
            "https://live.france24.com/hls/live/2037218-b/F24_EN_HI_HLS/master_5000.m3u8",
            ("france twenty four", "france twenty-four", "france24")),
    Channel("Al Jazeera English", "news world", "Al Jazeera's English news channel.",
            "https://live-hls-apps-aje-fa.getaj.net/AJE/index.m3u8", ("al jazeera", "aljazeera", "al jazeera news")),
    Channel("RT News", "news world russian", "Russia Today's English-language international news channel.",
            "https://rt-glb.rttv.com/live/rtnews/playlist.m3u8",
            ("rt", "r t", "russia today", "russian news", "rt television")),
    Channel("RetroStrange TV", "classic movie movies vintage", "Public-domain classic movies, vintage television and shorts, all day.",
            "https://live.retrostrange.com/hls/stream.m3u8",
            ("retrostrange", "retro strange", "classic movie", "classic movies", "old movies", "vintage television")),
    Channel("Bloomberg Television", "business finance news", "Live business, markets and financial news.",
            "https://www.bloomberg.com/media-manifest/streams/phoenix-us.m3u8",
            ("bloomberg", "bloomberg tv", "business news", "financial news", "market news")),
    Channel("NHK World-Japan", "japan culture news world", "English-language news, culture and documentaries from Japan's public broadcaster.",
            "https://master.nhkworld.jp/nhkworld-tv/playlist/live.m3u8",
            ("nhk", "nhk world", "n h k", "japanese television", "japan tv")),
)


def clean_request(spoken: str) -> str:
    """"the NASA TV channel" -> "nasa"; "some live tv" -> ""."""
    text = norm(spoken)
    text = re.sub(r"^(?:the|some|a|me|my)\s+", "", text)
    text = re.sub(r"\b(?:live|internet|online|streaming)\b", " ", text)
    text = re.sub(r"\s*(?:tv|television|telly)?\s*(?:channels?|stations?|streams?)?\s*$", "", text.strip())
    return " ".join(text.split())


def find(spoken: str, channels: tuple[Channel, ...] = CHANNELS) -> Channel | None:
    """The channel named (by name, alias or a genre word); the first channel when nothing is named."""
    wanted = clean_request(spoken)
    if not wanted:
        return channels[0] if channels else None
    for channel in channels:
        names = (clean_request(channel.name), norm(channel.name), *(norm(alias) for alias in channel.aliases))
        if wanted in names:
            return channel
    for channel in channels:
        if all(word in norm(channel.name).split() for word in wanted.split()):
            return channel
    for channel in channels:
        if wanted in channel.genre.split():
            return channel
    return None


def is_channel(spoken: str, channels: tuple[Channel, ...] = CHANNELS) -> bool:
    """True when the words name a channel (so "watch NASA TV" is television, not a video search)."""
    return bool(clean_request(spoken)) and find(spoken, channels) is not None


def numbered(channels: tuple[Channel, ...] = CHANNELS) -> list[str]:
    return [f"{index}. {channel.name}" for index, channel in enumerate(channels, 1)]
