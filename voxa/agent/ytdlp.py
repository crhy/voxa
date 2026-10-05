"""Resolve direct stream addresses with yt-dlp so "play ..." never needs a page."""

from __future__ import annotations

from dataclasses import dataclass

from voxa.agent.jsruntime import missing_runtime_message, ytdlp_options


@dataclass(frozen=True, slots=True)
class Media:
    title: str
    uploader: str
    duration: int
    webpage_url: str
    video_url: str | None
    audio_url: str | None
    combined: bool


class ResolveError(Exception):
    """Raised when no playable stream can be resolved."""


def _is_url(query_or_url: str) -> bool:
    return query_or_url.startswith(("http://", "https://"))


def _codec_rank(vcodec: str) -> int:
    """Prefer avc1/vp9 over av01 when picking a video-only stream."""
    if "avc" in vcodec or "vp9" in vcodec:
        return 2
    if "av01" in vcodec:
        return 1
    return 0


def _pick_audio(formats: list[dict]) -> str | None:
    audio = [f for f in formats if f.get("vcodec") == "none"]
    if not audio:
        return None
    best = max(audio, key=lambda f: f.get("abr") or 0)
    return best.get("url")


def _pick_video(formats: list[dict], max_height: int) -> str | None:
    video = [f for f in formats if f.get("acodec") == "none" and (f.get("height") or 0) <= max_height]
    if not video:
        return None
    best = max(video, key=lambda f: ((f.get("height") or 0), _codec_rank(f.get("vcodec") or "")))
    return best.get("url")


def _pick_combined(formats: list[dict], max_height: int) -> str | None:
    combined = [
        f
        for f in formats
        if f.get("vcodec") != "none" and f.get("acodec") != "none" and (f.get("height") or 0) <= max_height
    ]
    if not combined:
        return None
    best = max(combined, key=lambda f: ((f.get("height") or 0), f.get("abr") or 0))
    return best.get("url")


def _fetch(target: str, options: dict) -> dict:
    from yt_dlp import YoutubeDL

    try:
        with YoutubeDL({"quiet": True, "noplaylist": True, "skip_download": True, **options}) as ydl:
            return ydl.extract_info(target, download=False)
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many error types
        raise ResolveError(str(exc)) from exc


def resolve(query_or_url: str, max_height: int = 1080, audio_only: bool = False, extractor=None) -> Media:
    """Resolve a query or URL into a Media with directly playable stream addresses."""
    if extractor is not None:
        info = extractor(query_or_url)
    else:
        options = ytdlp_options()
        if "js_runtimes" not in options:
            raise ResolveError(missing_runtime_message())
        target = query_or_url if _is_url(query_or_url) else f"ytsearch1:{query_or_url}"
        info = _fetch(target, options)

    if not info.get("formats") and info.get("entries"):
        info = info["entries"][0]

    formats = info.get("formats") or []
    title = info.get("title") or ""
    uploader = info.get("uploader") or info.get("channel") or ""
    duration = int(info.get("duration") or 0)
    webpage_url = info.get("webpage_url") or ""

    if audio_only:
        audio = _pick_audio(formats)
        if audio is None:
            raise ResolveError("no audio-only stream available")
        return Media(title, uploader, duration, webpage_url, None, audio, False)

    video = _pick_video(formats, max_height)
    audio = _pick_audio(formats)
    if video is not None:
        return Media(title, uploader, duration, webpage_url, video, audio, False)

    combined = _pick_combined(formats, max_height)
    if combined is None:
        raise ResolveError("no playable stream available")
    return Media(title, uploader, duration, webpage_url, combined, None, True)


def latest_from_channel(channel: str, count: int = 1, extractor=None) -> list[str]:
    """Watch URLs of the newest uploads of a channel (extract_flat)."""
    url = f"https://www.youtube.com/@{channel}/videos"
    if extractor is not None:
        info = extractor(url)
    else:
        options = ytdlp_options()
        if "js_runtimes" not in options:
            raise ResolveError(missing_runtime_message())
        from yt_dlp import YoutubeDL

        try:
            with YoutubeDL(
                {"quiet": True, "noplaylist": True, "extract_flat": True, "skip_download": True, **options}
            ) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many error types
            raise ResolveError(str(exc)) from exc

    entries = info.get("entries") or []
    return [e.get("url") or e.get("watch_url") or "" for e in entries[:count]]
