"""Probe yt-dlp + VLC end to end: resolve a query, then really play the audio stream."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxa.agent.ytdlp import ResolveError, resolve


def main(query: str) -> int:
    try:
        media = resolve(query)
    except ResolveError as exc:
        print(f"Could not reach YouTube: {exc}")
        return 1

    print(f"title: {media.title}")
    print(f"duration: {media.duration}")
    print(f"split streams: {not media.combined}")
    if media.video_url:
        print(f"video_url[:80]: {media.video_url[:80]}")
    if media.audio_url:
        print(f"audio_url[:80]: {media.audio_url[:80]}")

    stream = media.audio_url or media.video_url
    argv = [
        "flatpak",
        "run",
        "org.videolan.VLC",
        "--intf",
        "dummy",
        "--no-video",
        "--play-and-exit",
        "--run-time",
        "6",
        stream,
    ]
    start = time.monotonic()
    proc = subprocess.run(argv, check=False)
    elapsed = time.monotonic() - start
    print(f"vlc exit code: {proc.returncode}")
    print(f"elapsed seconds: {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "big buck bunny"))
