from __future__ import annotations

from voxa.agent.ytdlp import ResolveError, latest_from_channel, resolve


def _info(formats, **meta):
    base = {"title": "T", "uploader": "U", "duration": 10, "webpage_url": "https://x/y"}
    base.update(meta)
    base["formats"] = formats
    return base


def _extractor(info):
    return lambda _q: info


def test_split_streams():
    info = _info(
        [
            {"format_id": "a", "url": "aud", "vcodec": "none", "acodec": "opus", "abr": 128},
            {"format_id": "v", "url": "vid", "vcodec": "h264", "acodec": "none", "height": 720, "abr": 1000},
        ]
    )
    media = resolve("q", extractor=_extractor(info))
    assert media.video_url == "vid"
    assert media.audio_url == "aud"
    assert media.combined is False


def test_combined_only():
    info = _info(
        [
            {"format_id": "18", "url": "comb", "vcodec": "h264", "acodec": "mp4a", "height": 480, "abr": 500},
        ]
    )
    media = resolve("q", extractor=_extractor(info))
    assert media.video_url == "comb"
    assert media.audio_url is None
    assert media.combined is True


def test_height_cap():
    info = _info(
        [
            {"format_id": "a", "url": "aud", "vcodec": "none", "acodec": "opus", "abr": 128},
            {"format_id": "v720", "url": "vid720", "vcodec": "h264", "acodec": "none", "height": 720, "abr": 1000},
            {"format_id": "v1080", "url": "vid1080", "vcodec": "h264", "acodec": "none", "height": 1080, "abr": 2000},
        ]
    )
    media = resolve("q", max_height=720, extractor=_extractor(info))
    assert media.video_url == "vid720"


def test_audio_only():
    info = _info(
        [
            {"format_id": "a1", "url": "aud1", "vcodec": "none", "acodec": "opus", "abr": 128},
            {"format_id": "a2", "url": "aud2", "vcodec": "none", "acodec": "opus", "abr": 256},
        ]
    )
    media = resolve("q", audio_only=True, extractor=_extractor(info))
    assert media.video_url is None
    assert media.audio_url == "aud2"
    assert media.combined is False


def test_prefers_avc_over_av01():
    info = _info(
        [
            {"format_id": "a", "url": "aud", "vcodec": "none", "acodec": "opus", "abr": 128},
            {"format_id": "av01", "url": "av01", "vcodec": "av01", "acodec": "none", "height": 720, "abr": 900},
            {"format_id": "avc", "url": "avc", "vcodec": "avc1", "acodec": "none", "height": 720, "abr": 800},
        ]
    )
    media = resolve("q", extractor=_extractor(info))
    assert media.video_url == "avc"


def test_query_becomes_ytsearch1(monkeypatch):
    seen = {}
    monkeypatch.setattr("voxa.agent.ytdlp.ytdlp_options", lambda: {"js_runtimes": {"node": {"path": "/n"}}})

    def fake_fetch(target, options):
        seen["target"] = target
        seen["options"] = options
        return _info([{"format_id": "18", "url": "comb", "vcodec": "h264", "acodec": "mp4a", "height": 480, "abr": 500}])

    monkeypatch.setattr("voxa.agent.ytdlp._fetch", fake_fetch)
    resolve("big buck bunny")
    assert seen["target"] == "ytsearch1:big buck bunny"
    assert seen["options"] == {"js_runtimes": {"node": {"path": "/n"}}}


def test_resolve_error():
    info = _info([])
    try:
        resolve("q", extractor=_extractor(info))
    except ResolveError:
        pass
    else:
        raise AssertionError("expected ResolveError")


def test_latest_from_channel_parsing():
    info = {
        "title": "chan",
        "entries": [
            {"url": "https://www.youtube.com/watch?v=1", "title": "new"},
            {"url": "https://www.youtube.com/watch?v=2", "title": "older"},
        ],
    }
    urls = latest_from_channel("chan", count=2, extractor=lambda _u: info)
    assert urls == ["https://www.youtube.com/watch?v=1", "https://www.youtube.com/watch?v=2"]


def test_no_runtime_gives_friendly_error(monkeypatch):
    monkeypatch.setattr("voxa.agent.ytdlp.ytdlp_options", lambda: {})
    try:
        resolve("big buck bunny")
    except ResolveError as exc:
        assert str(exc) == "Playing YouTube directly needs Node.js or Deno installed. Install one, or I'll use the browser instead."
    else:
        raise AssertionError("expected ResolveError")


def test_js_runtimes_reach_the_factory(monkeypatch):
    seen = {}

    class FakeYdl:
        def __init__(self, options):
            seen["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, _target, download=True):  # noqa: A002 - yt-dlp's signature
            return _info([{"format_id": "18", "url": "comb", "vcodec": "h264", "acodec": "mp4a", "height": 480, "abr": 500}])

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYdl)
    monkeypatch.setattr("voxa.agent.ytdlp.ytdlp_options", lambda: {"js_runtimes": {"node": {"path": "/app/bin/voxa-host-node"}}})
    media = resolve("q")
    assert seen["options"]["js_runtimes"] == {"node": {"path": "/app/bin/voxa-host-node"}}
    assert media.video_url == "comb"


def test_audio_only_falls_back_to_combined():
    info = _info(
        [
            {"format_id": "v", "url": "vid", "vcodec": "h264", "acodec": "none", "height": 720, "abr": 1000},
            {"format_id": "c", "url": "comb", "vcodec": "h264", "acodec": "opus", "height": 480, "abr": 128},
        ]
    )
    media = resolve("q", audio_only=True, extractor=_extractor(info))
    assert media.audio_url == "comb"
    assert media.video_url is None
    assert media.combined is True
