import hashlib
from pathlib import Path

import pytest

from voxa.faceinstall import Part
from voxa.torrentsource import (
    TorrentError,
    TorrentMeta,
    choose_source,
    decode_bencode,
    piece_sha1s,
    read_torrent,
    total_length,
    web_seed_urls,
    web_seed_urls_for,
)

SHA_A = "a" * 64
PIECE = bytes(range(20))


def bencode(value: object) -> bytes:
    """A tiny bencode writer, so the fixtures are the same bytes a client would publish."""
    if isinstance(value, int):
        return b"i%de" % value
    if isinstance(value, bytes):
        return b"%d:%s" % (len(value), value)
    if isinstance(value, list):
        return b"l" + b"".join(bencode(item) for item in value) + b"e"
    if isinstance(value, dict):
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in sorted(value.items())) + b"e"
    raise TypeError(type(value))


RELEASE_SEED = "https://github.com/crhy/voxa/releases/download/face-server-1/face-server-1.tar.xz.part00"


def torrent_bytes(
    *,
    name: str = "face-server-1.tar.xz.part00",
    length: int = 500,
    pieces: bytes = PIECE * 2,
    url_list: tuple[str, ...] = (RELEASE_SEED,),
    announce: str = "https://tracker.example/announce",
    piece_length: int = 400,
) -> bytes:
    return bencode({
        b"announce": announce.encode(),
        b"info": {
            b"length": length,
            b"piece length": piece_length,
            b"pieces": pieces,
            b"private": 1,
        },
        b"name": name.encode(),
        b"url-list": [url.encode() for url in url_list],
    })


def _part(name: str = "face-server-1.tar.xz.part00", length: int = 500) -> Part:
    return Part(name=name, url="https://github.com/crhy/voxa/releases/download/face-server-1/" + name,
                bytes=length, sha256=SHA_A)


def test_decoder_round_trips_a_torrent():
    raw = torrent_bytes()
    value, end = decode_bencode(raw)
    assert end == len(raw)
    assert isinstance(value, dict)
    assert value[b"name"] == b"face-server-1.tar.xz.part00"
    assert value[b"info"][b"piece length"] == 400
    nested = {b"k": [1, b"s"], b"n": {b"deep": b"d"}}
    assert decode_bencode(bencode(nested)) == (nested, len(bencode(nested)))


def test_read_torrent_collects_the_published_metadata():
    meta = read_torrent(torrent_bytes(url_list=("https://github.com/x/p0", "http://plain.example/p0")))
    assert meta.name == "face-server-1.tar.xz.part00"
    assert meta.announce == "https://tracker.example/announce"
    assert meta.piece_length == 400
    assert meta.length == 500
    assert meta.private is True
    assert meta.files == (("face-server-1.tar.xz.part00", 500),)
    assert meta.url_list == ("https://github.com/x/p0", "http://plain.example/p0")
    assert total_length(meta) == 500


def test_piece_sha1s_gives_one_digest_per_piece():
    meta = read_torrent(torrent_bytes(pieces=PIECE * 3))
    digests = piece_sha1s(meta)
    assert len(digests) == 3
    assert digests[0] == PIECE.hex()


def test_piece_sha1s_refuses_a_ragged_pieces_blob():
    meta = TorrentMeta(name="p", pieces=PIECE + b"\x01")
    with pytest.raises(TorrentError):
        piece_sha1s(meta)


def test_decoder_refuses_a_truncated_torrent():
    raw = torrent_bytes()
    with pytest.raises(TorrentError):
        read_torrent(raw[: len(raw) // 2])


def test_decoder_refuses_trailing_bytes():
    with pytest.raises(TorrentError):
        read_torrent(torrent_bytes() + b"junk")


def test_decoder_refuses_unsorted_keys():
    unsorted = b"d" + b"4:name" + b"1:p" + b"8:announce" + b"https://t" + b"e"
    with pytest.raises(TorrentError):
        decode_bencode(unsorted)


def test_decoder_refuses_negative_ints_and_truncated_strings():
    with pytest.raises(TorrentError):
        decode_bencode(b"i-5e")
    with pytest.raises(TorrentError):
        decode_bencode(b"9:abc")


def test_a_non_https_web_seed_is_never_accepted_as_a_source():
    meta = read_torrent(torrent_bytes(url_list=("http://plain.example/p0",)))
    assert web_seed_urls(meta) == ()
    assert web_seed_urls_for(meta, _part()) == ()
    kind, url, _reason = choose_source(meta, _part(), None)
    assert kind == "github"
    assert url == _part().url
    malformed = bencode({b"announce": b"https://t", b"url-list": b"https://x", b"name": b"p", b"info": {b"length": 1}})
    with pytest.raises(TorrentError):
        read_torrent(malformed)


def test_a_torrent_length_that_disagrees_with_the_manifest_raises():
    from voxa.faceinstall import parse_manifest

    text = (
        '{"version": "1", "parts": [{"name": "p0",'
        ' "url": "https://github.com/x/p0", "bytes": 500, "sha256": "' + SHA_A + '"}],'
        ' "torrent": "https://github.com/x/p0.torrent"}'
    )
    matching = read_torrent(torrent_bytes(name="p0", length=500))
    assert parse_manifest(text, matching).torrent == "https://github.com/x/p0.torrent"
    wrong = read_torrent(torrent_bytes(name="p0", length=480))
    with pytest.raises(ValueError):
        parse_manifest(text, wrong)
    with pytest.raises(TorrentError):
        total_length(TorrentMeta(name="p", length=500, files=(("p", 480),)))


def test_parse_manifest_refuses_a_non_https_torrent():
    from voxa.faceinstall import parse_manifest

    text = (
        '{"version": "1", "parts": [{"name": "p0",'
        ' "url": "https://github.com/x/p0", "bytes": 500, "sha256": "' + SHA_A + '"}],'
        ' "torrent": "http://insecure.example/p0.torrent"}'
    )
    with pytest.raises(ValueError):
        parse_manifest(text)


def test_choose_source_uses_a_peer_only_when_the_bytes_match_the_published_checksum(tmp_path):
    part = _part()
    good = tmp_path / part.name
    good.write_bytes(b"the exact bytes the release publishes")
    # Make the published checksum describe those bytes, as a real manifest would.
    part = Part(name=part.name, url=part.url, bytes=good.stat().st_size,
                sha256=hashlib.sha256(good.read_bytes()).hexdigest())
    kind, url, reason = choose_source(None, part, good)
    assert kind == "peer"
    assert url == str(good)
    assert "checksum" in reason


def test_choose_source_refuses_a_peer_partial_that_does_not_hash(tmp_path):
    part = _part()
    bad = tmp_path / part.name
    bad.write_bytes(b"someone else's bytes")
    kind, url, _reason = choose_source(None, part, bad)
    assert kind == "github"
    assert url == part.url
    assert url != str(bad)


def test_choose_source_prefers_a_web_seed_over_the_plain_download():
    part = _part()
    meta = read_torrent(torrent_bytes(name=part.name))
    kind, url, _reason = choose_source(meta, part, None)
    assert kind == "webseed"
    assert url == "https://github.com/crhy/voxa/releases/download/face-server-1/face-server-1.tar.xz.part00"

def test_choose_source_falls_back_when_the_torrent_serves_another_file():
    other = "https://cdn.example/face-packs-1/portrait.png"
    meta = read_torrent(torrent_bytes(name="portrait.png", length=10, url_list=(other,)))
    kind, url, _reason = choose_source(meta, _part(), None)
    assert kind == "github"
    assert url == _part().url


def test_choose_source_never_hands_out_a_metered_connection():
    part = _part()
    for metered in (
        "https://github.com/x/p0?mb=1",
        "https://github.com/x/p0?rate_limit=100",
        "https://github.com/x/p0?throttle=60",
        "https://cdn.example/p0?bandwidth=2m",
    ):
        meta = read_torrent(torrent_bytes(url_list=(metered,)))
        kind, url, _reason = choose_source(meta, part, None)
        assert kind == "github", metered
        assert url == part.url
        assert web_seed_urls_for(meta, part) == ()


def test_a_base_web_seed_url_gets_the_part_name_appended():
    part = _part()
    meta = read_torrent(torrent_bytes(url_list=("https://github.com/crhy/voxa/releases/download/face-server-1/",)))
    assert web_seed_urls_for(meta, part) == (
        "https://github.com/crhy/voxa/releases/download/face-server-1/" + part.name,
    )


def test_read_torrent_refuses_a_path_that_could_escape_the_download():
    info = {b"files": [{b"length": 10, b"path": [b"..", b"escape"]}], b"length": 10, b"piece length": 4}
    raw = bencode({b"name": b"bundle", b"info": info})
    with pytest.raises(TorrentError):
        read_torrent(raw)
