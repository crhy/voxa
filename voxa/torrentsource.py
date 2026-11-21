# SPDX-License-Identifier: GPL-3.0-or-later
"""Read a release .torrent and choose where one download part comes from (GitHub issue #117).

Voxa's big downloads (the High face bundle, face packs) are published as GitHub release
downloads, which are capped at 2 GB per file and come from a single source. A torrent published
with the release lets every machine share what it already has. This module is the offline half of
that: it reads the torrent's metadata and decides, per part, whether the bytes should come from a
peer on this machine, from the release's web seed, or from the plain GitHub download.

Four failure modes are designed around.

A peer must never change what gets installed: a file already on disk is used only when it hashes
to the checksum published with the release. A partial that does not hash is refused outright, not
resumed, because resuming would keep the wrong bytes and only the final checksum would notice.

A half-read torrent must never steer a download: the decoder is strict, so a truncated file,
trailing garbage after the top-level value, unsorted dictionary keys, a negative integer or a
string whose declared length runs past the end of the file all raise instead of yielding a
plausible-looking torrent. An unreadable torrent is not fatal further up: choose_source falls back
to the plain release download.

The default source must not be a metered connection: a web seed whose URL throttles the transfer
through its query string is refused, and the fallback is the release download at full speed.

Nothing here touches the network. No tracker is contacted, no client is started, no peer is
contacted: the stdlib decoder and the policy are exercised on fixtures, and the .torrent bytes are
handed in by the caller.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxa.faceinstall import Part

# A .torrent is a handful of nested dictionaries at most; anything deeper is a crafted file.
_MAX_BENCODE_DEPTH = 16
# Query parameters that throttle a transfer: GitHub's own metered release links use ?mb=,
# other hosts use ?rate=, ?rate_limit=, ?throttle=, ?bandwidth=, ?speed=, ?bps= or ?limit=.
_METERED_QUERY_RE = re.compile(r"(mb|rate|throttle|meter|bandwidth|speed|bps|limit)", re.IGNORECASE)
_SHA1_BYTES = 20
_READ_CHUNK = 1 << 20
# A web seed URL is handed to curl as one argument, so it must hold no whitespace or control
# characters; a URL that does is dropped rather than split into extra arguments.
_URL_SAFE_RE = re.compile(r"\A[^\s\x00-\x1f\x7f]+\Z")


class TorrentError(ValueError):
    """A torrent that could not be read as published; a ValueError, so callers that already
    catch bad manifests keep working."""


@dataclass(frozen=True)
class TorrentMeta:
    announce: str = ""
    announce_list: tuple[tuple[str, ...], ...] = ()
    url_list: tuple[str, ...] = ()
    name: str = ""
    piece_length: int = 0
    pieces: bytes = b""
    length: int = 0
    private: bool = False
    files: tuple[tuple[str, int], ...] = ()


def decode_bencode(data: bytes) -> tuple[object, int]:
    """Decode the one bencoded value at the start of *data*; returns (value, end offset).

    Keys come back as bytes, strings as bytes, integers as int, lists as list and dictionaries
    as dict with the keys in file order. Strict: trailing bytes after the value, unsorted or
    duplicate dictionary keys, negative or non-canonical integers and strings whose length runs
    past the end of the file all raise.
    """
    value, idx = _decode(data, 0, 0)
    if idx != len(data):
        raise TorrentError(f"trailing bytes after the torrent at offset {idx}")
    return value, idx


def _decode(data: bytes, i: int, depth: int) -> tuple[object, int]:
    if depth > _MAX_BENCODE_DEPTH:
        raise TorrentError("bencoded value nested too deeply")
    if i >= len(data):
        raise TorrentError("truncated torrent")
    head = data[i : i + 1]
    if head == b"d":
        out: dict[bytes, object] = {}
        i += 1
        prev: bytes | None = None
        while i < len(data) and data[i : i + 1] != b"e":
            key, i = _decode(data, i, depth + 1)
            if not isinstance(key, bytes):
                raise TorrentError("torrent dictionary key is not a string")
            if prev is not None and key <= prev:
                raise TorrentError("torrent dictionary keys are not sorted")
            prev = key
            value, i = _decode(data, i, depth + 1)
            out[key] = value
        if i >= len(data) or data[i : i + 1] != b"e":
            raise TorrentError("unterminated torrent dictionary")
        return out, i + 1
    if head == b"l":
        items: list[object] = []
        i += 1
        while i < len(data) and data[i : i + 1] != b"e":
            value, i = _decode(data, i, depth + 1)
            items.append(value)
        if i >= len(data) or data[i : i + 1] != b"e":
            raise TorrentError("unterminated torrent list")
        return items, i + 1
    if head == b"i":
        end = data.find(b"e", i + 1)
        if end < 0:
            raise TorrentError("unterminated torrent integer")
        token = data[i + 1 : end]
        if token.startswith(b"-"):
            raise TorrentError("negative integer in a torrent")
        if not token.isdigit() or (len(token) > 1 and token[:1] == b"0"):
            raise TorrentError(f"malformed torrent integer {token!r}")
        return int(token), end + 1
    if head.isdigit():
        colon = data.find(b":", i)
        if colon < 0:
            raise TorrentError("malformed torrent string")
        size_token = data[i:colon]
        if not size_token.isdigit():
            raise TorrentError("malformed torrent string length")
        size = int(size_token)
        start = colon + 1
        if start + size > len(data):
            raise TorrentError("torrent string runs past the end of the file")
        return data[start : start + size], start + size
    raise TorrentError(f"unexpected byte {head!r} in a torrent")


def _text(value: object, what: str) -> str:
    if not isinstance(value, bytes):
        raise TorrentError(f"torrent {what} is not a string")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TorrentError(f"torrent {what} is not valid UTF-8") from exc


def _int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TorrentError(f"torrent {what} is not an integer")
    return value


def _path(segments: object) -> str:
    """Join a torrent file's path segments, refusing anything that could escape the download."""
    if not isinstance(segments, list) or not segments:
        raise TorrentError("torrent file path is not a non-empty list")
    parts = [_text(seg, "file path segment") for seg in segments]
    for seg in parts:
        if not seg or "/" in seg or "\\" in seg or seg in (".", "..") or seg.startswith("."):
            raise TorrentError(f"bad torrent file path segment: {seg}")
    return "/".join(parts)


def read_torrent(data: bytes) -> TorrentMeta:
    """Read a .torrent file's bytes into TorrentMeta.

    Raises TorrentError rather than returning a half-read torrent: a file that is truncated,
    padded, unsorted or otherwise not what a client would have written is refused here, so the
    caller falls back to the plain release download instead of acting on a guess.
    """
    value, _ = decode_bencode(data)
    if not isinstance(value, dict):
        raise TorrentError("torrent is not a dictionary")
    top = value
    announce = _text(top[b"announce"], "announce") if b"announce" in top else ""
    announce_list: tuple[tuple[str, ...], ...] = ()
    if b"announce-list" in top:
        raw = top[b"announce-list"]
        if not isinstance(raw, list):
            raise TorrentError("torrent announce-list is not a list")
        announce_list = tuple(
            tuple(_text(entry, "announce") for entry in tier)
            for tier in raw
            if isinstance(tier, list)
        )
    url_list: tuple[str, ...] = ()
    if b"url-list" in top:
        raw = top[b"url-list"]
        if not isinstance(raw, list):
            raise TorrentError("torrent url-list is not a list")
        url_list = tuple(_text(entry, "url-list entry") for entry in raw)
    name = _text(top[b"name"], "name") if b"name" in top else ""
    if not name:
        raise TorrentError("torrent has no name")
    info = top.get(b"info")
    if not isinstance(info, dict):
        raise TorrentError("torrent has no info dictionary")
    piece_length = _int(info[b"piece length"], "piece length") if b"piece length" in info else 0
    if piece_length <= 0:
        raise TorrentError("torrent piece length is not positive")
    pieces = info[b"pieces"] if b"pieces" in info else b""
    if not isinstance(pieces, bytes):
        raise TorrentError("torrent pieces is not a string")
    private = info.get(b"private") == 1
    length = _int(info[b"length"], "length") if b"length" in info else 0
    files: list[tuple[str, int]] = []
    if b"files" in info:
        raw = info[b"files"]
        if not isinstance(raw, list) or not raw:
            raise TorrentError("torrent files is not a non-empty list")
        for entry in raw:
            if not isinstance(entry, dict):
                raise TorrentError("torrent file entry is not a dictionary")
            files.append((_path(entry.get(b"path")), _int(entry.get(b"length"), "file length")))
    elif length:
        files = [(name, length)]
    else:
        raise TorrentError("torrent declares neither files nor length")
    return TorrentMeta(
        announce=announce,
        announce_list=announce_list,
        url_list=url_list,
        name=name,
        piece_length=piece_length,
        pieces=pieces,
        length=length,
        private=private,
        files=tuple(files),
    )


def piece_sha1s(meta: TorrentMeta) -> list[str]:
    """The per-piece SHA-1 digests the torrent publishes, hex, in piece order.

    Raises when the pieces blob is not a whole number of 20-byte digests: a torrent whose
    pieces do not divide evenly is corrupt, and a partial digest would check nothing.
    """
    if len(meta.pieces) % _SHA1_BYTES:
        raise TorrentError(
            f"torrent pieces is {len(meta.pieces)} bytes, not a whole number of "
            f"{_SHA1_BYTES}-byte digests"
        )
    return [
        meta.pieces[i : i + _SHA1_BYTES].hex()
        for i in range(0, len(meta.pieces), _SHA1_BYTES)
    ]


def total_length(meta: TorrentMeta) -> int:
    """The number of bytes the torrent says it carries."""
    if not meta.files:
        return meta.length
    total = sum(size for _, size in meta.files)
    if meta.length and meta.length != total:
        raise TorrentError(f"torrent length {meta.length} disagrees with its files ({total})")
    return total


def web_seed_urls(meta: TorrentMeta) -> tuple[str, ...]:
    """The torrent's web seeds, keeping only https:// entries with no whitespace in them.

    A plain http seed is dropped: it can be rewritten on the way to the machine, and the release
    it stands for is published over https. A URL containing whitespace is dropped too, because
    the installer hands each one to curl as a single argument.
    """
    return tuple(
        url for url in meta.url_list if url.startswith("https://") and _URL_SAFE_RE.match(url)
    )


def web_seed_urls_for(meta: TorrentMeta | None, part: Part) -> tuple[str, ...]:
    """The web seeds that serve this part, in the order the torrent publishes them.

    A single-file torrent's entries are the file itself; an entry ending in "/" is a base URL the
    file name is appended to; anything else has to name the part. Metered entries are dropped.
    """
    if meta is None:
        return ()
    return tuple(
        seed
        for seed in (_web_seed_for(meta, url, part) for url in web_seed_urls(meta))
        if seed
    )


def _is_metered(url: str) -> bool:
    """True when the URL's query throttles the transfer (GitHub's metered links use ?mb=)."""
    query = url.split("?", 1)[1].split("#", 1)[0] if "?" in url else ""
    return any(_METERED_QUERY_RE.search(key) for key, _value in _parse_qsl(query))


def _parse_qsl(query: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for chunk in query.split("&"):
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        pairs.append((key, value))
    return pairs


def _web_seed_for(meta: TorrentMeta, url: str, part: Part) -> str:
    """The web seed URL that serves this part, or "" when this entry does not serve it."""
    if _is_metered(url):
        return ""
    if url.endswith("/"):
        return url + part.name  # a base URL: the file name is appended
    tail = url.rsplit("/", 1)[-1].split("?", 1)[0]
    if tail == part.name:
        return url
    if not meta.files and meta.name == part.name:
        return url  # a single-file torrent: the entry is the file itself
    return ""


def _file_sha256(path: Path) -> str:
    """The sha256 of a file on disk; "" when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def choose_source(
    meta: TorrentMeta | None,
    part: Part,
    local_partial: Path | None,
) -> tuple[str, str, str]:
    """Pick where one part comes from: (kind, url_or_path, reason).

    Exactly three kinds are returned.

    "peer" with the path already on disk, and only when that file hashes to the checksum published
    with the release. A partial that does not hash is refused, never resumed: resuming would keep
    someone else's bytes and leave only the final checksum to notice.

    "webseed" with an https url-list entry that serves this part, and never one whose query meters
    the connection.

    "github" with the part's own release URL: the plain download, and the answer whenever there is
    no torrent, no usable web seed, or the torrent could not be read.
    """
    if local_partial is not None:
        digest = _file_sha256(Path(local_partial))
        if digest and digest == part.sha256.lower():
            return (
                "peer",
                str(local_partial),
                f"{part.name} is already on disk and matches the published checksum",
            )
    if meta is not None:
        seeds = web_seed_urls_for(meta, part)
        if seeds:
            return ("webseed", seeds[0], f"the release publishes a web seed for {part.name}")
    return ("github", part.url, f"no peer and no web seed has {part.name}; using the release download")
