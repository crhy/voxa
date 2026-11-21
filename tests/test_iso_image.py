"""Tests for finding, checking and downloading the latest Spaced Linux ISO.

No network is touched: the release JSON is a string and the download goes through
a fake opener, the way voxa.smallbrain.download is tested.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.isoimage import (
    DownloadCancelled,
    DownloadError,
    download_iso,
    fetch_file,
    free_bytes,
    pick_iso,
)
from voxa.agent.tools import default_registry


def _sysupdate_module():
    """The seam tests patch the release lookup on (same one as test_update_spaced)."""
    return sys.modules["voxa.agent.tools.sysupdate"]


def _release_json(body: str = "") -> str:
    """Two releases, each with amd64 and arm64 ISO assets."""
    return json.dumps(
        [
            {
                "tag_name": "v11.26.1",
                "body": body,
                "assets": [
                    {
                        "name": "spaced-linux-11.26.1-amd64.iso",
                        "size": 5_000_000_000,
                        "browser_download_url": "https://example.invalid/amd64-11.iso",
                    },
                    {
                        "name": "spaced-linux-11.26.1-arm64.iso",
                        "size": 5_000_000_000,
                        "browser_download_url": "https://example.invalid/arm64-11.iso",
                    },
                ],
            },
            {
                "tag_name": "v10.26.9",
                "body": "",
                "assets": [
                    {
                        "name": "spaced-linux-10.26.9-amd64.iso",
                        "size": 4_000_000_000,
                        "browser_download_url": "https://example.invalid/amd64-10",
                    },
                    {
                        "name": "spaced-linux-10.26.9-arm64.iso",
                        "size": 4_000_000_000,
                        "browser_download_url": "https://example.invalid/arm64-10",
                    },
                ],
            },
        ]
    )


class _Response:
    """A canned download response: hands back one chunk per read()."""

    def __init__(self, chunks: list[bytes], status: int = 200, after=None):
        self.chunks = list(chunks)
        self.status = status
        self.after = after

    def read(self, size: int = -1) -> bytes:
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if self.after is not None:
            self.after(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _Opener:
    """A fake urlopen; `requests` shows what the download asked for."""

    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.requests: list = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return self.responses.pop(0)


def test_pick_iso_uses_the_newest_release_and_the_requested_arch() -> None:
    assert pick_iso(_release_json(), "amd64")[:3] == (
        "spaced-linux-11.26.1-amd64.iso",
        "https://example.invalid/amd64-11.iso",
        5_000_000_000,
    )
    assert pick_iso(_release_json(), "arm64")[0] == "spaced-linux-11.26.1-arm64.iso"
    assert pick_iso(_release_json())[0] == "spaced-linux-11.26.1-amd64.iso"


def test_pick_iso_says_no_when_there_is_nothing_to_download() -> None:
    assert pick_iso("not json") == ("", "", 0, "")
    assert pick_iso("") == ("", "", 0, "")
    assert pick_iso(_release_json(), "i386") == ("", "", 0, "")
    assert pick_iso(json.dumps([{"tag_name": "v11.26.1", "assets": []}])) == ("", "", 0, "")


def test_pick_iso_reads_a_digest_the_release_states() -> None:
    digest = "ab" * 32
    single = json.dumps(
        [
            {
                "tag_name": "v11.26.1",
                "body": f"sha256 of spaced-linux-11.26.1-amd64.iso: {digest}",
                "assets": [
                    {
                        "name": "spaced-linux-11.26.1-amd64.iso",
                        "size": 40,
                        "browser_download_url": "https://example.invalid/a.iso",
                    }
                ],
            }
        ]
    )
    assert pick_iso(single, "amd64")[3] == digest
    assert pick_iso(_release_json(), "amd64")[3] == ""


def test_fetch_file_verifies_the_sha256_and_reports_progress(tmp_path) -> None:
    body = b"spaced linux iso bytes" * 64
    digest = hashlib.sha256(body).hexdigest()
    dest = tmp_path / "spaced-linux-11.26.1-amd64.iso"
    reports: list[tuple[int, int]] = []
    opener = _Opener([_Response([body[:1000], body[1000:]])])

    path = fetch_file(
        "https://example.invalid/a.iso",
        dest,
        digest,
        lambda done, total: reports.append((done, total)),
        None,
        opener=opener,
        total=len(body),
    )

    assert path == dest
    assert dest.read_bytes() == body
    assert not dest.with_name(dest.name + ".part").exists()
    assert reports and reports[-1] == (len(body), len(body))
    assert all(done <= total for done, total in reports)


def test_fetch_file_resumes_with_a_range_header_after_a_cancel(tmp_path) -> None:
    body = b"abcdefghij" * 4
    dest = tmp_path / "spaced-linux-11.26.1-amd64.iso"
    part = dest.with_name(dest.name + ".part")
    cancel = threading.Event()
    opener = _Opener(
        [
            _Response([body[:10], body[10:20]], after=lambda chunk: cancel.set()),
            _Response([body[10:]], status=206),
        ]
    )

    with pytest.raises(DownloadCancelled):
        fetch_file("https://example.invalid/a.iso", dest, "", None, cancel, opener=opener, total=len(body))
    assert part.read_bytes() == body[:10]
    assert not dest.exists()

    cancel.clear()
    assert fetch_file("https://example.invalid/a.iso", dest, "", None, cancel, opener=opener, total=len(body)) == dest
    assert opener.requests[1].get_header("Range") == "bytes=10-"
    assert dest.read_bytes() == body


def test_fetch_file_refuses_a_download_that_does_not_match(tmp_path) -> None:
    body = b"spaced linux iso bytes" * 64
    dest = tmp_path / "spaced-linux-11.26.1-amd64.iso"
    opener = _Opener([_Response([body])])

    with pytest.raises(DownloadError) as raised:
        fetch_file("https://example.invalid/a.iso", dest, "0" * 64, None, None, opener=opener, total=len(body))

    assert "did not match the checksum" in str(raised.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_free_bytes_reports_the_file_system_or_none() -> None:
    assert free_bytes(".") is None or free_bytes(".") > 0
    assert free_bytes("/definitely-not-here-voxa") is None


def _tool_env(monkeypatch, payload: str, free: int | None) -> _Opener:
    """Patch the release lookup and the free-space check; return the fake opener."""
    monkeypatch.setattr(_sysupdate_module(), "fetch", lambda url: payload)
    monkeypatch.setattr("voxa.agent.isoimage.free_bytes", lambda path: free)
    return _Opener([])


def test_the_tool_checks_free_space_before_touching_the_network(tmp_path, monkeypatch) -> None:
    looked_up: list[str] = []

    def boom(url: str) -> str:
        looked_up.append(url)
        raise AssertionError("the network must not be touched")

    monkeypatch.setattr(_sysupdate_module(), "fetch", boom)
    monkeypatch.setattr("voxa.agent.isoimage.free_bytes", lambda path: 1024)
    opener = _Opener([])

    result = download_iso({"folder": str(tmp_path)}, opener=opener)

    assert not result.ok
    assert "free disk space" in result.speech
    assert "there is 0.0 GB" in result.speech
    assert looked_up == []
    assert opener.requests == []
    assert list(tmp_path.iterdir()) == []


def test_the_tool_refuses_when_the_iso_will_not_fit(tmp_path, monkeypatch) -> None:
    _tool_env(monkeypatch, _release_json(), int(2e9))
    opener = _Opener([_Response([b"x" * 40])])

    result = download_iso({"folder": str(tmp_path)}, opener=opener)

    assert not result.ok
    assert result.speech == (
        "The spaced-linux-11.26.1-amd64.iso download needs about 5.0 GB of free disk space, "
        "and there is 2.0 GB."
    )
    assert opener.requests == []


def test_the_tool_says_honestly_when_no_checksum_is_published(tmp_path, monkeypatch) -> None:
    body = b"spaced linux iso bytes"
    payload = json.dumps(
        [
            {
                "tag_name": "v11.26.1",
                "body": "",
                "assets": [
                    {
                        "name": "spaced-linux-11.26.1-amd64.iso",
                        "size": len(body),
                        "browser_download_url": "https://example.invalid/a.iso",
                    }
                ],
            }
        ]
    )
    _tool_env(monkeypatch, payload, int(50e9))
    opener = _Opener([_Response([body])])

    result = download_iso({"folder": str(tmp_path)}, opener=opener)

    assert result.ok
    assert "The release publishes no checksum, so I could not verify the download." in result.speech
    assert str(tmp_path / "spaced-linux-11.26.1-amd64.iso") in result.speech
    assert (tmp_path / "spaced-linux-11.26.1-amd64.iso").read_bytes() == body
    assert "Next:" in result.speech


def test_the_tool_reports_a_checksum_mismatch_without_deleting_anything_else(tmp_path, monkeypatch) -> None:
    body = b"spaced linux iso bytes"
    payload = json.dumps(
        [
            {
                "tag_name": "v11.26.1",
                "body": f"spaced-linux-11.26.1-amd64.iso sha256 {'0' * 64}",
                "assets": [
                    {
                        "name": "spaced-linux-11.26.1-amd64.iso",
                        "size": len(body),
                        "browser_download_url": "https://example.invalid/a.iso",
                    }
                ],
            }
        ]
    )
    _tool_env(monkeypatch, payload, int(50e9))
    opener = _Opener([_Response([body])])

    result = download_iso({"folder": str(tmp_path)}, opener=opener)

    assert not result.ok
    assert "did not match the checksum" in result.speech
    assert list(tmp_path.iterdir()) == []


def test_the_tool_keeps_the_partial_file_when_the_download_is_cancelled(tmp_path, monkeypatch) -> None:
    body = b"spaced linux iso bytes"
    payload = json.dumps(
        [
            {
                "tag_name": "v11.26.1",
                "body": "",
                "assets": [
                    {
                        "name": "spaced-linux-11.26.1-amd64.iso",
                        "size": len(body),
                        "browser_download_url": "https://example.invalid/a.iso",
                    }
                ],
            }
        ]
    )
    _tool_env(monkeypatch, payload, int(50e9))
    cancel = threading.Event()
    opener = _Opener([_Response([body[:10], body[10:]], after=lambda chunk: cancel.set())])

    result = download_iso({"folder": str(tmp_path)}, opener=opener, cancel=cancel)

    assert not result.ok
    assert "partial file is kept" in result.speech
    assert (tmp_path / "spaced-linux-11.26.1-amd64.iso.part").read_bytes() == body[:10]


def test_the_tool_says_when_there_is_nothing_to_download(tmp_path, monkeypatch) -> None:
    _tool_env(monkeypatch, "not json", int(50e9))
    assert "Spaced Linux release" in download_iso({"folder": str(tmp_path)}).speech

    _tool_env(monkeypatch, _release_json(), int(50e9))
    assert download_iso({"folder": str(tmp_path), "arch": "i386"}).speech.endswith("publishes no ISO image for i386.")


def test_the_tool_is_registered_with_a_long_timeout() -> None:
    tool = default_registry().get("download_iso")
    assert tool.risk == 1
    assert tool.timeout_seconds == 3600.0
    assert set(tool.parameters) == {"arch", "folder"}


def test_iso_phrases_route_to_download_iso() -> None:
    assert route("download the latest spaced iso") == ToolCall("download_iso", {"arch": "amd64"})
    assert route("get the newest spaced linux image") == ToolCall("download_iso", {"arch": "amd64"})
    assert route("fetch the amd64 iso") == ToolCall("download_iso", {"arch": "amd64"})
    assert route("download the arm64 iso into my downloads folder") == ToolCall(
        "download_iso", {"arch": "arm64", "folder": "my downloads folder"}
    )


def test_updating_the_system_still_routes_to_update_system() -> None:
    assert route("update my system") == ToolCall("update_system", {})
    assert route("update spaced linux") == ToolCall("update_system", {})
    assert route("download the image") is None
