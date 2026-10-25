from __future__ import annotations

import hashlib
import threading

import pytest

from voxa.smallbrain import (
    DownloadCancelled,
    DownloadError,
    ModelFile,
    download,
    is_installed,
    path_for,
)


class FakeResponse:
    def __init__(self, data: bytes, status: int):
        self.status = status
        self.headers: dict[str, str] = {}
        self._buf = data

    def read(self, n: int) -> bytes:
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, full: bytes, status: int = 200):
        self.full = full
        self.status = status
        self.ranges: list[str | None] = []

    def __call__(self, request, *, timeout):
        assert 0 < timeout <= 30
        rng = request.get_header("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        return FakeResponse(self.full[start:] if self.status == 206 else self.full, self.status)


def make_model(data: bytes, sha: str | None = None) -> ModelFile:
    return ModelFile(
        "tiny",
        "x/y",
        "rev",
        "tiny.gguf",
        len(data),
        sha if sha is not None else hashlib.sha256(data).hexdigest(),
    )


def record():
    calls: list[tuple[int, int]] = []
    return calls, lambda done, total: calls.append((done, total))


def test_full_download(tmp_path):
    data = b"hello world brain"
    model = make_model(data)
    calls, on_progress = record()
    path = download(model, on_progress, threading.Event(), tmp_path, FakeOpener(data))
    assert path == tmp_path / "tiny.gguf"
    assert path.read_bytes() == data
    assert not (tmp_path / "tiny.gguf.part").exists()
    assert calls[-1] == (len(data), len(data))


def test_resumed_download(tmp_path):
    data = b"hello world brain"
    model = make_model(data)
    part = tmp_path / "tiny.gguf.part"
    part.write_bytes(data[:6])
    opener = FakeOpener(data, status=206)
    calls, on_progress = record()
    path = download(model, on_progress, threading.Event(), tmp_path, opener)
    assert opener.ranges == ["bytes=6-"]
    assert path.read_bytes() == data
    assert calls[-1] == (len(data), len(data))


def test_damaged_download(tmp_path):
    data = b"hello world brain"
    model = make_model(data, sha="0" * 64)
    with pytest.raises(DownloadError):
        download(model, lambda *a: None, threading.Event(), tmp_path, FakeOpener(data))
    assert not (tmp_path / "tiny.gguf.part").exists()
    assert not (tmp_path / "tiny.gguf").exists()


def test_cancel_keeps_part(tmp_path):
    data = b"hello world brain"
    model = make_model(data)
    part = tmp_path / "tiny.gguf.part"
    part.write_bytes(data[:6])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadCancelled):
        download(model, lambda *a: None, cancel, tmp_path, FakeOpener(data, status=206))
    assert part.read_bytes() == data[:6]


def test_is_installed(tmp_path):
    data = b"hello world brain"
    model = make_model(data)
    assert not is_installed(model, tmp_path)
    path_for(model, tmp_path).write_bytes(data)
    assert is_installed(model, tmp_path)
    path_for(model, tmp_path).write_bytes(data[:5])
    assert not is_installed(model, tmp_path)


def test_complete_partial_file_is_verified_without_network(tmp_path):
    data = b"complete brain awaiting rename"
    model = make_model(data)
    (tmp_path / "tiny.gguf.part").write_bytes(data)
    calls, progress = record()

    def offline(*args, **kwargs):
        raise AssertionError("a complete verified download needs no network")

    path = download(model, progress, threading.Event(), tmp_path, offline)
    assert path.read_bytes() == data
    assert calls == [(len(data), len(data))]
    assert not (tmp_path / "tiny.gguf.part").exists()


@pytest.mark.parametrize("partial", [b"wrong", b"oversized partial file"])
def test_invalid_complete_partial_file_restarts(tmp_path, partial):
    data = b"brain"
    (tmp_path / "tiny.gguf.part").write_bytes(partial)
    opener = FakeOpener(data)
    path = download(make_model(data), lambda *a: None, threading.Event(), tmp_path, opener)
    assert opener.ranges == [None]
    assert path.read_bytes() == data


def test_server_ignoring_range_replaces_partial_file(tmp_path):
    data = b"hello world brain"
    (tmp_path / "tiny.gguf.part").write_bytes(data[:6])
    opener = FakeOpener(data, status=200)
    path = download(make_model(data), lambda *a: None, threading.Event(), tmp_path, opener)
    assert opener.ranges == ["bytes=6-"]
    assert path.read_bytes() == data


def test_cancel_before_download_does_not_connect(tmp_path):
    cancel = threading.Event()
    cancel.set()

    def offline(*args, **kwargs):
        raise AssertionError("cancelled download must not connect")

    with pytest.raises(DownloadCancelled):
        download(make_model(b"brain"), lambda *a: None, cancel, tmp_path, offline)


def test_download_creates_destination_directory(tmp_path):
    data = b"brain"
    path = download(make_model(data), lambda *a: None, threading.Event(), tmp_path / "models", FakeOpener(data))
    assert path.read_bytes() == data
