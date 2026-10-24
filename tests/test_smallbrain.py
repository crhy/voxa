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

    def __call__(self, request):
        rng = request.get_header("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        return FakeResponse(self.full[start:], self.status)


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
