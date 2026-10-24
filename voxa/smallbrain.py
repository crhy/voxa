"""Download Voxa's default GGUF brain into the models folder, checking it."""

from __future__ import annotations

import hashlib
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import engine

BLOCK = 1024 * 1024


class DownloadError(Exception):
    """A download that cannot be trusted or completed."""


class DownloadCancelled(Exception):
    """The user stopped a download; the partial file is kept for resuming."""


@dataclass(frozen=True)
class ModelFile:
    label: str
    repo: str
    revision: str
    filename: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


SMALL_BRAIN = ModelFile(
    "Small brain (Qwen3.5 0.8B)",
    "unsloth/Qwen3.5-0.8B-GGUF",
    "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    532517120,
    "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
)

LIGHTEST_BRAIN = ModelFile(
    "Lightest brain (Qwen2.5 0.5B)",
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "9217f5db79a29953eb74d5343926648285ec7e67",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    491400032,
    "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
)

BRAINS = (SMALL_BRAIN, LIGHTEST_BRAIN)


def path_for(model: ModelFile, directory: Path | None = None) -> Path:
    return (directory or engine.models_dir()) / model.filename


def is_installed(model: ModelFile, directory: Path | None = None) -> bool:
    path = path_for(model, directory)
    return path.is_file() and path.stat().st_size == model.bytes


def row_state(model: ModelFile, in_use_path: str) -> str:
    """How a brain row reads in the manager: "in use", "downloaded", or "missing".

    Pure so it can be tested without a window; ``in_use_path`` is the configured
    ``llamacpp_model`` (empty when none is selected)."""
    if in_use_path and in_use_path == str(path_for(model)):
        return "in use"
    if is_installed(model):
        return "downloaded"
    return "missing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    model: ModelFile,
    on_progress: Callable[[int, int], None],
    cancel: threading.Event,
    directory: Path | None = None,
    opener=urllib.request.urlopen,
) -> Path:
    directory = directory or engine.models_dir()
    final = path_for(model, directory)
    part = final.with_name(final.name + ".part")
    total = model.bytes

    resume_from = part.stat().st_size if part.is_file() else 0
    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(model.url, headers=headers)

    try:
        response = opener(request)
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError("The download failed; please try again.") from exc

    with response:
        status = getattr(response, "status", 200)
        if resume_from and status != 206:
            resume_from = 0
        done = resume_from
        last_report = 0.0
        mode = "ab" if resume_from else "wb"
        try:
            with part.open(mode) as handle:
                while True:
                    if cancel.is_set():
                        raise DownloadCancelled()
                    chunk = response.read(BLOCK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if cancel.is_set():
                        raise DownloadCancelled()
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        on_progress(done, total)
                        last_report = now
            on_progress(done, total)
        except DownloadCancelled:
            raise
        except OSError as exc:
            raise DownloadError("The download failed; please try again.") from exc

    if part.stat().st_size != model.bytes or _sha256(part) != model.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError("The download was damaged; please try again.")

    part.replace(final)
    return final
