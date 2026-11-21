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
from .hardware import _HEADROOM_FACTOR, _HEADROOM_FLOOR_GB

BLOCK = 1024 * 1024
DOWNLOAD_TIMEOUT = 30


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

# Pinned from Hugging Face's own tree API (size and lfs.oid per file), never
# invented, and held at the revision the digest was taken from. The ladder runs
# lightest to heaviest; anything a machine cannot hold is simply not suggested.
MEDIUM_BRAIN = ModelFile(
    "Medium brain (Qwen2.5 1.5B)",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "91cad51170dc346986eccefdc2dd33a9da36ead9",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    1117320736,
    "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
)

LARGE_BRAIN = ModelFile(
    "Large brain (Qwen3.5 4B)",
    "unsloth/Qwen3.5-4B-GGUF",
    "e87f176479d0855a907a41277aca2f8ee7a09523",
    "Qwen3.5-4B-Q4_K_M.gguf",
    2740937888,
    "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
)

LARGEST_BRAIN = ModelFile(
    "Largest brain (Qwen3.5 9B)",
    "unsloth/Qwen3.5-9B-GGUF",
    "3885219b6810b007914f3a7950a8d1b469d598a5",
    "Qwen3.5-9B-Q4_K_M.gguf",
    5680522464,
    "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
)

BRAINS = (LIGHTEST_BRAIN, SMALL_BRAIN, MEDIUM_BRAIN, LARGE_BRAIN, LARGEST_BRAIN)

# The brain Voxa selects when llama.cpp is the backend and none is configured:
# deliberately the 0.8B one, not simply the first rung of the ladder.
DEFAULT_BRAIN = SMALL_BRAIN


def path_for(model: ModelFile, directory: Path | None = None) -> Path:
    return (directory or engine.models_dir()) / model.filename


def approx_gb(model: ModelFile) -> float:
    """The pinned download size in GB, the figure the fit rule works on."""
    return model.bytes / (1024.0**3)


def fits_machine(model: ModelFile, available_gb: float) -> bool:
    """True when the brain fits with the headroom a model suggestion is held to.

    Reuses hardware.py's rule (1.3x the weights plus 1 GB) instead of inventing
    a second one, so the GGUF ladder and the Ollama catalog are sized the same
    way. Pass ``available_gb`` already reduced by whatever the face server and
    Whisper keep on the card.
    """
    return available_gb >= approx_gb(model) * _HEADROOM_FACTOR + _HEADROOM_FLOOR_GB


def brain_suggestions(
    available_gb: float,
    reserved_gb: float = 0.0,
    limit: int = 3,
) -> list[ModelFile]:
    """Up to ``limit`` GGUF brains for this machine, the largest that fits first.

    Only brains that fit are offered, and never an Ollama tag: llama.cpp thinks
    with GGUF files on disk, so a name like "qwen2.5:0.5b" is a model it cannot
    run. A machine nothing fits (no graphics card, little RAM) still gets the
    lightest brain rather than the empty list the issue reported.
    """
    usable_gb = max(available_gb - reserved_gb, 0.0)
    fitting = [model for model in BRAINS if fits_machine(model, usable_gb)]
    if not fitting:
        return [LIGHTEST_BRAIN]
    # BRAINS runs lightest first, so the largest that fits comes last.
    return list(reversed(fitting))[:limit]


def fit_note(model: ModelFile, available_gb: float, reserved_gb: float = 0.0) -> str:
    """The brains-page note saying whether a rung is worth this machine's memory.

    Pure so it can be tested without a window; ``reserved_gb`` is the VRAM the
    High face server and Whisper keep for themselves.
    """
    usable_gb = max(available_gb - reserved_gb, 0.0)
    if fits_machine(model, usable_gb):
        return "Fits this machine"
    return f"Too big for this machine (~{usable_gb:.0f} GB)"


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


def available_labels(in_use_path: str, directory: Path | None = None) -> list[str]:
    """Labels of the brains the user can pick in the model drop-down.

    A brain is offered when it is installed or already selected, so the picker
    never reads "No models available" while a brain is on disk. Pure (takes the
    directory) so it can be tested without a window."""
    directory = directory or engine.models_dir()
    labels: list[str] = []
    for model in BRAINS:
        path = path_for(model, directory)
        if (in_use_path and in_use_path == str(path)) or is_installed(model, directory):
            labels.append(model.label)
    return labels


def label_for_path(path: str) -> str:
    """The brain label for a configured ``llamacpp_model`` path ("" if none match)."""
    for model in BRAINS:
        if str(path_for(model)) == path:
            return model.label
    return ""


def path_for_label(label: str) -> str:
    """The on-disk path for a brain label ("" if no brain has that label)."""
    for model in BRAINS:
        if model.label == label:
            return str(path_for(model))
    return ""


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
    if cancel.is_set():
        raise DownloadCancelled()
    directory = directory or engine.models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    final = path_for(model, directory)
    part = final.with_name(final.name + ".part")
    total = model.bytes

    resume_from = part.stat().st_size if part.is_file() else 0
    if resume_from >= total:
        if resume_from == total and _sha256(part) == model.sha256:
            if cancel.is_set():
                raise DownloadCancelled()
            part.replace(final)
            on_progress(total, total)
            return final
        part.unlink()
        resume_from = 0
    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(model.url, headers=headers)

    try:
        response = opener(request, timeout=DOWNLOAD_TIMEOUT)
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
