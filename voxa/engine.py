from __future__ import annotations

import os
from pathlib import Path

from .config import gpu_choice_index

BUNDLED_SERVER = Path("/app/bin/llama-server")
ENGINE_PORT = 8781
ENGINE_URL = "http://127.0.0.1:8781"


def models_dir() -> Path:
    """Voxa's own GGUF folder; inside the Flatpak this is the app's writable data dir."""
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    path = base / "voxa" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_server() -> str | None:
    """The bundled llama-server path when it exists and is executable, else None."""
    if BUNDLED_SERVER.is_file() and os.access(BUNDLED_SERVER, os.X_OK):
        return str(BUNDLED_SERVER)
    return None


def server_command(model: str, host: str, port: int, *, server: str, gpu_choice: str = "auto") -> list[str]:
    """Command for Voxa's own engine: all layers on the GPU when there is one, else CPU.

    ``gpu_choice`` follows ``Settings.gpu_choice``: "auto" (the default) keeps
    llama.cpp's own pick, "cpu" pins every layer to the processor, and
    "gpu:<index>" names the card with llama.cpp's ``--device`` option. The card
    is named rather than hidden behind ``CUDA_VISIBLE_DEVICES`` because that
    variable renumbers the visible cards, which would make ``cuda:<index>``
    point at the wrong one.
    """
    layers = "0" if gpu_choice == "cpu" else "99"
    command = [
        server,
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        layers,
        "-c",
        "8192",
        "--jinja",
    ]
    index = gpu_choice_index(gpu_choice)
    if index is not None:
        command += ["--device", f"cuda:{index}"]
    return command
