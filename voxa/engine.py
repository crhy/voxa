from __future__ import annotations

import os
from pathlib import Path

from .hardware import parse_device_id

BUNDLED_SERVER = Path("/app/bin/llama-server")
ENGINE_PORT = 8781
ENGINE_URL = "http://127.0.0.1:8781"

# llama.cpp backends, keyed by the vendor token ``hardware.GpuDevice.vendor``
# uses. A card whose vendor cannot be read is left to llama.cpp's own backend
# choice rather than guessed at, and only ``--main-gpu`` pins the index.
_LLAMA_BACKENDS = {"nvidia": "cuda", "amd": "hip", "intel": "vulkan"}


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


def device_flags(device: str) -> list[str]:
    """The llama.cpp flags that put the process on one card; none for "auto" or "cpu".

    ``--device`` picks the backend for that vendor and ``--main-gpu`` the
    index inside it, so the card Voxa was chosen is the card the brain lands
    on. A card whose vendor is unknown gets only ``--main-gpu``: guessing a
    backend would be worse than letting llama.cpp pick its own.
    """
    normalized = device.strip().casefold()
    parsed = parse_device_id(normalized)
    if parsed is None:
        return []
    vendor, index = parsed
    flags: list[str] = []
    backend = _LLAMA_BACKENDS.get(vendor)
    if backend is not None:
        flags += ["--device", backend]
    return flags + ["--main-gpu", str(index)]


def server_command(model: str, host: str, port: int, *, server: str, device: str = "auto") -> list[str]:
    """Command for Voxa's own engine: all layers on the GPU when there is one, else CPU.

    ``device`` follows ``Settings.gpu_device``: "auto" (the default) keeps
    llama.cpp's own pick, exactly as before this feature; "cpu" keeps every
    layer on the processor; a card id such as "nvidia:1" names the card with
    llama.cpp's ``--device`` and ``--main-gpu`` options, which is how Voxa's
    small brain lands on a card the other workload is not holding (crhy/voxa#113).
    ``-ngl 99`` stays for every card: the point of the split is that the whole
    brain sits on that one card, not that part of it does.
    """
    layers = "0" if device.strip().casefold() == "cpu" else "99"
    return [
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
        *device_flags(device),
    ]
