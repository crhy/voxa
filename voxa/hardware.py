from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ModelSuggestion:
    name: str
    approx_gb: float
    description: str
    # Curated entries are the deliberately chosen ladder in MODEL_CATALOG;
    # catalog.discover_catalog() marks anything it finds in the library as
    # False so a discovered model cannot quietly displace a chosen one.
    curated: bool = True


# Approximate default-quantization weight size for each tag, in GB, smallest
# first. Used only to rank suggestions, not as an exact figure, and kept at or
# just above the real size so the headroom rule below stays conservative.
# catalog.refresh_catalog() re-measures these against Ollama's registry.
MODEL_CATALOG: tuple[ModelSuggestion, ...] = (
    ModelSuggestion("qwen2.5:0.5b", 0.4, "Fastest, runs on almost anything"),
    ModelSuggestion("gemma3:1b", 0.8, "Very small, still writes coherently"),
    ModelSuggestion("qwen2.5:1.5b", 1.0, "Very fast, good for low-memory devices"),
    ModelSuggestion("llama3.2:3b", 2.0, "Good balance for laptops without a GPU"),
    ModelSuggestion("qwen3.5:2b", 2.6, "Newer generation, small enough for a laptop"),
    ModelSuggestion("qwen3.5:4b", 3.2, "Newer generation, good on modest hardware"),
    ModelSuggestion("qwen2.5:7b", 4.7, "Strong general-purpose model"),
    ModelSuggestion("llama3.1:8b", 4.9, "Strong general-purpose model"),
    ModelSuggestion("qwen3.5:9b", 6.2, "Newer generation, strong for its size"),
    ModelSuggestion("qwen2.5:14b", 9.0, "Noticeably smarter, wants a mid-range GPU"),
    ModelSuggestion("qwen3.8:27b", 15.7, "Latest generation, wants a 24GB GPU"),
    ModelSuggestion("qwen2.5:32b", 20.0, "High quality, wants a 24GB+ GPU"),
    ModelSuggestion("qwen3.5:35b", 22.3, "Newer generation at the top end, wants 30GB+"),
    ModelSuggestion("llama3.1:70b", 40.0, "Top quality, wants multiple GPUs or a lot of unified memory"),
)

# Extra headroom beyond raw model weights for KV cache, activations, and the
# rest of the OS/desktop, so a suggestion isn't a model that merely fits on
# disk but chokes the moment inference starts.
# How far below the largest model that fits a curated one may be and still
# be preferred to it.
CURATED_PREFERENCE_BAND = 0.10

_HEADROOM_FACTOR = 1.3
_HEADROOM_FLOOR_GB = 1.0

# Measured on this machine: what the High face server and Whisper-on-GPU hold
# on the card for themselves, which model suggestions must keep out of the way.
FACE_SERVER_VRAM_GB = 4.8
WHISPER_VRAM_GB = 2.4


def reserved_gpu_gb(face_mode: str, whisper_on_gpu: bool) -> float:
    """VRAM the face server and the speech model keep for themselves on the GPU."""
    face = FACE_SERVER_VRAM_GB if face_mode == "live" else 0.0
    whisper = WHISPER_VRAM_GB if whisper_on_gpu else 0.0
    return face + whisper


def _fits(available_gb: float, model: ModelSuggestion) -> bool:
    return available_gb >= model.approx_gb * _HEADROOM_FACTOR + _HEADROOM_FLOOR_GB


# The only Ollama brains a machine without a graphics card is offered. Keep
# these names in the Ollama catalog; bundled GGUF brains are managed separately.
SMALL_BRAIN_NAMES = ("qwen2.5:0.5b", "gemma3:1b", "qwen2.5:1.5b")


def no_gpu_suggestions(
    catalog: tuple[ModelSuggestion, ...] | None = None,
) -> list[ModelSuggestion]:
    """The small brains to offer a machine with no GPU, in SMALL_BRAIN_NAMES order.

    Falls back to the smallest catalog entry when the catalog holds neither name,
    so the summary is never empty."""
    entries = catalog or MODEL_CATALOG
    by_name = {model.name: model for model in entries}
    found = [by_name[name] for name in SMALL_BRAIN_NAMES if name in by_name]
    return found or [entries[0]]


def suggest_models(
    available_gb: float,
    *,
    limit: int = 3,
    catalog: tuple[ModelSuggestion, ...] | None = None,
    reserved_gb: float = 0.0,
) -> list[ModelSuggestion]:
    """Return up to ``limit`` catalog models that fit in ``available_gb`` minus
    ``reserved_gb`` (VRAM other components keep), best first."""
    entries = catalog or MODEL_CATALOG
    usable_gb = max(available_gb - reserved_gb, 0.0)
    fitting = [model for model in entries if _fits(usable_gb, model)]
    if not fitting:
        return [entries[0]]
    ranked = sorted(fitting, key=lambda model: model.approx_gb, reverse=True)
    best = ranked[0]
    if not best.curated:
        # Size stands in for quality, which is fine for a handful of chosen
        # models and poor once discovery fills the list with near-identical
        # ones: a tenth of a gigabyte should not decide that an automatically
        # found model beats a deliberately chosen one. A curated model within
        # a short reach of the largest that fits takes the top slot instead.
        reach = best.approx_gb * (1 - CURATED_PREFERENCE_BAND)
        preferred = next(
            (model for model in ranked if model.curated and model.approx_gb >= reach), None
        )
        if preferred is not None:
            ranked.remove(preferred)
            ranked.insert(0, preferred)
    return ranked[:limit]


def detect_gpu_vram_gb(sysfs_base: Path = Path("/sys/class/drm")) -> float | None:
    """Best-effort total VRAM in GB for the most capable GPU, or None if undetectable.

    Reads the per-card listing below and takes the largest card, which is what
    this figure always meant: the most VRAM a single model can be sized against.
    """
    totals = [
        device.total_vram_gb
        for device in detect_gpu_devices(sysfs_base)
        if device.total_vram_gb is not None
    ]
    return max(totals) if totals else None


# Inside the Flatpak the host does not expose nvidia-smi, so hardware.py
# prefers the copy bundled at /app/lib/nvml when present and falls back to
# whatever ``nvidia-smi`` is on the PATH (the non-Flatpak dev case).
_BUNDLED_NVIDIA_SMI = Path("/app/lib/nvml/nvidia-smi")


def _nvidia_smi_command() -> str:
    return str(_BUNDLED_NVIDIA_SMI) if _BUNDLED_NVIDIA_SMI.exists() else "nvidia-smi"


def _nvidia_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            [_nvidia_smi_command(), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except ValueError:
        return None
    if not values:
        return None
    return max(values) / 1024.0


def _rocm_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    totals = [int(value) for value in re.findall(r'"VRAM Total Memory \(B\)":\s*"?(\d+)"?', result.stdout)]
    if not totals:
        return None
    return max(totals) / (1024.0**3)


def _sysfs_amdgpu_vram_gb(base: Path = Path("/sys/class/drm")) -> float | None:
    """Largest card readable straight from sysfs, for a machine with no GPU tool."""
    totals = [device.total_vram_gb for device in _sysfs_amdgpu_devices(base) if device.total_vram_gb is not None]
    return max(totals) if totals else None


# --- Which card to put Voxa's brain on (crhy/voxa#113) --------------------
#
# Everything above collapses the machine's cards into one number, which is
# enough to size a model but useless for choosing a card: with two cards
# "the GPU" is ambiguous, and #113 is precisely about Voxa taking a card the
# other workload is not holding. ``detect_gpu_devices`` keeps them apart, one
# entry per card, with the index each backend addresses it by. The single-value
# probes above stay in place (now reading the same listing) so the model
# suggestions, the window's gauge and their tests behave as they always did.


@dataclass(slots=True, frozen=True)
class GpuDevice:
    """One graphics card: how the backends name it, how big it is, how busy."""

    index: int
    # "nvidia", "amd", "intel", or "gpu" when the vendor cannot be read.
    vendor: str
    name: str
    total_vram_gb: float | None
    used_vram_gb: float | None
    utilization_percent: float | None
    # What ``Settings.gpu_device`` stores and what the status line says out loud.
    # The vendor is part of it because GPU 0 on the Nvidia side and GPU 0 on the
    # AMD side are different cards, and a saved choice has to survive a reboot
    # into the other tool's listing.
    device_id: str


def _gpu_device(
    vendor: str,
    index: int,
    name: str,
    total_gb: float | None = None,
    used_gb: float | None = None,
    utilization: float | None = None,
) -> GpuDevice:
    """Build one card entry; the id is always derived, never typed by hand."""
    return GpuDevice(
        index=index,
        vendor=vendor,
        name=name,
        total_vram_gb=total_gb,
        used_vram_gb=used_gb,
        utilization_percent=utilization,
        device_id=f"{vendor}:{index}",
    )


def _number_or_none(text: object) -> float | None:
    """Parse one figure out of a GPU tool's output.

    ``nvidia-smi`` writes "[N/A]" for memory on cards it cannot read, and
    ``rocm-smi`` quotes its byte counts, so anything that is not a number
    becomes None rather than a zero: a card of unknown size must not look
    empty, it must look unknown.
    """
    try:
        return float(str(text).strip().strip("[]"))
    except (TypeError, ValueError):
        return None


# sysfs vendor ids, mapped to the vendor token ``device_id`` uses.
_DRM_VENDORS = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel"}
# Vendor token to the words the app shows (and ``detect_graphics_adapter`` returns).
_GPU_VENDOR_NAMES = {"amd": "AMD GPU", "nvidia": "NVIDIA GPU", "intel": "Intel integrated GPU"}


def _vendor_words(vendor: str) -> str:
    return _GPU_VENDOR_NAMES.get(vendor, "GPU")


def _nvidia_devices() -> list[GpuDevice]:
    """One line per card from nvidia-smi: index, name, memory total/used, use %."""
    try:
        result = subprocess.run(
            [
                _nvidia_smi_command(),
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[GpuDevice] = []
    for position, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        if len(parts) >= 5:
            # The three figures are read from the end of the line because a
            # card name containing a comma shifts everything after it.
            index_text, name = parts[0], ", ".join(parts[1:-3]) or "GPU"
            total_mib, used_mib, utilization = parts[-3:]
        elif len(parts) == 1:
            # A one-figure answer is this module's older memory.total reading
            # (``_nvidia_vram_gb`` asks for that field alone), so the figure is
            # the card's total and its position in the listing is its index.
            index_text, name, total_mib, used_mib, utilization = str(position), "GPU", parts[0], "", ""
        else:
            continue
        try:
            index = int(index_text)
        except ValueError:
            continue
        total_gb = _number_or_none(total_mib)
        used_gb = _number_or_none(used_mib)
        utilization_percent = _number_or_none(utilization)
        devices.append(
            _gpu_device(
                "nvidia",
                index,
                name,
                total_gb / 1024.0 if total_gb is not None else None,
                used_gb / 1024.0 if used_gb is not None else None,
                utilization_percent,
            )
        )
    return devices


def _rocm_devices() -> list[GpuDevice]:
    """One entry per card from rocm-smi's VRAM JSON, in the order it lists them."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    devices: list[GpuDevice] = []
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        total_b = _number_or_none(entry.get("VRAM Total Memory (B)"))
        used_b = _number_or_none(entry.get("VRAM Total Used Memory (B)"))
        devices.append(
            _gpu_device(
                "amd",
                len(devices),
                "AMD GPU",
                total_b / (1024.0**3) if total_b is not None else None,
                used_b / (1024.0**3) if used_b is not None else None,
            )
        )
    return devices


def _drm_card_number(path: Path) -> int | None:
    """The card number of a sysfs entry, or None for the display links.

    ``/sys/class/drm`` holds ``card0`` next to ``card0-HDMI-1`` entries that
    point at the same card; counting those twice would invent a card.
    """
    match = re.fullmatch(r"card(\d+)", path.name)
    return int(match.group(1)) if match else None


def _sysfs_amdgpu_devices(base: Path = Path("/sys/class/drm")) -> list[GpuDevice]:
    """Cards read straight from sysfs when no GPU tool is installed.

    The index is the card's position in this listing, which is the order the
    backends number their own cards in; the card number in the path only keeps
    the listing stable.
    """
    numbered = [(number, card) for card in base.glob("card*") if (number := _drm_card_number(card)) is not None]
    devices: list[GpuDevice] = []
    for _, card in sorted(numbered, key=lambda pair: pair[0]):
        device_dir = card / "device"
        vendor = _DRM_VENDORS.get(_read_sysfs(device_dir / "vendor").strip().lower(), "gpu")
        name = _read_sysfs(device_dir / "name").strip() or _vendor_words(vendor)
        total_b = _number_or_none(_read_sysfs(device_dir / "mem_info_vram_total"))
        used_b = _number_or_none(_read_sysfs(device_dir / "mem_info_vram_used"))
        devices.append(
            _gpu_device(
                vendor,
                len(devices),
                name,
                total_b / (1024.0**3) if total_b is not None else None,
                used_b / (1024.0**3) if used_b is not None else None,
            )
        )
    return devices


def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


def detect_gpu_devices(sysfs_base: Path = Path("/sys/class/drm")) -> list[GpuDevice]:
    """Every graphics card this machine has, best effort, never raising.

    Returns ``[]`` when nothing is found, so the caller can say honestly that
    the machine has no card rather than guessing one. Each probe is a single
    subprocess call (or a directory walk), cheap enough to re-run when the
    choice has to be made again.
    """
    for probe in (_nvidia_devices, _rocm_devices):
        devices = probe()
        if devices:
            return devices
    return _sysfs_amdgpu_devices(sysfs_base)


def _free_vram_gb(device: GpuDevice, reserved_gb: float) -> float | None:
    """VRAM left on a card once what is loaded on it and ``reserved_gb`` are
    subtracted; None when the card's size cannot be read at all."""
    if device.total_vram_gb is None:
        return None
    return device.total_vram_gb - (device.used_vram_gb or 0.0) - reserved_gb


def pick_brain_device(
    devices: list[GpuDevice],
    *,
    reserved_gb: float,
    busy: set[str] | None = None,
) -> tuple[GpuDevice | None, str]:
    """Choose the card Voxa's brain runs on, and say why in plain words.

    ``reserved_gb`` comes from ``reserved_gpu_gb``: what the High face server
    and Whisper keep for themselves. ``busy`` names the cards another workload
    is holding - the big coding model rhY leaves running while Voxa works - and
    those are skipped so Voxa's small brain lands on a card of its own.

    The reason is returned because it is spoken out loud and because a wrong
    guess has to stay visible: a machine with one card cannot be split, and
    saying so is better than pretending the split happened.
    """
    devices = list(devices)
    busy = busy or set()
    if not devices:
        return None, "this machine has no graphics card, so Voxa's brain runs on the processor"

    if len(devices) == 1:
        return devices[0], "this machine has one GPU, Voxa will share it"

    free = {device.device_id: _free_vram_gb(device, reserved_gb) for device in devices}
    free_cards = [device for device in devices if device.device_id not in busy]
    if not free_cards:
        return None, "every graphics card here is already held by another workload, so Voxa has none to take"

    measured = [device for device in free_cards if free[device.device_id] is not None]
    if not measured:
        # Nothing about the cards' memory is readable: take the first free one
        # and say the choice is a guess instead of implying it was measured.
        chosen = free_cards[0]
        return chosen, f"GPU {chosen.index} is nobody else's card, but this machine's memory figures could not be read"

    fitting = [device for device in measured if free[device.device_id] > 0]
    if not fitting:
        return None, (
            "no card has room left once the face and the speech model keep their share, "
            "so Voxa's brain stays off the graphics cards"
        )

    chosen = max(fitting, key=lambda device: (free[device.device_id], -device.index))
    parts = [f"GPU {chosen.index} is the card nobody is using with the most room left"]
    held = [device for device in devices if device.device_id in busy]
    if held:
        parts.append(f"GPU {held[0].index} is left to the other workload")
    room = free[chosen.device_id]
    parts.append(f"{room:.1f} of {chosen.total_vram_gb:.0f} GB free on it")
    return chosen, "; ".join(parts)


# A card holding more than this much memory belongs to another workload - the
# big coding model rhY leaves running while Voxa works. It is a reading, not
# a certainty: the same figure covers Voxa's own model once she has started,
# which is why moving her off a card treats the card she is on as free.
BUSY_VRAM_GB = 1.0


def busy_devices(devices: list[GpuDevice]) -> set[str]:
    """The ids of cards that already hold a workload, for ``pick_brain_device``.

    Only cards with a memory reading can be judged: a machine whose tools
    report no usage has nothing to skip, which beats inventing a busy card.
    """
    return {device.device_id for device in devices if (device.used_vram_gb or 0.0) >= BUSY_VRAM_GB}


def parse_device_id(value: str) -> tuple[str, int] | None:
    """Split a saved card id ("nvidia:1") into vendor token and index.

    None for "auto", "cpu" and for anything malformed, which is how a stale
    or hand-edited setting is recognised without a second table of rules.
    """
    vendor, separator, index_text = str(value).strip().casefold().partition(":")
    if not separator or not vendor.isalnum() or not index_text.isdigit() or vendor in {"cpu", "auto"}:
        return None
    return vendor, int(index_text)


def resolve_brain_device(
    saved: str,
    devices: list[GpuDevice],
    *,
    reserved_gb: float,
    busy: set[str] | None = None,
) -> tuple[str, str]:
    """Turn the saved choice into what to launch with, plus the reason said aloud.

    This is the one place that knows what Voxa will really do, so both the
    launch (``server.py``) and the spoken status ask it. A saved card that is
    not in this machine today becomes "auto" rather than failing to start, and
    the reason says the saved card was not found: hardware changes between
    launches, so the choice is re-read every time instead of being repaired
    in the settings.

    "auto" launches with no card named at all - the backend keeps its own
    pick - but the reason still says which card ``pick_brain_device`` would
    take, because that is the card Voxa moves to when the user says to put
    her on the other one.
    """
    choice = str(saved or "").strip().casefold()
    if choice == "cpu":
        if not devices:
            return "cpu", "you asked for the processor, and this machine has no graphics card anyway"
        return "cpu", "you asked Voxa to keep her brain on the processor"

    parsed = parse_device_id(choice)
    if parsed is not None:
        wanted = f"{parsed[0]}:{parsed[1]}"
        found = next((device for device in devices if device.device_id == wanted), None)
        if found is not None:
            return found.device_id, f"you chose GPU {found.index}, a {_vendor_words(found.vendor)} card"
        return "auto", f"the saved card {wanted} is not in this machine today, so the backend picks the card"

    chosen, reason = pick_brain_device(devices, reserved_gb=reserved_gb, busy=busy)
    return "auto", reason


def describe_device(device: GpuDevice | None) -> str:
    """One card, in the words used in the status line and the logs."""
    if device is None:
        return "no graphics card"
    parts = [f"GPU {device.index}", device.name or _vendor_words(device.vendor)]
    if device.total_vram_gb is None:
        parts.append("memory unknown")
    else:
        parts.append(f"{device.total_vram_gb:.0f} GB")
        used = device.used_vram_gb
        parts.append(f"{used:.1f} GB in use" if used else "nothing on it")
    return ", ".join(parts)


def detect_system_ram_gb(meminfo_path: Path = Path("/proc/meminfo")) -> float | None:
    try:
        with meminfo_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return kib / (1024.0**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def detect_graphics_adapter(sysfs_base: Path = Path("/sys/class/drm")) -> str | None:
    """Describe a graphics adapter even when it has no dedicated VRAM counter.

    Read from sysfs only, exactly as before: this answer names the machine's
    adapter, it does not pick one, and it must not depend on a GPU tool being
    installed.
    """
    devices = _sysfs_amdgpu_devices(sysfs_base)
    return _vendor_words(devices[0].vendor) if devices else None


@dataclass(slots=True, frozen=True)
class GpuUsage:
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float


def sample_gpu_usage() -> GpuUsage | None:
    """One-shot GPU utilization + VRAM sample, or None if no GPU tool is available.

    Cheap enough to poll every second or so from a background thread — each
    call is a single subprocess invocation, not a persistent connection.
    With more than one card the busiest one is reported, which is the card the
    gauge is warning about; the single-card answer is unchanged.
    """
    usage = _busiest_device_usage(_tool_devices())
    if usage is not None:
        return usage
    for probe in (_nvidia_gpu_usage, _rocm_gpu_usage):
        usage = probe()
        if usage is not None:
            return usage
    return None


def _tool_devices() -> list[GpuDevice]:
    """The cards as a GPU tool reports them, never the sysfs fallback.

    sysfs carries no utilization, and a card read from sysfs is not a sample
    of a running workload, so the gauge keeps using the tool probes.
    """
    for probe in (_nvidia_devices, _rocm_devices):
        devices = probe()
        if devices:
            return devices
    return []


def _busiest_device_usage(devices: list[GpuDevice]) -> GpuUsage | None:
    """The card doing the most work, as the single-sample shape the gauge wants."""
    readable = [
        device
        for device in devices
        if device.utilization_percent is not None or device.used_vram_gb is not None
    ]
    if not readable:
        return None
    busiest = max(readable, key=lambda device: (device.utilization_percent or 0.0, device.used_vram_gb or 0.0))
    return GpuUsage(
        utilization_percent=busiest.utilization_percent or 0.0,
        memory_used_gb=busiest.used_vram_gb or 0.0,
        memory_total_gb=busiest.total_vram_gb or 0.0,
    )


def _nvidia_gpu_usage() -> GpuUsage | None:
    try:
        result = subprocess.run(
            [
                _nvidia_smi_command(),
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        util, used_mib, total_mib = (float(part) for part in parts)
    except ValueError:
        return None
    return GpuUsage(utilization_percent=util, memory_used_gb=used_mib / 1024.0, memory_total_gb=total_mib / 1024.0)


def _rocm_gpu_usage() -> GpuUsage | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    card = next(iter(payload.values()), None) if isinstance(payload, dict) else None
    if not isinstance(card, dict):
        return None
    try:
        util = float(card["GPU use (%)"])
        used_bytes = float(card["VRAM Total Used Memory (B)"])
        total_bytes = float(card["VRAM Total Memory (B)"])
    except (KeyError, TypeError, ValueError):
        return None
    return GpuUsage(
        utilization_percent=util,
        memory_used_gb=used_bytes / (1024.0**3),
        memory_total_gb=total_bytes / (1024.0**3),
    )


def detect_available_model_memory_gb() -> tuple[float, str]:
    """The best figure available to size a model against, and where it came from."""
    vram = detect_gpu_vram_gb()
    if vram is not None:
        return vram, "GPU VRAM"
    ram = detect_system_ram_gb()
    if ram is not None:
        adapter = detect_graphics_adapter()
        if adapter is not None:
            return ram, f"system RAM shared with {adapter}"
        return ram, "system RAM — no GPU detected"
    return 4.0, "an undetected machine (conservative default)"
