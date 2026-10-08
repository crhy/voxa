from __future__ import annotations

HEAVY_ON_CPU = ("small", "medium", "large-v3", "turbo")


def _display(model: str) -> str:
    return model[:1].upper() + model[1:]


def recommend_whisper(gpu_usable: bool, cpu_threads: int, ram_gb: float | None) -> str:
    if gpu_usable:
        return "turbo"
    if cpu_threads < 4 or (ram_gb is not None and ram_gb < 4):
        return "tiny"
    return "base"


def is_too_heavy(model: str, gpu_usable: bool) -> bool:
    return not gpu_usable and model in HEAVY_ON_CPU


def explain(model: str, gpu_usable: bool, cpu_threads: int) -> str:
    name = _display(model)
    if gpu_usable:
        return f"This computer has a graphics card Voxa can use, so {name} runs comfortably."
    if cpu_threads < 4:
        return (
            f"This computer has no graphics card Voxa can use and few processor threads, "
            f"so {name} is the model that keeps up."
        )
    return f"This computer has no graphics card Voxa can use, so {name} is the model that keeps up."


def spoken_note(old: str, new: str) -> str:
    return f"I switched my hearing to the {_display(new)} model, which this computer runs comfortably."


def detect() -> tuple[bool, int, float | None]:
    import os

    try:
        cpu_threads = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_threads = os.cpu_count() or 4
    from .hardware import detect_system_ram_gb
    from .transcription import _cuda_compute_usable

    try:
        gpu_usable = bool(_cuda_compute_usable())
    except Exception:
        gpu_usable = False
    return gpu_usable, cpu_threads, detect_system_ram_gb()
