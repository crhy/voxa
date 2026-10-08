from __future__ import annotations

import time
import wave
from pathlib import Path

HEAVY_ON_CPU = ("small", "medium", "large-v3", "turbo")

BUDGET_RTF = 0.5  # leave half the CPU for the wake listener, the voice, the face and a local model
COST_VS_TINY = {"tiny": 1.0, "base": 1.9, "small": 6.3, "medium": 19.7, "large-v3": 39.7, "turbo": 20.7}
# PROVISIONAL: parameter-count ratios (39M, 74M, 244M, 769M, 1550M, 809M). Measured values replace them.
ACCURACY_ORDER = ("tiny", "base", "small", "turbo")  # candidates, least to most accurate

BENCHMARK_CLIP = Path(__file__).resolve().parent / "ui" / "assets" / "benchmark-speech.wav"


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


def clip_pcm() -> tuple[bytes, float]:
    """The bundled benchmark clip as int16 bytes and its length in seconds."""
    with wave.open(str(BENCHMARK_CLIP), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError("benchmark clip must be mono 16-bit")
        frames = reader.readframes(reader.getnframes())
        return frames, reader.getnframes() / reader.getframerate()


def measure_rtf(transcribe, pcm: bytes, seconds: float, runs: int = 2, clock=time.perf_counter) -> float:
    """Real-time factor of the transcriber on the clip: the FASTEST run's seconds per second of speech."""
    best: float | None = None
    for _ in range(runs):
        start = clock()
        transcribe(pcm)
        elapsed = clock() - start
        if best is None or elapsed < best:
            best = elapsed
    return best / seconds


def predicted_rtf(tiny_rtf: float, measured: dict[str, float]) -> dict[str, float]:
    """Predicted RTF per model: measured values win, others scale tiny's time by the cost table."""
    return {
        model: measured[model] if model in measured else tiny_rtf * cost
        for model, cost in COST_VS_TINY.items()
    }


def choose(tiny_rtf: float, measured: dict[str, float], gpu_usable: bool, budget: float = BUDGET_RTF) -> str:
    """Most accurate candidate whose predicted RTF fits the budget; the benchmark is for CPU-only boxes."""
    if gpu_usable:
        return recommend_whisper(True, 0, None)
    predictions = predicted_rtf(tiny_rtf, measured)
    chosen = "tiny"
    for model in ACCURACY_ORDER:
        if predictions[model] <= budget:
            chosen = model
    return chosen


def summary(model: str, rtf: float) -> str:
    seconds = 10.0 * rtf
    text = f"{seconds:.1f}".rstrip("0").rstrip(".")
    return f"{_display(model)} transcribes 10 seconds of speech in about {text} seconds on this computer."
