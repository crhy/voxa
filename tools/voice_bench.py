#!/usr/bin/env python3
"""Benchmark local TTS engines (Piper, Kokoro-82M ONNX, espeak-ng).

Runs three benchmark texts through each engine/voice, writes sample WAVs to
.voxa-spec/out/voices/samples/<engine>-<voice>-<n>.wav, and writes a report to
.voxa-spec/out/voices/VOICES.md.

Run with the task venv:
    .voxa-spec/out/voices/venv/bin/python tools/voice_bench.py
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import soundfile
from kokoro_onnx import Kokoro
from piper.voice import PiperVoice

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".voxa-spec" / "out" / "voices"
SAMPLES = OUT / "samples"
MODELS = OUT / "models"

TEXTS = [
    "Opening Pluma.",
    "The weather in San Diego is sunny with a high of seventy two degrees.",
    "The morning light slipped through the tall window, warming the wooden "
    "floor and the quiet rows of books that lined every wall in the small "
    "reading room near the corner of the old campus library here in San "
    "Diego. Outside, birds sang softly nearby today.",
]

assert len(TEXTS[2].split()) == 45


def wav_write(path: Path, samples: np.ndarray, rate: int) -> None:
    soundfile.write(path, samples, rate)


def bench_piper(model: Path, config: Path, voice: str, text: str) -> dict:
    piper = PiperVoice.load(str(model), config_path=str(config))
    rate = piper.config.sample_rate
    tracemalloc.start()
    t0 = time.perf_counter()
    first = None
    chunks: list[np.ndarray] = []
    alignments = False
    for chunk in piper.synthesize(text, include_alignments=True):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(chunk.audio_float_array)
        if chunk.phoneme_alignments:
            alignments = True
    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return {
        "audio": audio,
        "rate": rate,
        "first": first,
        "total": total,
        "peak_mb": peak / 1e6,
        "alignments": alignments,
    }


async def _kokoro_stream(kokoro: Kokoro, text: str, voice: str):
    first = None
    chunks: list[np.ndarray] = []
    rate = 0
    t0 = time.perf_counter()
    async for samples, rate_val in kokoro.create_stream(text, voice=voice):
        rate = rate_val
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(samples)
    return first, chunks, rate, time.perf_counter() - t0


def bench_kokoro(model: Path, voices: Path, voice: str, text: str) -> dict:
    kokoro = Kokoro(str(model), str(voices))
    tracemalloc.start()
    first, chunks, rate, total = asyncio.run(_kokoro_stream(kokoro, text, voice))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return {
        "audio": audio,
        "rate": rate,
        "first": first,
        "total": total,
        "peak_mb": peak / 1e6,
        "alignments": False,
    }


def bench_espeak(voice: str, text: str, out_wav: Path) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    subprocess.run(
        ["espeak-ng", "-v", voice, "-w", str(out_wav), text],
        check=True,
        capture_output=True,
    )
    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audio, rate = soundfile.read(out_wav)
    return {
        "audio": audio,
        "rate": rate,
        "first": None,
        "total": total,
        "peak_mb": peak / 1e6,
        "alignments": False,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)

    engines = [
        (
            "piper",
            "lessac",
            MODELS / "en_US-lessac-medium-aligned.onnx",
            MODELS / "en_US-lessac-medium.onnx.json",
        ),
        ("piper", "alan", MODELS / "en_GB-alan-low.onnx", MODELS / "en_GB-alan-low.onnx.json"),
        (
            "kokoro",
            "af_sarah",
            MODELS / "kokoro" / "kokoro-v1.0.int8.onnx",
            MODELS / "kokoro" / "voices-v1.0.bin",
        ),
        (
            "kokoro",
            "am_michael",
            MODELS / "kokoro" / "kokoro-v1.0.int8.onnx",
            MODELS / "kokoro" / "voices-v1.0.bin",
        ),
        ("espeak-ng", "en", Path("/usr/bin/espeak-ng"), None),
    ]

    rows: list[dict] = []
    sample_files: list[str] = []
    for engine, voice, model, config in engines:
        for n, text in enumerate(TEXTS, start=1):
            name = f"{engine}-{voice}-{n}"
            wav = SAMPLES / f"{name}.wav"
            if engine == "piper":
                r = bench_piper(model, config, voice, text)
                wav_write(wav, r["audio"], r["rate"])
            elif engine == "kokoro":
                r = bench_kokoro(model, config, voice, text)
                wav_write(wav, r["audio"], r["rate"])
            else:
                r = bench_espeak(voice, text, wav)
            duration = len(r["audio"]) / r["rate"]
            rows.append(
                {
                    "name": name,
                    "engine": engine,
                    "voice": voice,
                    "text_n": n,
                    "first": r["first"],
                    "total": r["total"],
                    "rtf": r["total"] / duration if duration else float("nan"),
                    "duration": duration,
                    "peak_mb": r["peak_mb"],
                    "alignments": r["alignments"],
                }
            )
            sample_files.append(f"samples/{name}.wav")
            print(f"{name}: {r['total']:.3f}s total, {duration:.2f}s audio")

    def size_mb(path: Path) -> float:
        return path.stat().st_size / 1e6

    model_sizes = {
        "piper lessac": size_mb(MODELS / "en_US-lessac-medium-aligned.onnx")
        + size_mb(MODELS / "en_US-lessac-medium.onnx.json"),
        "piper alan": size_mb(MODELS / "en_GB-alan-low.onnx")
        + size_mb(MODELS / "en_GB-alan-low.onnx.json"),
        "kokoro": size_mb(MODELS / "kokoro" / "kokoro-v1.0.int8.onnx")
        + size_mb(MODELS / "kokoro" / "voices-v1.0.bin"),
        "espeak-ng": size_mb(Path("/usr/bin/espeak-ng")) * 1000,
    }

    lines = ["# Voice engine benchmark (N10)\n"]
    lines.append("## Results\n")
    lines.append(
        "| sample | engine | voice | text | first audio (s) | total (s) | "
        "audio (s) | RTF | peak python mem (MB) | phoneme timings |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        first = f"{r['first']:.3f}" if r["first"] is not None else "n/a (no streaming)"
        lines.append(
            f"| {r['name']} | {r['engine']} | {r['voice']} | {r['text_n']} | "
            f"{first} | {r['total']:.3f} | {r['duration']:.2f} | {r['rtf']:.2f} | "
            f"{r['peak_mb']:.1f} | {'yes' if r['alignments'] else 'no'} |"
        )
    lines.append("\n## Model sizes on disk (MB)\n")
    for k, v in model_sizes.items():
        unit = "KB (system binary)" if k == "espeak-ng" else "MB"
        lines.append(f"- {k}: {v:.1f} {unit}")
    lines.append("\n## Licenses\n")
    lines.append("- piper (rhasspy/piper) code: MIT")
    lines.append(
        "- piper voice lessac: dataset Blizzard 2013 (lessac), license at "
        "https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html"
    )
    lines.append(
        "- piper voice alan: dataset via MycroftAI/mimic3-voices (apope_low), "
        "license 'see URL' at https://github.com/MycroftAI/mimic3-voices/blob/master/voices/en_UK/apope_low"
    )
    lines.append("- kokoro-onnx (thewh1teagle/kokoro-onnx) code: MIT")
    lines.append("- Kokoro-82M model (hexgrad/Kokoro-82M): Apache-2.0")
    lines.append("- espeak-ng: system binary, Open-Source (GPL-3.0)")
    lines.append("\n## Flatpak notes\n")
    lines.append(
        "- Voxa ships as a Flatpak: engines must be bundled in the sandbox or "
        "declared as runtime dependencies. espeak-ng is a system dependency "
        "already present on this machine; piper and kokoro need their python "
        "packages plus the model files above (piper voice ~63 MB each, "
        "kokoro model ~92 MB + voices ~10 MB)."
    )
    lines.append("\n## Timing availability\n")
    lines.append(
        "- piper: phoneme-level alignments only when the voice model is patched "
        "with an alignment output (piper.patch_voice_with_alignment.add_alignment_output); "
        "stock rhasspy voices expose no alignments. No word-level timings."
    )
    lines.append("- kokoro: no word or phoneme timings.")
    lines.append("- espeak-ng: no timings.")
    lines.append("\n## Samples\n")
    for f in sample_files:
        lines.append(f"- {f}")
    (OUT / "VOICES.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'VOICES.md'}")


if __name__ == "__main__":
    main()
