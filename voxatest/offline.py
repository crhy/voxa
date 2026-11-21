"""V2: voxatest --offline-audio - comprehension measured through Voxa's real hearing pipeline.

The live harness speaks every prompt through the speakers and hears it back through the microphone,
so its comprehension numbers need a room, a working speaker and a microphone (only rhY runs those).
This module measures the same two numbers - ``hearing_word_accuracy`` and ``routed_tool_accuracy`` -
with no sound and no microphone at all:

1. render each prompt with voxatest's own test voice to 16 kHz audio, through Voxa's real TTS path
   (:meth:`voxa.speech.SpeechService._fetch_to_file` with Edge TTS word boundaries, eSpeak NG when
   the network is unavailable), cached under ``/home/rhy/voxa-bench-data/voxatest/``;
2. optionally degrade it the way the handoff's test matrix asks: level, babble/music at a stated
   SNR, speaking rate, a second voice - all seeded, so ``--seed 7`` twice gives the same bytes;
3. push that audio through the SAME pipeline the hearing bench runs
   (:class:`voxa.bench.runner.ClipRunner`: reference ring + canceller + mask, the conversation
   controller's Silero detector, ``ConversationController.feed``, ``BargeInController``,
   ``OwnSpeech``, ``SpeechSegmenter`` + endpoint, faster-whisper pinned to temperature 0 / one beam
   / ``condition_on_previous_text=False`` / word timestamps / Voxa's default model);
4. hand what the pipeline delivers to the assistant to Voxa's deterministic command router
   (``voxa.agent.intents.route`` through ``voxatest/commands.py``, the same check the
   ``voxatest commands`` bench uses) - no LLM, no speakers, no microphone;
5. fill ``heard`` and the routed tool into the run result and grade it with V1
   (:mod:`voxatest.comprehension`). Reply grading is skipped and marked n/a: nothing is spoken,
   so there is no reply to grade, and the ``misanswered`` layer stays 0 by construction.

What this measures is the hearing pipeline and the router. What it cannot measure is the room:
the microphone, the speakers, the echo of Voxa's own voice and the real desktop side effects stay
in the live suite. The voice is whatever Voxa's TTS path produces here: Edge TTS when the network
is up, eSpeak NG when it is not - the engine used is recorded per case, so a table built on the
robotic fallback never pretends to be a natural-voice measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxa.bench.render import SR, decode_audio, edge_voice, safe_key, write_wav
from voxa.bench.runner import ClipRunner, PinnedASR
from voxa.config import Settings
from voxa.speech import _download_google_voice, espeak_voice

from .cases import TestCase
from .commands import CommandCase
from .commands import check as route_check
from .comprehension import hearing_accuracy, score_run
from .grader import GradeResult
from .runner import RunResult

DEFAULT_DATA_ROOT = Path("/home/rhy/voxa-bench-data")
DEFAULT_VARIANTS = "clean"
# Voxa is speaking in every hearing case, so the hearing suite runs the product's own echo stages
# by default; ``--echo-mode`` names another one (``off`` = the speaker path uncancelled).
DEFAULT_HEARING_ECHO_MODE = "voxa"
# The second voice of the ``voice2`` variant: a different Edge voice that is also a different
# eSpeak voice offline (``voxa.speech.espeak_voice`` maps it to a female ``en-us+f3``).
SECOND_VOICE = "en-US-JennyNeural"
# Silence around every stimulus: the pre-roll the capture ring keeps, and enough quiet for the
# speech-pause rules to end the utterance (Voxa's own ``silence_ms`` is 700 ms).
LEAD_SILENCE_SECONDS = 0.30
TAIL_SILENCE_SECONDS = 1.20
# The gap between "Voxa." and the question when the wake word is spoken in its own breath
# (``--wake-mode separate``, the default: see ``spoken_prompt``).
WAKE_GAP_SECONDS = 1.20
DEFAULT_WAKE_MODE = "separate"
WAKE_MODES = ("separate", "single")
PEAK_LIMIT = 0.98  # voxa/bench/render.py MIX_PEAK: the same uniform-gain guard
GRADING_NOTE = "Reply grading skipped (n/a): nothing is spoken offline."


def log_line(message: str) -> None:
    """The default progress log: a line a caller watching a log file sees as it happens."""
    print(message, flush=True)


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True, slots=True)
class Variant:
    """One way of degrading the rendered prompt, named on the command line."""

    name: str
    level_db: float = 0.0        # loudness at the microphone, relative to the clean render
    noise: str = ""              # "", "babble", "music", "pink"
    snr_db: float = 10.0         # signal over noise, when noise is used
    rate_scale: float = 1.0      # speaking rate multiplier (0.85 slow .. 1.2 fast)
    voice: str = ""              # a TTS voice other than the harness test voice; "" = the test voice


VARIANTS: dict[str, Variant] = {
    "clean": Variant("clean"),
    "quiet": Variant("quiet", level_db=-15.0),
    "noisy": Variant("noisy", noise="babble", snr_db=10.0),
    "music": Variant("music", noise="music", snr_db=10.0),
    "fast": Variant("fast", rate_scale=1.2),
    "slow": Variant("slow", rate_scale=0.85),
    "voice2": Variant("voice2", voice=SECOND_VOICE),
}


def parse_variants(spec: str) -> list[Variant]:
    """``"clean,quiet,noisy"`` -> the variants in that order; unknown names are an error."""
    names = [item.strip() for item in str(spec or "").split(",") if item.strip()]
    if not names:
        names = ["clean"]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(
            f"unknown variant(s) {', '.join(unknown)}; choose from: {', '.join(VARIANTS)}"
        )
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return [VARIANTS[name] for name in seen]


# --------------------------------------------------------------------------- rendering


def cache_key(text: str, voice: str, rate: int) -> str:
    """The cache name of one rendered prompt: text, voice and speaking rate."""
    return f"{safe_key(text)}-{safe_key(voice)}-{int(rate)}"


def espeak_utterance(text: str, voice: str, rate: int) -> tuple[np.ndarray, list[dict]]:
    """Voxa's own offline speech fallback: eSpeak NG speaks the WHOLE utterance in one call.

    That is the shape the product uses (``voxa/speech.py:648``: one ``espeak-ng`` call for the whole
    reply). The hearing bench's lab renderer (``voxa/bench/render.py:espeak_words``) speaks one word
    at a time, because it needs word timings; the gaps it puts between words are not what Voxa
    sounds like, and this measurement needs no word timings. So the stimulus is rendered the way the
    product speaks.
    """
    proc = subprocess.run(
        ["espeak-ng", "--stdout", "-v", espeak_voice(voice), "-s", str(int(rate)), text],
        capture_output=True, check=True, stdin=subprocess.DEVNULL,
    )
    return decode_audio(proc.stdout, "wav"), []


def google_voice(text: str, voice: str, rate: int) -> tuple[np.ndarray, list[dict]]:
    """Voxa's own online fallback voice (``voxa/speech.py:_download_google_voice``), file only.

    Voxa's natural voice chain is Edge TTS first and this endpoint second; the hearing bench uses
    the same chain, so a suite measured here is measured on the voice the product would speak when
    Edge is unavailable, not on a robotic substitute. No word timings come back from it and none are
    needed: comprehension scores words, not their onsets.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="voxa-voxatest-", suffix=".mp3", delete=False) as handle:
        path = handle.name
    try:
        _download_google_voice(text, path)
        raw = Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)
    return decode_audio(raw, "mp3"), []


def natural_voice(text: str, voice: str, rate: int, tmp_dir: Path) -> tuple[np.ndarray, list[dict], str]:
    """Voxa's natural voice chain, in Voxa's own order: Edge TTS, then the Google endpoint."""
    try:
        samples, words = edge_voice(text, voice, rate, tmp_dir)
        return samples, words, "edge_tts"
    except Exception as edge_error:  # noqa: BLE001 - the product's own first fallback
        print(f"Edge TTS unavailable ({str(edge_error)[:110]}); trying Voxa's Google voice fallback",
              file=sys.stderr, flush=True)
    samples, words = google_voice(text, voice, rate)
    return samples, words, "google-tts"


def render_prompt(text: str, voice: str, rate: int, data_root: Path) -> dict[str, Any]:
    """One prompt as 16 kHz float samples through Voxa's own TTS path, cached under ``voxatest/``.

    The cache is the point: a 20-case suite run over six variants speaks the same prompt text many
    times, and Edge TTS/eSpeak must not be asked twice for the same (text, voice, rate).
    """
    cache = Path(data_root) / "voxatest"
    key = cache_key(text, voice, rate)
    wav, meta = cache / f"{key}.wav", cache / f"{key}.json"
    if wav.exists() and meta.exists():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        return {
            "key": key, "text": payload["text"], "voice": payload["voice"], "rate": payload["rate"],
            "engine": payload["engine"], "samples": _read_wav_float(wav), "words": payload["words"],
            "cached": True,
        }
    cache.mkdir(parents=True, exist_ok=True)
    engine = "edge_tts"
    try:
        samples, words, engine = natural_voice(text, voice, rate, cache / "_tmp")
    except Exception as exc:  # noqa: BLE001 - the product's last resort: eSpeak NG, and say so
        print(f"natural voices unavailable ({str(exc)[:120]}); using eSpeak NG for {text!r}",
              file=sys.stderr, flush=True)
        samples, words = espeak_utterance(text, voice, rate)
        engine = "espeak-ng"
    samples = np.asarray(samples, dtype=np.float32)
    write_wav(wav, samples)
    meta.write_text(json.dumps(
        {"text": text, "voice": voice, "rate": int(rate), "engine": engine, "sr": SR, "words": words},
        indent=2) + "\n")
    # Re-read the 16-bit cache so a prompt built in this run is bit-for-bit what a later run reads.
    return {"key": key, "text": text, "voice": voice, "rate": int(rate), "engine": engine,
            "samples": _read_wav_float(wav), "words": words, "cached": False}


def _read_wav_float(path: Path) -> np.ndarray:
    """A cached 16-bit mono 16 kHz WAV as float samples in -1..1."""
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("only 16-bit mono WAVs are supported")
        if handle.getframerate() != SR:
            raise ValueError(f"expected {SR} Hz, found {handle.getframerate()} Hz")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- noise


def babble_source(data_root: Path, rng: np.random.Generator, seconds: float) -> tuple[np.ndarray, str]:
    """Other people talking: real speech clips from the bench's user-voice cache, tiled to length.

    Returns (samples, kind). When the bench sources are not on disk the mix says so and uses pink
    noise, so a report never claims babble it did not play.
    """
    clips = sorted((Path(data_root) / "sources" / "user").glob("*.wav"))
    if not clips:
        return pink_noise(seconds, rng), "pink"
    picks: list[np.ndarray] = []
    order = rng.permutation(len(clips))
    for index in order[:3]:
        samples = _read_wav_float(clips[int(index)])
        if samples.size:
            picks.append(samples)
    if not picks:
        return pink_noise(seconds, rng), "pink"
    stream = np.concatenate(picks)
    start = int(rng.integers(0, max(1, stream.size)))
    while stream[start:].size < seconds * SR:
        stream = np.concatenate([stream, stream])
    return stream[start : start + int(seconds * SR)], "babble"


def pink_noise(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Deterministic 1/f-ish noise from the run's seed (the fallback when there is no babble)."""
    samples = int(max(1, seconds * SR))
    white = rng.standard_normal(samples)
    pink = np.empty(samples, dtype=np.float64)
    state = 0.0
    for index, value in enumerate(white):
        state = 0.97 * state + 0.03 * value
        pink[index] = state
    return pink.astype(np.float32)


def music_source(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Deterministic synthesised music: a seeded chord progression with a beat envelope.

    Synthesised on purpose: there is no licensed music corpus in the bench data, and a report that
    says "music" must say what kind. It is broadband, modulated, non-speech interference - which is
    what the ``noisy``/``music`` cells of the handoff matrix ask for.
    """
    samples = int(max(1, seconds * SR))
    time_axis = np.arange(samples, dtype=np.float64) / SR
    root = float(rng.uniform(90.0, 200.0))
    ratios = (1.0, 1.25, 1.5, 2.0)
    beat = float(rng.uniform(0.35, 0.65))
    notes = int(np.ceil(seconds / beat))
    out = np.zeros(samples, dtype=np.float64)
    for note in range(notes):
        ratio = ratios[int(rng.integers(0, len(ratios)))]
        lo = int(note * beat * SR)
        hi = min(samples, int((note + 1) * beat * SR))
        if hi <= lo:
            continue
        span = time_axis[lo:hi] - time_axis[lo]
        freq = root * ratio
        tone = (np.sin(2 * np.pi * freq * span)
                + 0.45 * np.sin(2 * np.pi * 2 * freq * span)
                + 0.22 * np.sin(2 * np.pi * 3 * freq * span))
        tone *= np.exp(-1.6 * span)  # a struck, decaying note
        out[lo:hi] += tone
    # Soften the pure tones a little so it behaves like a mix, not like a test signal.
    kernel = np.ones(9) / 9.0
    out = np.convolve(out, kernel, mode="same")
    return out.astype(np.float32)


def mix_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise at a stated SNR (power ratio at the microphone, not a volume slider)."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if noise.size < signal.size:
        noise = np.tile(noise, int(np.ceil(signal.size / max(1, noise.size))))
    noise = noise[: signal.size]
    signal_power = float(np.mean(signal**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 1e-18:
        return signal.astype(np.float32)
    wanted_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = (wanted_noise_power / noise_power) ** 0.5
    return (signal + noise * scale).astype(np.float32)


def realised_snr_db(signal: np.ndarray, mixed: np.ndarray) -> float:
    """The SNR the mix really produced, measured on the samples that went into the pipeline."""
    residual = np.asarray(mixed, dtype=np.float64) - np.asarray(signal, dtype=np.float64)
    signal_power = float(np.mean(np.asarray(signal, dtype=np.float64) ** 2))
    noise_power = float(np.mean(residual**2))
    if noise_power <= 1e-18:
        return float("inf")
    return round(10.0 * np.log10(max(signal_power, 1e-18) / noise_power), 3)


def apply_variant(samples: np.ndarray, variant: Variant, rng: np.random.Generator,
                  data_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Level, noise and the clipping guard, in the order the microphone would have heard them."""
    clean = np.asarray(samples, dtype=np.float32)
    info: dict[str, Any] = {"variant": variant.name, "level_db": variant.level_db,
                            "noise": variant.noise or "none", "snr_db": None, "noise_kind": "none"}
    mixed = clean
    if variant.noise:
        if variant.noise == "music":
            noise, kind = music_source(len(clean) / SR, rng), "music (synthesised)"
        elif variant.noise == "pink":
            noise, kind = pink_noise(len(clean) / SR, rng), "pink"
        else:
            noise, kind = babble_source(Path(data_root), rng, len(clean) / SR)
            kind = "babble" if kind == "babble" else f"{kind} (no babble sources)"
        mixed = mix_noise(clean, noise, variant.snr_db)
        info["noise_kind"] = kind
        info["snr_db"] = realised_snr_db(clean, mixed)
    if variant.level_db:
        mixed = mixed * float(10.0 ** (variant.level_db / 20.0))
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > PEAK_LIMIT:  # the same uniform gain the bench renderer uses: ratios stay exact
        mixed = mixed * (PEAK_LIMIT / peak)
        info["peak_guard"] = round(PEAK_LIMIT / peak, 4)
    info["peak"] = round(float(np.max(np.abs(mixed))) if mixed.size else 0.0, 4)
    return np.asarray(mixed, dtype=np.float32), info


def build_clip(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The microphone stream (int16) and the silent reference stream for one stimulus.

    Voxa is not speaking in this test, so the reference is silence: the echo stages have nothing to
    cancel, which is exactly the ``off`` echo mode of the bench.
    """
    lead = np.zeros(int(LEAD_SILENCE_SECONDS * SR), dtype=np.float32)
    tail = np.zeros(int(TAIL_SILENCE_SECONDS * SR), dtype=np.float32)
    clip = np.concatenate([lead, np.asarray(samples, dtype=np.float32), tail])
    mic = np.clip(np.rint(clip * 32767.0), -32768, 32767).astype("<i2")
    return mic, np.zeros(mic.size, dtype=np.float32)


# --------------------------------------------------------------------------- the pipeline


def pipeline_settings(settings: Any) -> Settings:
    """Voxa's own settings for the pipeline, taken from the harness (which reads Voxa's config)."""
    return Settings(
        whisper_model=getattr(settings, "whisper_model", "base") or "base",
        language=getattr(settings, "language", "en") or "en",
        silence_ms=int(getattr(settings, "silence_ms", 700)),
        voice_threshold=int(getattr(settings, "voice_threshold", 450)),
        max_segment_seconds=float(getattr(settings, "max_segment_seconds", 20.0)),
        wake_word=str(getattr(settings, "wake_word", "voxa") or "voxa"),
    ).normalized()


def hear(samples: np.ndarray, asr: Any, settings: Any, case_id: str,
         echo_mode: str = "off") -> dict[str, Any]:
    """One stimulus through the bench pipeline; the requests it delivered and what it cost."""
    mic, ref = build_clip(samples)
    truth = {
        "id": case_id, "scenario": "voxatest", "playback": {"text": ""}, "expect": "keep",
        "overlap_position": None, "pair": None,
    }
    runner = ClipRunner(asr=asr, echo_mode=echo_mode,
                        wake_word=str(getattr(settings, "wake_word", "voxa") or "voxa"),
                        settings=pipeline_settings(settings))
    summary = runner.run(mic, ref, truth)
    return {"summary": summary, "requests": list(summary["requests"]),
            "transcripts": list(summary["transcript_texts"]),
            "turns": list(summary["turns"]), "counts": dict(summary["counts"]),
            "events": list(runner.events), "audio_seconds": float(summary["seconds"]),
            "asr_calls": int(summary["asr_calls"]),
            "asr_processing_seconds": float(summary["asr_processing_seconds"])}


def spoken_prompt(prompt: str, wake_word: str) -> str:
    """The words of the single-breath stimulus: ``runner._with_wake_word`` speaks ``"wake, prompt"``.

    Used by ``--wake-mode single``. The default is ``separate``: the harness addresses Voxa in one
    breath ("Voxa.") and asks in the next, which is the flow the conversation controller is built
    around (the wake utterance arms the prompt, the prompt is transcribed with no wake hint).
    ``--wake-mode single`` measures the one-breath flow the live harness speaks today, wake word and
    all, and is how the wake-word loss is measured (see the f143 notes).
    """
    wake = str(wake_word or "").strip()
    if not wake or prompt.casefold().startswith(wake.casefold()):
        return prompt
    return f"{wake}, {prompt}"


def wake_clip_text(wake_word: str) -> str:
    """The address on its own: what a person says first, before the request."""
    wake = str(wake_word or "").strip()
    return f"{wake}." if wake and not wake.endswith((".", "!", "?")) else wake


def build_stimulus(case: TestCase, settings: Any, variant: Variant, data_root: Path,
                   wake_word: str, wake_mode: str = DEFAULT_WAKE_MODE) -> tuple[np.ndarray, dict[str, Any]]:
    """The rendered prompt audio (and the wake clip with it), before any degradation."""
    voice = variant.voice or settings.tts_voice
    rate = max(80, int(round(settings.tts_rate * variant.rate_scale)))
    gap = np.zeros(int(WAKE_GAP_SECONDS * SR), dtype=np.float32)
    if wake_mode == "single":
        rendered = render_prompt(spoken_prompt(case.prompt, wake_word), voice, rate, Path(data_root))
        return rendered["samples"], {"engine": rendered["engine"], "voice": rendered["voice"],
                                     "rate": rendered["rate"], "wake_mode": wake_mode,
                                     "spoken": spoken_prompt(case.prompt, wake_word)}
    wake = render_prompt(wake_clip_text(wake_word), voice, rate, Path(data_root))
    prompt = render_prompt(case.prompt, voice, rate, Path(data_root))
    return (np.concatenate([wake["samples"], gap, prompt["samples"]]),
            {"engine": prompt["engine"], "voice": prompt["voice"], "rate": prompt["rate"],
             "wake_mode": wake_mode, "spoken": f"{wake_clip_text(wake_word)} | {case.prompt}",
             "wake_engine": wake["engine"]})


# --------------------------------------------------------------------------- one case


def routing_matches(case: TestCase, got_tool: str | None,
                    got_args: dict[str, Any]) -> tuple[bool, str]:
    """Does the routed call answer this case? Voxa's own argument rule: letter case is free.

    Three rules, in this order:

    * a case that names no expected tool cannot be scored for routing (V1's rule), so it always matches;
    * the tool must be the one the case expects;
    * the arguments are compared the way the live harness compares them
      (:func:`voxatest.runner._same_argument`: letter case ignored, ``a|b`` lists accepted spellings).

    The command bench (:mod:`voxatest.commands`) compares arguments exactly, which is right for a text
    corpus; here the words arrive from an ear, so "YouTube" and "youtube" are the same answer and a
    letter case must never be reported as a hearing failure.
    """
    from .runner import _same_argument

    if not case.expected_tool:
        return True, ""
    if got_tool != case.expected_tool:
        return False, f"expected {case.expected_tool}, got {got_tool or 'nothing'}"
    for key, value in case.expected_args.items():
        actual = (got_args or {}).get(key)
        if not _same_argument(value, actual):
            return False, f"argument {key}: expected {value!r}, got {actual!r}"
    return True, ""


def run_offline_case(case: TestCase, settings: Any, asr: Any, registry: Any, variant: Variant,
                     seed: int, data_root: Path, *, index: int = 0, variant_index: int = 0,
                     wake_word: str | None = None, log: Callable[[str], None] = log_line,
                     echo_mode: str = "off", wake_mode: str = DEFAULT_WAKE_MODE) -> RunResult:
    """One case: render -> degrade -> the real hearing pipeline -> the command router -> V1 grading."""
    started = time.perf_counter()
    wake = str(wake_word if wake_word is not None else settings.wake_word)
    stimulus_raw, rendered = build_stimulus(case, settings, variant, Path(data_root), wake, wake_mode)
    rng = np.random.default_rng(np.random.SeedSequence(
        entropy=int(seed), spawn_key=(int(index), int(variant_index))))
    stimulus, mix_info = apply_variant(stimulus_raw, variant, rng, Path(data_root))
    heard_result = hear(stimulus, asr, settings, case.id, echo_mode=echo_mode)
    heard = " ".join(heard_result["requests"]).strip()

    routed = route_check(
        CommandCase(case.id, case.category, heard, case.expected_tool or None,
                    dict(case.expected_args) if case.expected_args else None),
        registry,
    )
    accuracy = hearing_accuracy(case.prompt, heard, wake)
    # V1's rule: a case that names no expected tool cannot be scored for routing, so routing is
    # n/a for it and can never be the reason a case fails. It is still reported (what the router
    # really did with those words) in the notes and the per-case block.
    routing_ok, routing_problem = routing_matches(case, routed.got_tool, routed.got_args)
    passed = bool(accuracy == 1.0 and routing_ok and heard)
    notes = (
        f"Offline audio ({rendered['engine']}, variant {variant.name}): heard \"{heard}\"; "
        f"router did {routed.got_tool or 'nothing'}{f' ({routing_problem})' if routing_problem else ''}. "
        f"{GRADING_NOTE}"
    )
    grade = GradeResult(100 if passed else int(round(100.0 * (accuracy if accuracy is not None else 0.0))),
                        passed, notes, "offline")
    wall = time.perf_counter() - started
    action_record = {
        "heard": heard, "route": "offline-audio", "tool": routed.got_tool or "",
        "args": dict(routed.got_args), "ok": routing_ok, "variant": variant.name,
    }
    offline = {
        "variant": variant.name,
        "engine": rendered["engine"],
        "voice": rendered["voice"],
        "rate": rendered["rate"],
        "wake_mode": rendered["wake_mode"],
        "spoken": rendered["spoken"],
        "rate_scale": variant.rate_scale,
        "level_db": variant.level_db,
        "noise": mix_info["noise"],
        "noise_kind": mix_info["noise_kind"],
        "snr_db": mix_info["snr_db"],
        "peak": mix_info["peak"],
        "peak_guard": mix_info.get("peak_guard"),
        "audio_seconds": round(heard_result["audio_seconds"], 3),
        "wall_seconds": round(wall, 3),
        "asr_calls": heard_result["asr_calls"],
        "asr_processing_seconds": heard_result["asr_processing_seconds"],
        "requests": heard_result["requests"],
        "turns": heard_result["turns"],
        "counts": heard_result["counts"],
        "routed_tool": routed.got_tool,
        "routed_args": dict(routed.got_args),
        "routing_problem": routing_problem,
        "hearing_word_accuracy": None if accuracy is None else round(accuracy, 4),
        "echo_mode": echo_mode,
    }
    log(f"  {variant.name}: heard \"{heard}\" -> {routed.got_tool or 'no tool'} "
        f"({wall:.2f}s wall, {heard_result['audio_seconds']:.2f}s audio)")
    return RunResult(
        case.id, case.category, case.prompt, "", grade, "", wall,
        action_record, case.expected_tool, heard, case.capability,
        [routed.got_tool] if routed.got_tool else [], heard_result["requests"], offline,
    )


# --------------------------------------------------------------------------- a whole run


def hearing_grade(case: TestCase, measures: dict[str, Any]) -> GradeResult:
    """Did this hearing case get the turn decision it asked for?

    ``yield``  : she stopped her own playback.       ``keep``   : she kept speaking (nobody took the floor).
    ``answer`` : her words reached her whole.        Reply grading is n/a: nothing is spoken here.
    """
    expect = str(measures.get("expect") or case.expect)
    if expect == "yield":
        ok = bool(measures.get("yielded"))
        detail = (f"she {'stopped' if ok else 'did not stop'} her own playback "
                  f"(stop latency {measures.get('stop_latency_ms')} ms)")
    elif expect == "keep":
        ok = bool(measures.get("kept_turn"))
        detail = (f"she {'kept' if ok else 'lost'} the turn "
                  f"(self-interruptions {measures.get('self_interrupt_count')})")
    else:
        ok = bool(measures.get("request_complete"))
        detail = (f"request {'complete' if ok else 'incomplete'}: "
                  f"retention {measures.get('word_retention')} of "
                  f"{measures.get('expected_request')!r} -> {measures.get('request_text')!r}")
    score = 100 if ok else int(round(100.0 * float(measures.get("word_retention") or 0.0)))
    notes = (f"Hearing ({expect}, cell {measures.get('cell')}, echo {measures.get('echo_mode')}, "
             f"onset {measures.get('user_onset_s')} s, {measures.get('clip', {}).get('level_db')} dB at the mic): "
             f"{detail}. {GRADING_NOTE}")
    return GradeResult(score, ok, notes, "hearing")


def run_hearing_offline_case(case: TestCase, settings: Any, asr: Any, registry: Any, variant: Variant,
                             seed: int, data_root: Path, *, index: int = 0, variant_index: int = 0,
                             wake_word: str | None = None, log: Callable[[str], None] = log_line,
                             echo_mode: str | None = None) -> RunResult:
    """One hearing case: the overlap is BUILT with the bench renderer, no speakers, no microphone.

    Voxa is speaking in every hearing case, so the echo stages that run are the product's own
    (``voxa``) unless the caller names a mode: ``off`` measures the same cells with the speaker
    path uncancelled, which is the condition the frozen bench's ``off`` mode means.
    """
    from . import hearing

    mode = str(echo_mode or DEFAULT_HEARING_ECHO_MODE)
    measures = hearing.run_hearing_case(case, settings, asr, registry, seed, Path(data_root),
                                        index=index, echo_mode=mode, wake_word=wake_word,
                                        voice=variant.voice, log=log)
    grade = hearing_grade(case, measures)
    routed_tool = str(measures.get("routed_tool") or "")
    action_record = {"heard": measures.get("request_text", ""), "route": "offline-audio",
                     "tool": routed_tool, "args": dict(measures.get("routed_args") or {}),
                     "variant": variant.name, "hearing": True}
    offline = {
        "mode": "offline-audio",
        "variant": variant.name,
        "engine": measures.get("clip", {}).get("reply_engine") or measures.get("clip", {}).get("user_engine"),
        "voice": measures.get("clip", {}).get("user_voice"),
        "rate": None,
        "wake_mode": "hearing",
        "spoken": f"{case.reply} || {case.prompt}".strip(),
        "rate_scale": variant.rate_scale,
        "level_db": measures.get("clip", {}).get("level_db"),
        "noise": "none",
        "noise_kind": "none",
        "snr_db": None,
        "peak": None,
        "peak_guard": measures.get("clip", {}).get("peak_guard"),
        "audio_seconds": measures.get("audio_seconds"),
        "wall_seconds": measures.get("wall_seconds"),
        "asr_calls": None,
        "asr_processing_seconds": None,
        "requests": [measures.get("request_text")] if measures.get("request_text") else [],
        "turns": measures.get("turns") or [],
        "counts": measures.get("counts") or {},
        "routed_tool": measures.get("routed_tool"),
        "routed_args": dict(measures.get("routed_args") or {}),
        "routing_problem": "",
        "hearing_word_accuracy": measures.get("hearing_word_accuracy"),
        "echo_mode": measures.get("echo_mode"),
        "hearing": measures,
    }
    return RunResult(case.id, case.category, case.prompt, "", grade, "",
                     float(measures.get("wall_seconds") or 0.0), action_record,
                     case.expected_tool, measures.get("request_text", ""), case.capability,
                     [routed_tool] if routed_tool else [],
                     [measures.get("request_text")] if measures.get("request_text") else [],
                     offline, measures)



def load_offline_cases(path: Path) -> list[TestCase]:
    """Cases for the offline run, from either case-file shape voxatest writes.

    ``voxatest generate``/``run`` files carry ``prompt`` (+ optional ``expected_tool``/``expected_args``);
    the command bench files (:mod:`voxatest.commands`) carry ``say``/``tool``/``args``. Both are read here,
    mixed files included; the command bench's ``setup``/``verify``/``after`` host steps are carried along
    for the report even though an offline run performs none of them: nothing is typed, clicked, spoken or
    replied here.
    """
    if not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("cases", payload.get("test_cases", [])) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ValueError("Test case file must contain a JSON array or an object with a cases array.")
    cases: list[TestCase] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        data = dict(item)
        if not str(data.get("prompt", "")).strip() and str(data.get("say", "") or "").strip():
            data["prompt"] = str(data["say"]).strip()
        # A JSON null tool/args means "no command expected" (a question, a dictation cue, a
        # compound), NOT a tool literally named "None": ``str(None)`` would score such a case as a
        # misroute and count it in the routing denominator (V1: no expected tool cannot be scored).
        if not str(data.get("expected_tool", "") or "").strip() and str(data.get("tool", "") or "").strip():
            data["expected_tool"] = str(data["tool"]).strip()
        if not data.get("expected_args") and isinstance(data.get("args"), dict):
            data["expected_args"] = dict(data["args"])
        cases.append(TestCase.from_dict(data))
    return cases


def _offline_block(results: list[RunResult], variants: list[Variant], seed: int,
                   data_root: Path, model: str, wall_total: float,
                   wake_mode: str = DEFAULT_WAKE_MODE) -> dict[str, Any]:
    """The ``offline`` block of the report: how the audio was made and what it cost in wall time."""
    from . import hearing

    walls = [float(result.duration_seconds) for result in results]
    audio = [float((result.offline or {}).get("audio_seconds") or 0.0) for result in results]
    engines = sorted({str((result.offline or {}).get("engine", "")) for result in results})
    hearing_cases = [row for row in (result.hearing for result in results) if isinstance(row, dict)]
    block = {
        "mode": "offline-audio",
        "seed": int(seed),
        "variants": [variant.name for variant in variants],
        "wake_mode": wake_mode,
        "data_root": str(data_root),
        "whisper_model": model,
        "engines": engines,
        "cases": len(results),
        "wall_seconds": round(wall_total, 3),
        "wall_seconds_per_case": {
            "median": _median(walls), "p95": _percentile(walls, 95), "max": _maximum(walls),
        },
        "audio_seconds": round(sum(audio), 3),
        "realtime_factor": round(sum(audio) / wall_total, 3) if wall_total > 0 else None,
        "reply_grading": "n/a",
        "not_measured": [
            "microphone and speakers (no audio device was touched)",
            "the reply she would have spoken (no model was called)",
        ],
    }
    if hearing_cases:
        block["hearing_cases"] = len(hearing_cases)
        block["echo_modes"] = sorted({str(row.get("echo_mode")) for row in hearing_cases})
        block["hearing"] = hearing.score_hearing(hearing_cases)
    else:
        block["not_measured"].append("Voxa's own voice as echo (she never speaks in this mode)")
    return block


def _median(values: list[float]) -> float | None:
    return round(float(np.median(np.asarray(values, dtype=np.float64))), 3) if values else None


def _percentile(values: list[float], percent: float) -> float | None:
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), percent)), 3) if values else None


def _maximum(values: list[float]) -> float | None:
    return round(max(values), 3) if values else None


def run_offline(cases: list[TestCase], settings: Any, variants: list[Variant], seed: int,
                data_root: Path, *, registry: Any = None, asr: Any = None, model: str = "",
                echo_mode: str | None = None, wake_mode: str = DEFAULT_WAKE_MODE,
                log: Callable[[str], None] = log_line) -> tuple[list[RunResult], dict[str, Any]]:
    """Every case through every variant, on one Whisper model loaded once.

    Hearing cases (the ones that name a turn decision) go through :func:`run_hearing_offline_case`,
    which builds the overlap with the bench renderer; every other case goes through V2's path.
    """
    from . import hearing

    if registry is None:
        from voxa.agent.tools import default_registry

        registry = default_registry()
    if asr is None:
        started = time.perf_counter()
        asr = PinnedASR(model_name=model or settings.whisper_model)
        log(f"Whisper model {asr.model_name} loaded in {time.perf_counter() - started:.1f} s "
            f"(one model for the whole run)")
    started = time.perf_counter()
    results: list[RunResult] = []
    total = len(cases) * len(variants)
    number = 0
    for variant_index, variant in enumerate(variants):
        for index, case in enumerate(cases):
            number += 1
            if case.vm_only:
                log(f"[{number}/{total}] SKIP {case.id} (vm_only: needs a real desktop)")
                continue
            problem = hearing.validate_hearing(case)
            if problem:
                log(f"[{number}/{total}] SKIP {case.id}: {problem}")
                continue
            log(f"[{number}/{total}] {case.id} ({case.category}) variant {variant.name}: "
                f"\"{case.prompt}\"")
            if hearing.is_hearing_case(case):
                results.append(run_hearing_offline_case(case, settings, asr, registry, variant, seed,
                                                        data_root, index=index,
                                                        variant_index=variant_index,
                                                        wake_word=settings.wake_word, log=log,
                                                        echo_mode=echo_mode))
                continue
            results.append(run_offline_case(case, settings, asr, registry, variant, seed, data_root,
                                            index=index, variant_index=variant_index,
                                            wake_word=settings.wake_word, log=log,
                                            echo_mode=echo_mode or "off",
                                            wake_mode=wake_mode))
    return results, _offline_block(results, variants, seed, data_root, model,
                                   time.perf_counter() - started)


def format_offline(offline: dict[str, Any], comprehension: dict[str, Any]) -> list[str]:
    """The Offline-audio section of the text report: how it was made, the wall times, the numbers."""
    lines = [
        "Offline audio (no speakers, no microphone)",
        f"  variants {', '.join(offline.get('variants') or [])}  seed {offline.get('seed')}"
        f"  TTS engine(s) {', '.join(offline.get('engines') or [])}",
        f"  cases {offline.get('cases')}  audio {offline.get('audio_seconds')} s"
        f"  wall {offline.get('wall_seconds')} s"
        f"  ({offline.get('realtime_factor')}x audio time)",
        "  wall time per case: median {median} s  p95 {p95} s  max {max} s".format(
            **offline.get("wall_seconds_per_case") or {}),
        f"  hearing_word_accuracy {_fmt(comprehension.get('hearing_word_accuracy'))}"
        f"  routed_tool_accuracy {_fmt(comprehension.get('routed_tool_accuracy'))}",
        "  reply grading: n/a (nothing is spoken offline)",
    ]
    return lines


def _fmt(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def offline_main(argv: list[str] | None = None) -> int:
    """``python -m voxatest offline-audio --cases FILE [--variants …] [--seed N]`` (the same run)."""
    parser = argparse.ArgumentParser(
        prog="python -m voxatest offline-audio",
        description="measure comprehension through Voxa's real hearing pipeline without speakers or a mic",
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model", default="", help="faster-whisper model (default: Voxa's)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--echo-mode", choices=("voxa", "webrtc", "off"), default=None,
                        help="echo stages: voxa for the hearing cases (she is speaking), off for the rest")
    parser.add_argument("--wake-mode", choices=WAKE_MODES, default=DEFAULT_WAKE_MODE,
                        help="separate: \"Voxa.\" then the question; single: the one-breath prompt the live harness speaks")
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--text-report", default=None)
    parser.add_argument("--reports", default=None)
    parser.add_argument("--failure-report", default=None)
    args = parser.parse_args(argv)
    from .config import load_settings
    from .report import report_paths, write_reports

    settings = load_settings()
    cases = load_offline_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No test cases found in {args.cases}.", file=sys.stderr)
        return 1
    results, offline = run_offline(cases, settings, parse_variants(args.variants), args.seed,
                                   Path(args.data_root), model=args.model, echo_mode=args.echo_mode,
                                   wake_mode=args.wake_mode)
    reports = Path(args.reports or settings.reports_dir)
    json_path, text_path = report_paths(
        reports,
        json_path=Path(args.json_report) if args.json_report else None,
        text_path=Path(args.text_report) if args.text_report else None,
    )
    rows = [result.to_dict() for result in results]
    write_reports(results, reports, json_path=json_path, text_path=text_path,
                  failure_path=Path(args.failure_report) if args.failure_report else None,
                  complete=True, wake_word=settings.wake_word, extra={"offline": offline})
    print("\n".join(format_offline(offline, score_run(rows, settings.wake_word))))
    print(f"JSON report: {json_path}", flush=True)
    print(f"Text report: {text_path}", flush=True)
    return 0


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_VARIANTS",
    "GRADING_NOTE",
    "SECOND_VOICE",
    "VARIANTS",
    "Variant",
    "apply_variant",
    "babble_source",
    "build_clip",
    "build_stimulus",
    "cache_key",
    "decode_audio",
    "espeak_utterance",
    "format_offline",
    "hear",
    "load_offline_cases",
    "log_line",
    "mix_noise",
    "music_source",
    "offline_main",
    "parse_variants",
    "pipeline_settings",
    "pink_noise",
    "realised_snr_db",
    "render_prompt",
    "routing_matches",
    "run_offline",
    "run_offline_case",
    "spoken_prompt",
    "wake_clip_text",
]
