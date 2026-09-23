from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voxa.dictation import segment_stream

from .cases import TestCase
from .config import HarnessSettings
from .grader import GradeResult, grade_reply


@dataclass(slots=True)
class RunResult:
    case_id: str
    category: str
    prompt: str
    transcript: str
    grade: GradeResult
    error: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "prompt": self.prompt,
            "transcript": self.transcript,
            "score": self.grade.score,
            "passed": self.grade.passed,
            "notes": self.grade.notes,
            "method": self.grade.method,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def run_case(
    case: TestCase,
    settings: HarnessSettings,
    client: Any,
    whisper: Any,
    capture: Any,
    speech: Any,
    *,
    log: Callable[[str], None] = print,
) -> RunResult:
    started = time.monotonic()
    prompt = _with_wake_word(case.prompt, settings.wake_word)
    log(f"Speaking: {prompt}")

    if settings.settle_seconds > 0:
        time.sleep(settings.settle_seconds)

    speak_done = threading.Event()
    speak_errors: list[str] = []

    def speech_error(message: str) -> None:
        speak_errors.append(message)
        speak_done.set()

    speech.speak(
        prompt,
        settings.tts_rate,
        settings.tts_voice,
        on_started=lambda: None,
        on_done=speak_done.set,
        on_error=speech_error,
    )

    if not speak_done.wait(settings.speak_timeout_seconds):
        return RunResult(case.id, case.category, case.prompt, "", GradeResult(0, False, "TTS timed out.", "error"), "TTS timed out.", time.monotonic() - started)
    if speak_errors:
        return RunResult(case.id, case.category, case.prompt, "", GradeResult(0, False, "TTS failed.", "error"), speak_errors[-1], time.monotonic() - started)

    audio_queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
    stop_event = threading.Event()
    capture_errors: list[str] = []

    def feed_audio(pcm: bytes, level: float) -> None:
        try:
            audio_queue.put_nowait((pcm, level))
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_queue.put_nowait((pcm, level))
            except queue.Full:
                pass

    def level_callback(_level: float) -> None:
        return None

    def capture_error(message: str) -> None:
        capture_errors.append(message)
        stop_event.set()

    try:
        capture.start(settings.input_device, feed_audio, level_callback, capture_error)
    except Exception as exc:  # noqa: BLE001 - hardware boundary
        return RunResult(case.id, case.category, case.prompt, "", GradeResult(0, False, "Microphone capture failed.", "error"), str(exc), time.monotonic() - started)

    segments: list[bytes] = []
    watchdog = threading.Timer(settings.max_reply_seconds, stop_event.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        for segment in segment_stream(
            audio_queue,
            stop_event,
            threshold=settings.voice_threshold,
            silence_seconds=settings.silence_ms / 1000.0,
            max_segment_seconds=settings.max_segment_seconds,
            idle_timeout_seconds=settings.reply_wait_seconds,
            on_idle_timeout=stop_event.set,
            trailing_silence_seconds=settings.end_silence_seconds,
        ):
            segments.append(segment)
    except Exception as exc:  # noqa: BLE001 - listening boundary
        capture_errors.append(str(exc))
    finally:
        watchdog.cancel()
        stop_event.set()
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        capture.stop()

    transcript_parts: list[str] = []
    for segment in segments:
        try:
            text = whisper.transcribe(segment, settings.language)
        except Exception as exc:  # noqa: BLE001 - transcription boundary
            capture_errors.append(str(exc))
            continue
        if text:
            transcript_parts.append(text)

    transcript = " ".join(transcript_parts).strip()
    grade = grade_reply(case, transcript, client, settings.model)
    error = capture_errors[-1] if capture_errors else ""
    return RunResult(case.id, case.category, case.prompt, transcript, grade, error, time.monotonic() - started)


def _with_wake_word(prompt: str, wake_word: str) -> str:
    wake = wake_word.strip()
    if not wake or prompt.casefold().startswith(wake.casefold()):
        return prompt
    return f"{wake}, {prompt}"
