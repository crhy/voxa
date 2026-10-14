from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxa.agent.actionlog import ActionLog
from voxa.dictation import segment_stream

from .cases import TestCase
from .config import HarnessSettings
from .grader import REPLY_MATCH, GradeResult, grade_reply, keyword_grade, reply_match, spoken_words

PLAYBACK_LISTEN_SECONDS = 10.0
STOP_SETTLE_SECONDS = 5.0
PLAYBACK_REPLY_MATCH = 0.5
MEDIA_START_TOOLS = frozenset({"play_radio", "play_tv", "play_music", "play_video", "play_youtube"})


def stop_media(log: Callable[[str], None] = print) -> int:
    """Stop every media player on the session bus (MPRIS); returns how many were told to stop."""
    try:
        from gi.repository import Gio, GLib

        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        names = bus.call_sync(
            "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "ListNames",
            None, GLib.VariantType("(as)"), Gio.DBusCallFlags.NONE, 2000, None,
        ).unpack()[0]
        players = [name for name in names if name.startswith("org.mpris.MediaPlayer2.")]
        for name in players:
            bus.call_sync(
                name, "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player", "Stop",
                None, None, Gio.DBusCallFlags.NONE, 2000, None,
            )
        log(f"Stopped playback ({len(players)} player(s)).")
        # The stream's last buffered seconds still come out of the speakers after "Stop".
        time.sleep(STOP_SETTLE_SECONDS)
        return len(players)
    except Exception as exc:  # noqa: BLE001 - clean-up must never fail a case
        log(f"Could not stop playback: {exc}")
        return 0


@dataclass(slots=True)
class RunResult:
    case_id: str
    category: str
    prompt: str
    transcript: str
    grade: GradeResult
    error: str = ""
    duration_seconds: float = 0.0
    action_record: dict[str, Any] | None = None
    expected_tool: str = ""
    heard: str = ""
    capability: int = 0

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
            "expected_tool": self.expected_tool,
            "heard": self.heard,
            "capability": self.capability,
            "action_record": self.action_record,
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
    action_log = ActionLog(settings.action_log_path)
    action_offset = len(action_log.read())
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
    # Music never falls silent, so a case that starts playback listens for a fixed short time.
    listen_seconds = min(settings.max_reply_seconds, PLAYBACK_LISTEN_SECONDS) if case.after == "stop_media" else settings.max_reply_seconds
    watchdog = threading.Timer(listen_seconds, stop_event.set)
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
        media_started = {str(record.get("tool", "")) for record in action_log.read()[action_offset:]} & MEDIA_START_TOOLS
        if case.after == "stop_media" or media_started:
            # Also when playback was not expected (a misheard prompt started a video): left on,
            # it would talk over every later case.
            stop_media(log)

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
    if case.expected_tool and transcript and not (case.expected_contains or case.expected_keywords):
        # Nothing to look for in the words themselves: the action log decides below.
        grade = GradeResult(100, True, "Checked against Voxa's action log and spoken reply.", "action_log")
    else:
        grade = grade_reply(case, transcript, client, settings.model)
    action_record = None
    new_records = action_log.read()[action_offset:]
    heard = " / ".join(str(record.get("heard", "")) for record in new_records if record.get("heard"))
    if case.expected_tool:
        grade, action_record = _require_action(case, grade, new_records, transcript)
    if grade.passed and case.verify:
        grade = verify_outcome(case, grade, settings, whisper, capture, log)
    error = capture_errors[-1] if capture_errors else ""
    return RunResult(
        case.id, case.category, case.prompt, transcript, grade, error,
        time.monotonic() - started, action_record, case.expected_tool, heard, case.capability,
    )


def check_file(verify: dict[str, Any]) -> str:
    """"" when the file holds what the case expects, else what is wrong with it."""
    path = Path(str(verify.get("file", ""))).expanduser()
    try:
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
    except OSError as exc:
        return f"{path} could not be read: {exc}."
    wanted = str(verify.get("file_contains", "")).casefold()
    unwanted = str(verify.get("file_lacks", "")).casefold()
    if wanted and wanted not in text:
        return f"{path.name} does not contain \"{verify['file_contains']}\"."
    if unwanted and unwanted in text:
        return f"{path.name} still contains \"{verify['file_lacks']}\"."
    return ""


def hear_until(
    phrase: str, seconds: float, settings: HarnessSettings, whisper: Any, capture: Any,
) -> tuple[bool, str]:
    """Listen for up to ``seconds`` for Voxa to say ``phrase`` unprompted; (heard it, everything heard)."""
    audio_queue: queue.Queue[tuple[bytes, float] | None] = queue.Queue(maxsize=80)
    stop_event = threading.Event()

    def feed(pcm: bytes, level: float) -> None:
        try:
            audio_queue.put_nowait((pcm, level))
        except queue.Full:
            pass

    capture.start(settings.input_device, feed, lambda _level: None, lambda _message: stop_event.set())
    watchdog = threading.Timer(seconds, stop_event.set)
    watchdog.daemon = True
    watchdog.start()
    heard: list[str] = []
    try:
        for segment in segment_stream(
            audio_queue,
            stop_event,
            threshold=settings.voice_threshold,
            silence_seconds=settings.silence_ms / 1000.0,
            max_segment_seconds=settings.max_segment_seconds,
        ):
            text = whisper.transcribe(segment, settings.language)
            if text:
                heard.append(text)
                # Exact words only: a loose match once took "see you next time" for "timer".
                if any(alternative.strip().casefold() in " ".join(spoken_words(text)) for alternative in phrase.split("|") if alternative.strip()):
                    return True, " ".join(heard)
    finally:
        watchdog.cancel()
        stop_event.set()
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        capture.stop()
    return False, " ".join(heard)


def verify_outcome(
    case: TestCase, grade: GradeResult, settings: HarnessSettings, whisper: Any, capture: Any,
    log: Callable[[str], None] = print,
) -> GradeResult:
    """Check what the command really did: a file's contents, or Voxa speaking up later by herself."""
    verify = case.verify
    notes: list[str] = []
    if verify.get("file"):
        problem = check_file(verify)
        if problem:
            return GradeResult(0, False, f"She said it was done, but {problem}", "outcome")
        notes.append(f"{Path(str(verify['file'])).name} checked")
    if verify.get("says_within"):
        seconds = float(verify["says_within"])
        phrase = str(verify.get("says", ""))
        log(f"Waiting up to {seconds:.0f}s for Voxa to speak up about: {phrase}")
        try:
            spoke, heard = hear_until(phrase, seconds, settings, whisper, capture)
        except Exception as exc:  # noqa: BLE001 - hardware boundary
            return GradeResult(0, False, f"Could not listen for the outcome: {exc}", "outcome")
        if not spoke:
            said = f" Heard instead: {heard}" if heard else " She said nothing."
            return GradeResult(0, False, f"Nothing about \"{phrase}\" was spoken within {seconds:.0f} seconds.{said}", "outcome")
        notes.append(f"she spoke up by herself: \"{heard}\"")
    return GradeResult(grade.score, True, f"{grade.notes} Outcome verified: {'; '.join(notes)}.", "outcome")


def _require_action(
    case: TestCase, grade: GradeResult, records: list[dict[str, Any]], transcript: str = "",
) -> tuple[GradeResult, dict[str, Any] | None]:
    """Fail a spoken case unless its matching tool really ran and its reply was heard."""
    matching = [record for record in records if str(record.get("tool", "")) == case.expected_tool]
    if not matching:
        observed = sorted({str(record.get("tool", "")) for record in records if record.get("tool")})
        suffix = f" Observed: {', '.join(observed)}." if observed else " No tool action was logged."
        return GradeResult(0, False, f"Expected action {case.expected_tool} was not logged.{suffix}", "action_log"), None
    record = matching[-1]
    actual_args = record.get("args") if isinstance(record.get("args"), dict) else {}
    wrong_args = {
        key: {"expected": value, "actual": actual_args.get(key)}
        for key, value in case.expected_args.items()
        if not _same_argument(value, actual_args.get(key))
    }
    if wrong_args:
        return GradeResult(0, False, f"Action {case.expected_tool} used the wrong arguments: {wrong_args}.", "action_log"), record
    if record.get("ok") is not True:
        detail = str(record.get("detail") or record.get("speech") or "unknown failure")
        return GradeResult(0, False, f"Action {case.expected_tool} failed: {detail}", "action_log"), record
    spoken = str(record.get("speech") or "")
    if grade.method == "keywords" and not grade.passed and spoken and reply_match(spoken, transcript) >= REPLY_MATCH:
        # The harness's own ear slipped on a word ("fifty-six" written "50-60"). The sentence was
        # heard, and Voxa's log holds her exact words: check the expected words there instead.
        logged = keyword_grade(case, spoken)
        if logged.passed:
            grade = GradeResult(100, True, f"Heard the reply; the expected words are in what Voxa said: \"{spoken}\"", "keywords")
    if not spoken and not transcript:
        return GradeResult(100, True, "Voxa did it without speaking, as that command is meant to.", "action_log"), record
    if spoken and transcript and not case.verify.get("reply_in_other_language"):
        # (a reply spoken in another language cannot be compared through an English transcript)
        share = reply_match(spoken, transcript)
        # A station that starts at once plays over the end of "Playing …": half the words is enough there.
        needed = PLAYBACK_REPLY_MATCH if case.after == "stop_media" else REPLY_MATCH
        if share < needed:
            return GradeResult(
                int(share * 100), False,
                f"Voxa said \"{spoken}\" but only {share:.0%} of it was heard through the microphone.",
                "reply_match",
            ), record
    return grade, record


def _same_argument(expected: Any, actual: Any) -> bool:
    """Arguments match regardless of letter case; ``a|b`` in a case lists accepted spellings."""
    wanted = {part.strip().casefold() for part in str(expected).split("|")}
    return str(actual if actual is not None else "").strip().casefold() in wanted


def _with_wake_word(prompt: str, wake_word: str) -> str:
    wake = wake_word.strip()
    if not wake or prompt.casefold().startswith(wake.casefold()):
        return prompt
    return f"{wake}, {prompt}"
