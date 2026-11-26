"""Grade Voxa's REAL answers offline: real TTS, real Whisper, real Ollama, no speakers or mics.

Every case is spoken with the TTS VoxaTest uses for live runs into a WAV, fed through Voxa's real
hearing path (wake handling, segmentation, Whisper, cleaning) and the real request pipeline, exactly
as :func:`tools.replay_request` runs the pipeline itself but listening instead of typing. The spoken
reply, the heard text, the route/tool and the per-stage timings come back from Voxa's own code and
action log; VoxaTest's own grader grades them.

It never opens a microphone, never plays sound and never lands a window on the owner's desktop: it
re-runs itself inside ``xvfb-run`` when needed. Command cases run against the command router's fake
executor (:data:`DRY_TOOLS`), so a "turn the volume up" case can never touch the machine's volume.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cases import TestCase
from .config import HarnessSettings
from .grader import GradeResult, grade_reply
from .report import format_results, report_paths, write_reports
from .runner import _require_action, _with_wake_word

CHUNK_SECONDS = 0.04  # 40 ms of 16 kHz mono s16le, like the live capture callback
TAIL_SILENCE_SECONDS = 1.5  # after the spoken prompt: silence_ms ends the utterance
CASE_TIMEOUT_SECONDS = 90.0
WHISPER_LOAD_SECONDS = 300.0

# Tools the fake executor runs for real because they only compute locally (pure, no side effects).
REAL_DRY_TOOLS = frozenset({"calculate"})
# Tools the fake executor answers with canned speech so the spoken reply stays natural and
# non-empty; the case itself is graded on the router's record, not on this sentence.
DRY_SPEECH = {
    "set_timer": lambda a: f"Timer set for {a.get('duration', '')}." if a.get("duration") else "Timer set.",
    "set_reminder": lambda a: f"I'll remind you to {a.get('text', '')} {a.get('when', '')}.".strip(),
    "take_note": lambda a: f"Noted: {a.get('text', '')}.".strip(),
    "read_notes": lambda a: "Reading your notes.",
    "open_notes": lambda a: "Opening your notes.",
    "delete_last_note": lambda a: "Deleted your last note.",
    "list_reminders": lambda a: "Here are your reminders.",
    "cancel_reminders": lambda a: "Your timers and reminders are cancelled.",
    "open_app": lambda a: f"Opening {a.get('name', '')}.".strip(),
    "close_app": lambda a: f"Closing {a.get('name', '')}.".strip(),
    "system_volume": lambda a: "Louder." if a.get("action") in ("louder", "up") else "Quieter.",
    "list_add": lambda a: f"Added {a.get('item', '')} to your {a.get('list', '')} list.".strip(),
}


@dataclass(slots=True)
class OfflineRunResult:
    """One offline case, in the same shape as the live run's RunResult plus timings and reply."""

    case_id: str
    category: str
    prompt: str
    transcript: str  # the reply Voxa spoke, graded like the live run's microphone transcript
    grade: GradeResult
    error: str = ""
    duration_seconds: float = 0.0
    action_record: dict[str, Any] | None = None
    expected_tool: str = ""
    heard: str = ""
    capability: int = 0
    reply: str = ""
    fail_reason: str = ""
    timings: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "prompt": self.prompt,
            "transcript": self.transcript,
            "reply": self.reply,
            "score": self.grade.score,
            "passed": self.grade.passed,
            "notes": self.grade.notes,
            "method": self.grade.method,
            "error": self.error,
            "fail_reason": self.fail_reason,
            "duration_seconds": round(self.duration_seconds, 2),
            "expected_tool": self.expected_tool,
            "heard": self.heard,
            "capability": self.capability,
            "timings": self.timings,
            "action_record": self.action_record,
        }


def _own_virtual_display() -> None:
    """Run the whole offline run under Xvfb so no window ever lands on the owner's desktop."""
    if os.environ.get("VOXATEST_OFFLINE_XVFB") == "1" or shutil.which("xvfb-run") is None:
        return
    os.environ["VOXATEST_OFFLINE_XVFB"] = "1"
    cmd = ["xvfb-run", "-a", sys.executable, "-m", "voxatest", *sys.argv[1:]]
    sys.exit(subprocess.call(cmd, env=os.environ))


def _import_gi():
    """Import PyGObject, reaching the distro copy when a plain venv lacks it."""
    try:
        import gi
    except ImportError:
        import site

        for extra in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.14/dist-packages"):
            if (Path(extra) / "gi").is_dir():
                site.addsitedir(extra)
                break
        import gi
    return gi


def _find_config(explicit: str | None) -> Path | None:
    """Voxa's real config: the Flatpak app's if present, else the user's (same as replay_request)."""
    from .config import choose_voxa_config, voxa_config_candidates

    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
    return choose_voxa_config(voxa_config_candidates())


def synthesize_utterance(text: str, settings: HarnessSettings, log) -> tuple[str, str]:
    """Speak ``text`` with the live-run TTS to an audio file: (path, engine).

    Edge TTS first (exactly what live runs use through SpeechService.prepare), then the app's own
    Google voice fallback, then its offline eSpeak fallback. No sound is ever played.
    """
    from voxa.speech import (
        SpeechService,
        _download_google_voice,
        espeak_voice,
        google_voice_params,
    )

    speech = SpeechService(allow_online_fallback=False, allow_offline_fallback=False)
    item = speech.prepare(text, settings.tts_rate, settings.tts_voice)
    item.ready.wait(settings.speak_timeout_seconds)
    if not item.error and item.path:
        return item.path, "edge-tts"
    log(f"Edge TTS unavailable for offline speech ({item.error or 'no audio'}); using Voxa's fallback voices.")

    language, tld = google_voice_params(settings.tts_voice)
    google_path = str(Path(tempfile.mkdtemp(prefix="voxa-offline-tts-")) / "say.mp3")
    try:
        # Google's socket can stall without ever erroring (seen mid-run); a hung
        # request must not hang the whole run, so race it with a deadline.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_download_google_voice, text, google_path, lang=language, tld=tld).result(
                timeout=min(20.0, settings.speak_timeout_seconds)
            )
        return google_path, "google-tts"
    except Exception as exc:  # noqa: BLE001 - network/service boundary
        log(f"Google voice failed for offline speech ({exc}); eSpeak NG next.")

    wav_path = str(Path(google_path).parent / "say.wav")
    subprocess.run(
        ["espeak-ng", "--stdout", "-v", espeak_voice(settings.tts_voice), "-s", str(settings.tts_rate), text],
        stdout=open(wav_path, "wb"),
        check=True,
        timeout=30,
    )
    return wav_path, "espeak-ng"


def decode_pcm(path: str) -> bytes:
    """Decode any audio file to the 16 kHz mono s16le PCM the app's capture pipeline feeds."""
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return data.stdout


def _rms(pcm: bytes) -> float:
    """One chunk's level exactly like voxa.audio.AudioCapture._rms computes it."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2")
    if not samples.size:
        return 0.0
    stride = max(1, samples.size // 2048)
    chosen = samples[::stride].astype(np.float64)
    return float(np.sqrt(np.mean(chosen * chosen)))


class SpeechRecorder:
    """Records every piece of speech the pipeline produces, with timestamps; no sound, no network."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.first_text_at: float | None = None
        self.started_at: float | None = None

    def reset(self) -> None:
        self.spoken = []
        self.first_text_at = None
        self.started_at = None

    def _note(self, text: str) -> None:
        if str(text).strip():
            self.spoken.append(str(text))
            if self.first_text_at is None:
                self.first_text_at = time.monotonic()

    def stop(self) -> None:
        pass

    def position(self) -> float:
        """Speech is spoken instantly: nothing is ever mid-playback."""
        return 0.0

    def speak(self, text, rate, voice, *, on_started, on_done, on_error, on_words=None, before_play=None) -> None:
        self._note(text)
        if on_started is not None:
            on_started()
        if on_done is not None:
            on_done()

    def prepare(self, text, rate, voice, on_ready=None):
        class _Item:
            ready = type("_R", (), {"wait": lambda self, timeout=None: True})()
            error = ""
            path = ""
            words = ()
            cancelled = type("_C", (), {"is_set": lambda self: False})()

        item = _Item()
        self._note(text)
        return item

    def speak_prepared(self, item, *, on_started, on_done, on_error, on_words=None, before_play=None) -> None:
        if on_started is not None:
            on_started()
        if on_done is not None:
            on_done()


class FakeAudio:
    """A microphone that never captures: the conversation controller is fed WAV PCM by hand."""

    def __init__(self) -> None:
        self.is_active = False
        self.reference_source = None
        self.pulse_source = None
        self.canceller = None

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.is_active = True

    def stop(self) -> None:
        self.is_active = False


def fail_reason(case: TestCase, grade: GradeResult, reply: str, heard: str, timed_out: bool) -> str:
    """Why a case failed, in the vocabulary the task prints: not heard / wrong route / empty reply
    / wrong answer / timeout."""
    if grade.passed:
        return ""
    if not heard.strip():
        return "not heard"
    if not reply.strip():
        return "empty reply"
    if case.expected_tool and grade.method == "action_log":
        return "wrong route"
    if timed_out:
        return "timeout"
    if grade.method in {"keywords", "model", "fallback"}:
        return "wrong answer"
    return grade.method or "failed"


def grade_case(
    case: TestCase, reply: str, heard: str, records: list[dict[str, Any]], timed_out: bool,
    client: Any, model: str,
) -> tuple[GradeResult, dict[str, Any] | None]:
    """VoxaTest's own grading, offline: the grader first, the action log for commands, style last."""
    action_record: dict[str, Any] | None = None
    if not reply.strip():
        grade = GradeResult(0, False, "Voxa spoke nothing.", "empty")
    elif case.expected_tool and not (case.expected_contains or case.expected_keywords):
        grade = GradeResult(100, True, "Checked against Voxa's action log and spoken reply.", "action_log")
    else:
        grade = grade_reply(case, reply, client, model)
    if case.expected_tool:
        grade, action_record = _require_action(case, grade, records, reply)
    if grade.passed and case.reply_style:
        problems = spoken_style_problems(case, reply)
        if problems:
            grade = GradeResult(0, False, f"The reply is not a short spoken sentence: {'; '.join(problems)}.", "style")
    return grade, action_record


def spoken_style_problems(case: TestCase, reply: str) -> list[str]:
    """What is un-spoken about ``reply`` for a case that asked for spoken style (issue #72 offline)."""
    style = case.reply_style
    problems: list[str] = []
    if not reply.strip():
        return ["nothing was spoken"]
    lowered = reply.casefold()
    if style.get("no_markdown") and any(mark in reply for mark in ("*", "#", "`", "|", "```")):
        problems.append("it contains markdown")
    if any(marker in lowered for marker in ("thinking process", "<think", "</think", "reasoning:")):
        problems.append("it reads the model's reasoning")
    max_words = int(style.get("max_words") or (40 if style.get("short") else 0))
    if max_words and len(reply.split()) > max_words:
        problems.append(f"it is {len(reply.split())} words long")
    return problems


def run_offline(
    settings: HarnessSettings,
    cases: list[TestCase],
    *,
    reports_dir: str,
    json_path: str | None = None,
    text_path: str | None = None,
    failure_path: str | None = None,
    config: str | None = None,
    case_timeout: float = CASE_TIMEOUT_SECONDS,
) -> int:
    """Run ``cases`` through real Voxa with no speakers or microphone; same reports as a live run."""
    _own_virtual_display()
    if not cases:
        print("No test cases to run.", file=sys.stderr, flush=True)
        return 1

    os.environ.setdefault("PULSE_SERVER", "unix:/nonexistent-voxa-offline")
    os.environ["VOXA_NO_WELCOME"] = "1"
    os.environ["GSK_RENDERER"] = "cairo"

    real_config = _find_config(config)
    if real_config is None:
        print("offline run: no Voxa config.json found", file=sys.stderr, flush=True)
        return 2
    print(f"offline run: config {real_config}", flush=True)

    real_ollama_models = Path.home() / ".ollama" / "models"
    if real_ollama_models.is_dir():
        os.environ.setdefault("OLLAMA_MODELS", str(real_ollama_models))

    # Whisper and the Silero VAD model live in the owner's Hugging Face cache: the sandboxed
    # HOME below must not force a re-download of models that are already on disk.
    real_hf_home = Path.home() / ".cache" / "huggingface"
    if real_hf_home.is_dir():
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    tmp = Path(tempfile.mkdtemp(prefix="voxa-offline-"))
    (tmp / "config/voxa").mkdir(parents=True)
    (tmp / "config/voxa/config.json").write_bytes(real_config.read_bytes())
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_STATE_HOME"] = str(tmp / "state")
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
    os.environ["HOME"] = str(tmp)

    gi = _import_gi()
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, GLib, Gtk

    if not Gtk.init_check():
        print("offline run: no display (run under xvfb-run)", file=sys.stderr, flush=True)
        return 2
    Adw.init()

    from voxa import window as window_module  # noqa: F401  (after env, like replay_request)
    from voxa.agent.result import ToolResult
    from voxa.audio import AudioDevice
    from voxa.ui.state import AssistantState
    from voxa.window import MainWindow

    for name in (
        "_detect_hardware_async",
        "_refresh_ollama_models",
        "_refresh_devices",
        "_start_ai_server_async",
        "_restart_ai_server_async",
        "_stop_ai_server_async",
        "_prepare_microphone",
    ):
        setattr(MainWindow, name, lambda *a, **k: None)

    # The command router's fake executor: routing, logging and replying stay real; the machine is
    # not touched. Pure local calculators run for real.
    real_tool_result = MainWindow._tool_result

    def fake_tool_result(self, call):
        if call.tool in REAL_DRY_TOOLS:
            return real_tool_result(self, call)
        speech = DRY_SPEECH.get(call.tool, lambda _a: "Done.")(dict(call.args))
        return ToolResult.success(speech)

    MainWindow._tool_result = fake_tool_result

    app = Adw.Application(application_id="io.github.crhy.voxa.offline")
    app.register(None)
    window = MainWindow(app)
    window.present()

    recorder = SpeechRecorder()
    window.speech = recorder
    window.audio = FakeAudio()
    window.devices = [AudioDevice(identifier="mic", name="Microphone", device=None)]
    notices: list[str] = []
    window._toast = lambda text: notices.append(str(text))

    prompt_at: list[float] = []
    original_prompt = window._on_conversation_prompt

    def timed_prompt(text: str) -> bool:
        prompt_at.clear()
        prompt_at.append(time.monotonic())
        return original_prompt(text)

    window._on_conversation_prompt = timed_prompt

    client = settings.make_client()
    reports = Path(reports_dir)
    json_out, text_out = report_paths(
        reports,
        json_path=Path(json_path) if json_path else None,
        text_path=Path(text_path) if text_path else None,
    )
    failure_out = Path(failure_path) if failure_path else None

    action_log = window.action_log
    loop = GLib.MainLoop()
    results: list[OfflineRunResult] = []
    state: dict[str, Any] = {"index": 0, "phase": "idle", "case_start": 0.0, "timed_out": False,
                            "done": threading.Event(), "error": ""}

    def start_conversation() -> str:
        if not window.whisper.ready or not window.wake_whisper.ready:
            return ""  # pump() waits for it
        if not window.assistant.is_active:
            if not window.assistant.activate():
                return f"could not go ACTIVE: {getattr(window, '_start_failure', '')}"
        return ""

    def speak_and_feed(case: TestCase) -> None:
        """Worker: TTS the prompt, then feed its PCM to the live conversation controller."""
        try:
            prompt = _with_wake_word(case.prompt, settings.wake_word)
            path, _engine = synthesize_utterance(prompt, settings, lambda message: print(message, flush=True))
            pcm = decode_pcm(path)
            with contextlib.suppress(OSError):
                Path(path).unlink()
            controller = window.conversation
            if controller is None:
                state["error"] = "conversation mode is not listening"
                state["done"].set()
                return
            started = time.monotonic()
            offset = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + int(16000 * 2 * CHUNK_SECONDS)]
                offset += len(chunk)
                controller.feed(chunk, _rms(chunk))
                ahead = started + offset / 32000.0
                wait = ahead - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            silence = b"\0" * int(16000 * 2 * CHUNK_SECONDS)
            for _ in range(int(TAIL_SILENCE_SECONDS / CHUNK_SECONDS)):
                controller.feed(silence, 0.0)
                time.sleep(CHUNK_SECONDS)
            state["done"].set()
        except Exception as exc:  # noqa: BLE001 - harness boundary, reported as the case's error
            state["error"] = str(exc)
            state["done"].set()

    def finish_case(case: TestCase, grade: GradeResult, action_record, reply: str, heard: str,
                    records: list[dict[str, Any]], error: str) -> None:
        timed_out = state["timed_out"]
        reason = fail_reason(case, grade, reply, heard, timed_out)
        timings: dict[str, int] = {}
        for record in reversed(records):
            if isinstance(record.get("timings"), dict):
                timings.update(record["timings"])
                break
        if prompt_at and recorder.first_text_at is not None:
            timings["first_reply_ms"] = int((recorder.first_text_at - prompt_at[0]) * 1000)
        if state["case_start"]:
            timings["total_ms"] = int((time.monotonic() - state["case_start"]) * 1000)
        route = ""
        tool = ""
        for record in records:
            route = str(record.get("route", route))
            tool = str(record.get("tool", tool)) or tool
        result = OfflineRunResult(
            case_id=case.id,
            category=case.category,
            prompt=case.prompt,
            transcript=reply,
            grade=grade,
            error=error,
            duration_seconds=time.monotonic() - state["case_start"],
            action_record=action_record,
            expected_tool=case.expected_tool,
            heard=heard,
            capability=case.capability,
            reply=reply,
            fail_reason=reason,
            timings=timings,
        )
        results.append(result)
        status = "ERROR" if error else ("PASS" if grade.passed else "FAIL")
        print(
            f"[{len(results)}/{len(cases)}] {status} {case.id} score {grade.score} "
            f"({result.duration_seconds:.1f}s)",
            flush=True,
        )
        if reason:
            print(f"    why: {reason}: {grade.notes}", flush=True)
        if heard:
            print(f"    heard: {heard}", flush=True)

    def pump() -> bool:
        """Advance the run; True while cases remain, False when done (the loop quits)."""
        index = state["index"]
        if index >= len(cases):
            return False
        case = cases[index]
        phase = state["phase"]

        if phase == "idle":
            if not window.whisper.ready or not window.wake_whisper.ready:
                return True
            problem = start_conversation()
            if problem:
                finish_case(case, GradeResult(0, False, problem, "error"), None, "", "", [], problem)
                state["index"] += 1
                state["phase"] = "idle"
                return True
            state.update(phase="speak", case_start=time.monotonic(), timed_out=False,
                        error="", done=threading.Event())
            recorder.reset()
            state["offset"] = len(action_log.read())
            threading.Thread(target=speak_and_feed, args=(case,), name="offline-speak", daemon=True).start()
            return True

        if phase == "speak":
            # The feed thread ended with a broken harness: report it, do not judge the case.
            if state["done"].is_set() and state["error"]:
                finish_case(case, GradeResult(0, False, f"Offline harness: {state['error']}", "error"),
                            None, "", "", [], state["error"])
                state["index"] += 1
                state["phase"] = "idle"
                return True
            state["phase"] = "listen"
            return True

        # phase == listen: wait for Voxa to hear, route, answer and speak it (or time out).
        # Done when the reply was spoken and the assistant is idle again (READY, or LISTENING
        # in the follow-up window); the request is logged as soon as it is heard, so a reply
        # plus an idle assistant can only mean the answer is complete.
        records = action_log.read()[state["offset"]:]
        reply = " ".join(recorder.spoken).strip()
        heard = " / ".join(str(record.get("heard", "")) for record in records if record.get("heard"))
        settled = window.assistant_model.state in (AssistantState.READY, AssistantState.LISTENING)
        complete = bool(reply) and bool(records) and any(record.get("heard") for record in records) and settled
        if not complete and time.monotonic() - state["case_start"] > case_timeout:
            state["timed_out"] = True
            complete = True
        if not complete:
            return True

        grade, action_record = grade_case(case, reply, heard, records, state["timed_out"], client, settings.model)
        error = state["error"] or (notices[-1] if notices and not grade.passed else "")
        finish_case(case, grade, action_record, reply, heard, records, error)
        state["index"] += 1
        state["phase"] = "idle"
        return True

    def tick() -> bool:
        if pump():
            write_reports(results, reports, json_path=json_out, text_path=text_out,
                          failure_path=failure_out, complete=False)
            return True
        loop.quit()
        return False

    GLib.timeout_add(50, tick)
    print(
        f'offline run: wake word "{settings.wake_word}", model {settings.model} at {settings.url}, '
        f"whisper {settings.whisper_model}, {len(cases)} cases",
        flush=True,
    )
    loop.run()

    write_reports(results, reports, json_path=json_out, text_path=text_out,
                  failure_path=failure_out, complete=True)

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    first_reply = [r.timings.get("first_reply_ms", 0) for r in results if r.timings.get("first_reply_ms")]
    hear_ms = [r.timings.get("hear_ms", 0) for r in results if r.timings.get("hear_ms")]
    passed = sum(1 for result in results if result.grade.passed)
    print(format_results(results, complete=True), flush=True)
    print(
        f"offline run: {passed}/{len(results)} passed; time-to-first-reply-text median "
        f"{_median([v / 1000 for v in first_reply]):.2f}s worst "
        f"{(max(first_reply) / 1000 if first_reply else 0.0):.2f}s; hear_ms median "
        f"{_median([float(v) for v in hear_ms]):.0f}ms",
        flush=True,
    )
    print(f"JSON report: {json_out}", flush=True)
    print(f"Text report: {text_out}", flush=True)

    window._closing = True
    window.destroy()
    for _ in range(20):
        GLib.MainContext.default().iteration(False)
    if os.environ.get("VOXATEST_OFFLINE_KEEP_TMP") == "1":
        print(f"offline run: sandbox kept at {tmp}", flush=True)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == len(results) else 1
