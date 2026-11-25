from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from voxa.agent.actionlog import default_path

from . import isolation
from .cases import load_cases, save_cases
from .commands import (
    DEFAULT_COMMAND_CASES_PATH,
    format_markdown,
    load_command_cases,
    run_commands,
    summarize,
)
from .comprehension import check as check_comprehension
from .comprehension import comprehension_main, score_case
from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_FAILURE_REPORT_PATH,
    DEFAULT_REPORTS_DIR,
    load_settings,
    refuses_echo_mode,
)
from .generate_cases import generate_cases
from .grader import GradeResult
from .latency import format_latency, latency_report, load_records
from .report import emit_event, format_results, report_paths, write_reports
from .runner import RunResult, run_case


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voxatest", description="Speak-test Voxa end to end.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate spoken test cases with the local model.")
    generate.add_argument("--count", type=int, default=30)
    generate.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    generate.add_argument("--batch-size", type=int, default=10)

    run = subparsers.add_parser("run", help="Run cases through real TTS, microphone capture, Whisper, and grading.")
    run.add_argument("--cases", default=str(DEFAULT_DATA_PATH))
    run.add_argument("--limit", type=int)
    run.add_argument("--reports", help=f"Report directory (default: {DEFAULT_REPORTS_DIR})")
    run.add_argument("--json-report")
    run.add_argument("--text-report")
    run.add_argument(
        "--failure-report",
        help=f"Git-friendly latest-failures Markdown file (default: {DEFAULT_FAILURE_REPORT_PATH})",
    )
    run.add_argument("--input-device", default=None)
    run.add_argument("--output-device", default=None)
    run.add_argument("--wake-word", default=None)
    run.add_argument("--deadline-seconds", type=float, help="Stop the run after this many seconds (0 means no limit).")
    run.add_argument("--retries", type=int, help="Retry a case when the microphone heard no reply.")
    run.add_argument("--end-silence", type=float, help="Seconds of silence that end a spoken reply.")
    run.add_argument(
        "--talk-over",
        action="store_true",
        help=(
            "Live pacing (issue #72, task v11 step 6): do not wait for a long reply to finish. "
            "On replies longer than ~4 s VoxaTest speaks the next question over her 2-4 s in, "
            "measures her stop latency, and whether she then answered the new question."
        ),
    )
    run.add_argument(
        "--talk-over-seconds",
        type=float,
        default=None,
        help="How far into her reply to talk over her (default 3.0; the task says 2-4 s).",
    )
    run.add_argument(
        "--settle-seconds",
        type=float,
        default=None,
        help="Seconds to settle before the next prompt after her reply ends (issue #72: 1.5 -> 1.0).",
    )
    run.add_argument("--start-at", type=int, default=1, help="1-based index of the first case to run.")
    run.add_argument(
        "--max-silent",
        type=int,
        help="Stop the run after this many consecutive cases with no reply (0 disables).",
    )
    run.add_argument(
        "--offline-audio",
        action="store_true",
        help="No speakers and no microphone: render each prompt with the test voice, push the audio "
             "through the hearing bench's pipeline and Voxa's command router, and score comprehension.",
    )
    run.add_argument("--variants", default="clean",
                     help="offline audio: clean,quiet,noisy,music,fast,slow,voice2 (comma separated)")
    run.add_argument("--seed", type=int, default=0, help="offline audio: the noise seed (deterministic)")
    run.add_argument("--bench-data", default=None,
                     help="offline audio: the bench data root (default /home/rhy/voxa-bench-data)")
    run.add_argument("--echo-mode", choices=("off", "voxa", "webrtc"), default=None,
                     help="offline audio: which echo stages the pipeline runs. Default: voxa for the hearing "
                          "cases (Voxa is speaking) and off for the others (she is silent)")
    run.add_argument("--wake-mode", choices=("separate", "single"), default="separate",
                     help="offline audio: \"Voxa.\" in its own breath, then the question (separate), "
                          "or the one-breath prompt the live harness speaks today (single)")

    commands = subparsers.add_parser("commands", help="Bench the voice-command router in milliseconds, without audio.")
    commands.add_argument("--cases", default=str(DEFAULT_COMMAND_CASES_PATH))
    commands.add_argument("--report-dir", default=str(DEFAULT_REPORTS_DIR))
    commands.add_argument("--action-log", help="Optional Voxa action log to report real-world misses from.")
    commands.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Repeat the deterministic corpus this many times (default: 10).",
    )
    commands.add_argument(
        "--min-checks-per-hour",
        type=float,
        default=1000.0,
        help="Fail if measured command-router throughput is below this rate (default: 1000).",
    )
    commands.add_argument(
        "--capability-report",
        help="Also write the Markdown audit to this stable path.",
    )

    status = subparsers.add_parser("capabilities", help="Write the capability status (issue 102) from the run reports.")
    status.add_argument("--reports", default=str(DEFAULT_REPORTS_DIR))
    status.add_argument("--markdown")
    status.add_argument("--html")
    status.add_argument("--since", default="", help="Only count reports from this date on, as YYYYMMDD.")
    status.add_argument("--stamp", default="")

    latency = subparsers.add_parser("latency", help="Report command latency stats from the action log.")
    latency.add_argument(
        "--action-log",
        default=str(default_path()),
        help="Action log path (default: the Voxa action log).",
    )
    latency.add_argument("--last", type=int, help="Only consider the last N records.")

    # Both take their own options, so the whole tail is handed to them untouched.
    comprehension = subparsers.add_parser(
        "comprehension", help="Score what Voxa heard against what was said in a written run report.",
    )
    comprehension.add_argument("--report", required=True, help="the run report JSON written by `voxatest run`")
    comprehension.add_argument("--out", help="write the re-scored report here")
    comprehension.add_argument("--wake-word", default=None, help="wake word to strip from both sides")
    comprehension.add_argument("--top", type=int, default=None, help="how many confusion rows to keep")
    comprehension.add_argument("--json", action="store_true", help="print the comprehension block as JSON")

    check = subparsers.add_parser(
        "check", help="Assert comprehension conditions on a run report (exits 1 on any failure).",
    )
    check.add_argument("report", help="the run report JSON written by `voxatest run`")
    check.add_argument("conditions", nargs="+", help='conditions like hearing_word_accuracy>=0.9')

    soak = subparsers.add_parser(
        "soak",
        help="Live soak: VoxaTest speaks long varied replies for hours with nobody talking, and counts "
             "every self-command and self-pause from Voxa's action log and barge-in events.",
    )
    soak.add_argument("--hours", type=float, default=1.0, help="how long to run (default 1 hour)")
    soak.add_argument("--seed", type=int, default=0, help="the plan seed: the same seed replays the same replies")
    soak.add_argument("--gap", type=float, default=2.5, help="seconds of silence between replies")
    soak.add_argument("--voices", default="en-US-GuyNeural,en-US-JennyNeural",
                      help="comma separated TTS voices to rotate through")
    soak.add_argument("--action-log", help="Voxa's action log to count from (default: the one Voxa writes)")
    soak.add_argument("--json-report")
    soak.add_argument("--text-report")
    soak.add_argument("--plan", action="store_true", help="print the plan and stop: plays no sound")
    soak.add_argument("--yes", action="store_true",
                      help="start the live run. It speaks through the speakers: rhY must know first")

    doctor = subparsers.add_parser(
        "doctor", help="Check the setup before a live hearing run (PASS / WARN / FAIL lines).",
    )
    doctor.add_argument("--dry-run", action="store_true",
                        help="ask nothing of the sound server or the microphone")
    doctor.add_argument("--hours", type=float, default=1.0, help="the soak length to plan for")
    doctor.add_argument("--seed", type=int, default=0, help="the soak seed to plan for")
    doctor.add_argument("--gap", type=float, default=2.5, help="seconds of silence between replies")
    doctor.add_argument("--json", action="store_true", help="print the checks as JSON")

    return parser.parse_args(argv)


def _apply_cli_overrides(settings, args: argparse.Namespace) -> None:
    input_device = getattr(args, "input_device", None)
    output_device = getattr(args, "output_device", None)
    wake_word = getattr(args, "wake_word", None)
    deadline_seconds = getattr(args, "deadline_seconds", None)
    retries = getattr(args, "retries", None)
    end_silence = getattr(args, "end_silence", None)
    max_silent = getattr(args, "max_silent", None)

    if input_device is not None:
        settings.input_device = input_device
    if output_device is not None:
        settings.output_device = output_device
    if wake_word is not None:
        settings.wake_word = wake_word
    if deadline_seconds is not None:
        settings.run_deadline_seconds = deadline_seconds
    if retries is not None:
        settings.retries = retries
    if end_silence is not None:
        settings.end_silence_seconds = end_silence
    if max_silent is not None:
        settings.max_consecutive_no_reply = max_silent
    # Issue #72, task v11 step 6: live talk-over pacing and the shortened settle.
    settings.talk_over = bool(getattr(args, "talk_over", False)) or settings.talk_over
    talk_over_seconds = getattr(args, "talk_over_seconds", None)
    if talk_over_seconds is not None:
        settings.talk_over_seconds = talk_over_seconds
    settle_seconds = getattr(args, "settle_seconds", None)
    if settle_seconds is not None:
        settings.settle_seconds = settle_seconds


def _load_whisper(settings, log):
    from voxa.transcription import WhisperService

    whisper = WhisperService()
    ready = threading.Event()
    failed = threading.Event()
    errors: list[str] = []

    def on_ready(model_name: str, backend: str) -> None:
        log(f"Whisper ready: {model_name} ({backend})")
        ready.set()

    def on_error(message: str) -> None:
        log(f"Whisper load failed: {message}")
        errors.append(message)
        failed.set()

    whisper.load_async(settings.whisper_model, on_ready, on_error)

    deadline = time.monotonic() + settings.whisper_load_timeout_seconds
    while not ready.is_set() and not failed.is_set():
        if time.monotonic() > deadline:
            return whisper, f"Whisper load timed out after {settings.whisper_load_timeout_seconds} seconds."
        time.sleep(0.25)

    if failed.is_set():
        return whisper, errors[-1] if errors else "Whisper load failed."
    return whisper, ""


def _start_glib_main_loop():
    """Run GLib callbacks while the command-line harness blocks on test cases."""
    from gi.repository import GLib

    loop = GLib.MainLoop()
    thread = threading.Thread(target=loop.run, name="voxatest-glib", daemon=True)
    thread.start()
    return loop, thread


def _run_case_safely(case, settings, client, whisper, capture, speech, log) -> RunResult:
    """Run one case, turning any harness crash into a recorded error result."""
    try:
        return run_case(case, settings, client, whisper, capture, speech, log=log)
    except Exception as exc:  # noqa: BLE001 - one broken case must never abort the run
        grade = GradeResult(0, False, "Harness error.", "error")
        return RunResult(case.id, case.category, case.prompt, "", grade, str(exc))


def _run_offline(settings, args: argparse.Namespace) -> int:
    """``voxatest run --offline-audio``: comprehension with no speakers and no microphone.

    The exit code says the measurement ran. It is not the verdict on the numbers: the cases' own
    pass/fail is a comprehension measurement, not a gate, and thresholds are asserted with
    ``voxatest check REPORT "hearing_word_accuracy>=0.9"`` exactly like ``voxa.bench check``.
    """
    from . import offline

    try:
        variants = offline.parse_variants(args.variants)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    try:
        cases = offline.load_offline_cases(Path(args.cases))
    except (OSError, ValueError) as exc:
        print(f"Cannot read {args.cases}: {exc}", file=sys.stderr, flush=True)
        return 2
    start_at = max(1, int(args.start_at or 1))
    if start_at > 1:
        cases = cases[start_at - 1 :]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No test cases found in {args.cases}.", file=sys.stderr, flush=True)
        return 1

    data_root = Path(args.bench_data or offline.DEFAULT_DATA_ROOT)
    total = len(cases) * len(variants)
    print(
        f'Offline audio: {len(cases)} cases x {len(variants)} variant(s) [{", ".join(v.name for v in variants)}] '
        f'seed {args.seed}, wake word "{settings.wake_word}", test voice {settings.tts_voice}, '
        f'bench data {data_root}',
        flush=True,
    )
    results, block = offline.run_offline(
        cases, settings, variants, args.seed, data_root,
        model=settings.whisper_model, echo_mode=args.echo_mode, wake_mode=args.wake_mode,
    )

    reports = Path(args.reports or settings.reports_dir)
    json_path, text_path = report_paths(
        reports,
        json_path=Path(args.json_report) if args.json_report else None,
        text_path=Path(args.text_report) if args.text_report else None,
    )
    failure_path = Path(args.failure_report or str(settings.failure_report_path))
    paths = write_reports(
        results, reports, json_path=json_path, text_path=text_path, failure_path=failure_path,
        complete=True, wake_word=settings.wake_word, extra={"offline": block},
    )
    for index, result in enumerate(results, start=1):
        emit_event(
            "case_result", index=index, total=total, id=result.case_id, category=result.category,
            prompt=result.prompt, status="PASS" if result.grade.passed else "FAIL",
            reply="", heard=result.heard, layer=score_case(result.to_dict(), settings.wake_word)["layer"],
            notes=result.grade.notes, seconds=round(result.duration_seconds, 2),
        )
    print(format_results(results, complete=True, wake_word=settings.wake_word, offline=block), flush=True)
    print(f"JSON report: {paths['json_path']}", flush=True)
    print(f"Text report: {paths['text_path']}", flush=True)
    print(f"Failure report: {paths['failure_path']}", flush=True)
    if not results:
        return 1
    return 0


def _run(
    settings,
    cases: list,
    json_report: str | None,
    text_report: str | None,
    reports_dir: str,
    failure_report: str,
    first_index: int = 1,
) -> int:
    if settings.output_device:
        os.environ.setdefault("GST_AUDIOSINK", settings.output_device)

    if refuses_echo_mode(settings.voxa_echo_mode, settings.output_device):
        print(
            "Voxa echo cancellation on this mode listens to the whole system output, so it will "
            "remove VoxaTest's spoken prompts. In Voxa, set Preferences > Speech > Echo "
            "cancellation to Off, or to DTLN (which cancels only her own playback), before "
            "running acoustic tests.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    client = settings.make_client()

    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001 - the grading backend is a network boundary
        print(f"Grading model backend is not reachable at {settings.url}: {exc}", file=sys.stderr, flush=True)
        return 2
    if isinstance(models, list) and settings.model not in models:
        print(
            f"Warning: grading model {settings.model} is not listed by {settings.url}; available: {', '.join(models)}",
            flush=True,
        )

    settings_source = settings.voxa_config_path if settings.voxa_config_path else "not found, using defaults"
    print(
        f'Voxa settings: {settings_source} — wake word "{settings.wake_word}", model {settings.model} at {settings.url}',
        flush=True,
    )

    try:
        from voxa.audio import AudioCapture
        from voxa.speech import SpeechService
    except ImportError as exc:
        print(f"Cannot import Voxa audio/speech modules: {exc}", file=sys.stderr, flush=True)
        return 1

    main_loop, main_loop_thread = _start_glib_main_loop()
    speech = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _stop_on_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_on_sigterm)

    reports = Path(reports_dir)
    json_path, text_path = report_paths(
        reports,
        json_path=Path(json_report) if json_report else None,
        text_path=Path(text_report) if text_report else None,
    )
    failure_path = Path(failure_report)

    def log(message: str) -> None:
        print(message, flush=True)

    results: list[RunResult] = []
    interrupted = False
    deadline_hit = False
    aborted = False
    try:
        whisper, whisper_error = _load_whisper(settings, log)
        if whisper_error:
            print(f"Whisper is unavailable: {whisper_error}", file=sys.stderr, flush=True)
            return 1

        capture = AudioCapture()
        if settings.input_device:
            try:
                capture.list_devices()
            except Exception as exc:  # noqa: BLE001 - device discovery is a hardware boundary
                print(f"Microphone device discovery failed: {exc}", file=sys.stderr, flush=True)

        # A robotic offline voice is not a valid end-to-end recognition test. If both natural providers are
        # unavailable, report the setup failure instead of grading Voxa against synthetic speech it cannot hear.
        # A substituted provider can silently change both voice and rate, which
        # invalidates an acoustic comparison. Fail the case instead.
        speech = SpeechService(
            allow_online_fallback=False,
            allow_offline_fallback=False,
            stream_tag=isolation.APPLICATION_NAME,
        )
        started = time.monotonic()
        total = len(cases)
        silent_streak = 0

        for index, case in enumerate(cases, start=1):
            if settings.run_deadline_seconds and time.monotonic() - started > settings.run_deadline_seconds:
                print("Run deadline reached before all cases completed.", flush=True)
                deadline_hit = True
                break

            emit_event("case_start", index=first_index + index - 1, total=first_index + total - 1,
                       id=case.id, category=case.category, prompt=case.prompt)
            result = _run_case_safely(case, settings, client, whisper, capture, speech, log)
            if result.grade.method == "skipped":
                log(f"Skipped {result.case_id} (vm_only, not in test VM).")
                continue
            attempts = 0
            while not result.transcript and result.grade.method == "empty" and attempts < settings.retries:
                attempts += 1
                log(f"Empty transcript for {case.id}; retrying (attempt {attempts + 1} of {settings.retries + 1}).")
                result = _run_case_safely(case, settings, client, whisper, capture, speech, log)

            results.append(result)
            write_reports(
                results,
                reports,
                json_path=json_path,
                text_path=text_path,
                failure_path=failure_path,
                complete=False,
                wake_word=settings.wake_word,
            )
            status = "ERROR" if result.error else ("PASS" if result.grade.passed else "FAIL")
            layer = score_case(result.to_dict(), settings.wake_word)["layer"]
            emit_event("case_result", index=first_index + index - 1, total=first_index + total - 1,
                       id=case.id, category=case.category, prompt=case.prompt, status=status,
                       reply=result.transcript, heard=result.heard, layer=layer,
                       notes=result.grade.notes, seconds=round(result.duration_seconds, 1))
            log(
                f"[{first_index + index - 1}/{first_index + total - 1}] "
                f"{status} {result.case_id} score {result.grade.score} ({result.duration_seconds:.1f}s)"
            )

            if result.grade.method == "empty":
                silent_streak += 1
                if settings.max_consecutive_no_reply > 0 and silent_streak >= settings.max_consecutive_no_reply:
                    print(
                        f"No reply from Voxa for {silent_streak} cases in a row. Is Voxa running with Conversation mode "
                        f"on, listening for the wake word \"{settings.wake_word}\"? Stopping.",
                        flush=True,
                    )
                    aborted = True
                    break
            else:
                silent_streak = 0
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted — writing partial report.", flush=True)
    finally:
        if speech is not None:
            speech.stop()
        main_loop.quit()
        main_loop_thread.join(timeout=2.0)
        signal.signal(signal.SIGTERM, previous_sigterm)

    complete = not interrupted and not deadline_hit and not aborted
    paths = write_reports(
        results,
        reports,
        json_path=json_path,
        text_path=text_path,
        failure_path=failure_path,
        complete=complete,
        wake_word=settings.wake_word,
    )
    print(format_results(results, complete=complete, wake_word=settings.wake_word), flush=True)
    print(f"JSON report: {paths['json_path']}", flush=True)
    print(f"Text report: {paths['text_path']}", flush=True)
    print(f"Failure report: {paths['failure_path']}", flush=True)

    if interrupted:
        return 130
    if aborted:
        return 3
    return 0 if paths["summary"]["failed"] == 0 else 1


def _soak_voices(raw: str) -> list[str]:
    voices = [item.strip() for item in (raw or "").split(",") if item.strip()]
    return voices or [DEFAULT_TTS_VOICE]


def _format_soak_text(result: dict) -> str:
    """The soak's text report: the headline bound, the plan, and every self-trigger named."""
    from . import soak as soak_mod

    lines = ["Voxa soak: her own voice, nobody talking, every self-trigger counted.", ""]
    lines.append(soak_mod.format_bound(result))
    lines.append("")
    plan = result.get("plan", {})
    lines.append(
        f"Replies spoken: {result.get('replies_spoken', 0)} of {plan.get('replies', 0)} planned "
        f"({plan.get('trigger_replies', 0)} carrying trigger words)."
    )
    lines.append(f"Isolation: {isolation.describe()}")
    for check in result.get("checks", []):
        lines.append(f"  {check.get('status', ''):4} {check.get('name', '')}: {check.get('detail', '')}")
    commands = result.get("self_commands", [])
    pauses = result.get("self_pauses", [])
    lines.append("")
    lines.append(f"Self-commands ({len(commands)}):")
    for record in commands[:40]:
        lines.append(f"  {record.get('time', '')} heard={record.get('heard', '')!r} tool={record.get('tool', '')}")
    if not commands:
        lines.append("  none")
    lines.append(f"Self-pauses ({len(pauses)}):")
    for event in pauses[:40]:
        fields = " ".join(f"{key}={value}" for key, value in event.items() if key != "event")
        lines.append(f"  {event.get('event', '')} {fields}")
    if not pauses:
        lines.append("  none")
    return "\n".join(lines)


def _run_soak(settings, args: argparse.Namespace) -> int:
    """``voxatest soak``: the live self-trigger soak.

    It speaks through the speakers, so it needs two things before it makes a sound: ``--yes`` (the run
    is rhY's, who must know the room is about to be used) and a doctor pass with no FAIL. The
    microphone is never opened here: Voxa owns it, and a soak is about what she does with nobody talking.
    """
    from . import soak as soak_mod
    from .doctor import format_checks, run_checks

    try:
        plan = soak_mod.soak_plan(args.hours, seed=args.seed, gap_seconds=args.gap,
                                  voices=_soak_voices(args.voices), wake_word=settings.wake_word)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    summary = soak_mod.plan_summary(plan, args.gap)
    if args.plan:
        print(soak_mod.format_plan(plan, summary), flush=True)
        print("A clean run of this length proves:", flush=True)
        print(soak_mod.format_bound(soak_mod.count_self_events([], [], args.hours, len(plan))), flush=True)
        return 0
    if not args.yes:
        print("A soak speaks through the speakers for hours. Tell rhY first "
              "(docs/notes/voxa-tell-before-live-audio-tests.md), then add --yes.", flush=True)
        print("The plan, with no sound:", flush=True)
        print(soak_mod.format_plan(plan, summary), flush=True)
        return 0

    checks = run_checks(settings, hours=args.hours, seed=args.seed, gap_seconds=args.gap)
    print(format_checks(checks), flush=True)
    if any(check.status == "FAIL" for check in checks):
        print("Not starting: the FAIL lines above say the run would measure nothing.", file=sys.stderr, flush=True)
        return 2

    try:
        from voxa.speech import SpeechService
    except ImportError as exc:  # noqa: BLE001 - the speech stack is a hard requirement of a live soak
        print(f"Cannot import Voxa's speech module: {exc}", file=sys.stderr, flush=True)
        return 1

    main_loop, main_loop_thread = _start_glib_main_loop()
    speech = SpeechService(allow_online_fallback=False, allow_offline_fallback=False,
                           stream_tag=isolation.APPLICATION_NAME)
    errors: list[str] = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _stop_on_sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_on_sigterm)

    def speak(reply) -> None:
        done = threading.Event()
        speech.speak(reply.text, settings.tts_rate, reply.voice,
                     on_started=lambda: None, on_done=done.set,
                     on_error=lambda message: (errors.append(message), done.set()))
        if not done.wait(settings.speak_timeout_seconds):
            errors.append(f"TTS timed out on reply {reply.index}")

    action_log = Path(args.action_log) if args.action_log else Path(settings.action_log_path)
    logs = soak_mod.SoakLogs(action_log=action_log, barge_log=soak_mod.barge_log_path(action_log))
    print(f"Counting self-triggers from {logs.action_log} and {logs.barge_log}", flush=True)

    result: dict[str, Any] = {}
    try:
        result = soak_mod.run_soak(
            plan, logs, hours=args.hours, speak=speak, gap_seconds=args.gap,
            on_event=lambda payload: emit_event("soak", **payload),
        )
    finally:
        speech.stop()
        main_loop.quit()
        main_loop_thread.join(timeout=2.0)
        signal.signal(signal.SIGTERM, previous_sigterm)

    result["isolation"] = isolation.playback_properties()
    result["checks"] = [check.as_dict() for check in checks]
    result["errors"] = errors[-5:]
    numbers = {key: value for key, value in result.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    payload = {"soak": result, "comprehension": numbers}
    json_path = Path(args.json_report) if args.json_report else (
        Path(settings.reports_dir) / f"soak_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = _format_soak_text(result)
    if args.text_report:
        Path(args.text_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.text_report).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"JSON report: {json_path}", flush=True)
    return 130 if result.get("interrupted") else 0


def _command_result_dict(result) -> dict:
    return {
        "id": result.case.id,
        "category": result.case.category,
        "say": result.case.say,
        "expected_tool": result.case.tool,
        "expected_args": result.case.args,
        "got_tool": result.got_tool,
        "got_args": result.got_args,
        "passed": result.passed,
        "problem": result.problem,
        "micros": result.micros,
    }


def _run_command_bench(
    cases_path: Path,
    reports_dir: Path,
    action_log: str | None,
    *,
    repeats: int,
    minimum_checks_per_hour: float,
    capability_report: str | None,
) -> int:
    cases, wishlist = load_command_cases(cases_path)
    if not cases:
        print(f"No command cases found in {cases_path}.", file=sys.stderr, flush=True)
        return 1
    if repeats < 1:
        print("--repeat must be at least 1.", file=sys.stderr, flush=True)
        return 2
    if minimum_checks_per_hour < 0:
        print("--min-checks-per-hour cannot be negative.", file=sys.stderr, flush=True)
        return 2

    from voxa.agent.tools import default_registry

    registry = default_registry()
    started = time.perf_counter()
    results = run_commands(cases * repeats, registry)
    elapsed_seconds = time.perf_counter() - started
    wishlist_results = run_commands(wishlist, registry)
    summary = summarize(
        results,
        elapsed_seconds=elapsed_seconds,
        unique_cases=len(cases),
        repeats=repeats,
        minimum_checks_per_hour=minimum_checks_per_hour,
    )
    markdown = format_markdown(summary, wishlist_results)

    if action_log:
        from voxa.agent.actionlog import ActionLog
        from voxa.agent.actionlog import summarize as summarize_actions

        stats = summarize_actions(ActionLog(Path(action_log)).read())
        lines = [markdown.rstrip(), "", "## Heard in real use but not routed"]
        if stats["unrouted"]:
            lines.append("| heard | count |")
            lines.append("| --- | ---: |")
            for entry in stats["unrouted"]:
                lines.append(f"| {entry['heard']} | {entry['count']} |")
        else:
            lines.append("None.")
        lines.append("")
        lines.append("## Tool failures in real use")
        if stats["failures"]:
            lines.append("| tool | heard | detail |")
            lines.append("| --- | --- | --- |")
            for failure in stats["failures"]:
                lines.append(
                    f"| {failure.get('tool', '')} | {failure.get('heard', '')} | {failure.get('detail', '')} |"
                )
        else:
            lines.append("None.")
        markdown = "\n".join(lines) + "\n"

    print(markdown, flush=True)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "command-bench.md").write_text(markdown, encoding="utf-8")
    payload = {
        "summary": summary,
        "results": [_command_result_dict(result) for result in results],
        "wishlist": [_command_result_dict(result) for result in wishlist_results],
    }
    (reports_dir / "command-bench.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if capability_report:
        capability_path = Path(capability_report)
        capability_path.parent.mkdir(parents=True, exist_ok=True)
        capability_path.write_text(markdown, encoding="utf-8")

    if summary["failed"]:
        return 1
    return 0 if summary["throughput_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings()
    _apply_cli_overrides(settings, args)

    if args.command == "generate":
        client = settings.make_client()
        cases = generate_cases(client, settings.model, args.count, batch_size=args.batch_size)
        save_cases(Path(args.output), cases)
        print(f"Generated {len(cases)} test cases into {args.output}")
        return 0

    if args.command == "run":
        if getattr(args, "offline_audio", False):
            return _run_offline(settings, args)
        cases = load_cases(Path(args.cases))
        start_at = max(1, args.start_at)
        if start_at > 1:
            cases = cases[start_at - 1 :]
        if args.limit:
            cases = cases[: args.limit]
        if not cases:
            print(f"No test cases found in {args.cases}. Generate cases first.", file=sys.stderr)
            return 1
        return _run(
            settings,
            cases,
            args.json_report,
            args.text_report,
            args.reports or str(settings.reports_dir),
            args.failure_report or str(settings.failure_report_path),
            first_index=start_at,
        )

    if args.command == "commands":
        return _run_command_bench(
            Path(args.cases),
            Path(args.report_dir),
            args.action_log,
            repeats=args.repeat,
            minimum_checks_per_hour=args.min_checks_per_hour,
            capability_report=args.capability_report,
        )

    if args.command == "capabilities":
        from .capabilities import write_status

        info = write_status(
            Path(args.reports),
            Path(args.markdown) if args.markdown else None,
            Path(args.html) if args.html else None,
            since=args.since,
            stamp=args.stamp,
        )
        print(json.dumps(info["counts"]), f"{info['passed']}/{info['runs']} spoken checks passed", flush=True)
        return 0

    if args.command == "latency":
        report = latency_report(load_records(Path(args.action_log), args.last))
        print(format_latency(report))
        return 0

    if args.command == "comprehension":
        argv = ["--report", args.report]
        if args.out:
            argv += ["--out", args.out]
        if args.wake_word is not None:
            argv += ["--wake-word", args.wake_word]
        if args.top is not None:
            argv += ["--top", str(args.top)]
        if args.json:
            argv.append("--json")
        return comprehension_main(argv)

    if args.command == "check":
        return check_comprehension([args.report, *args.conditions])

    if args.command == "soak":
        return _run_soak(settings, args)

    if args.command == "doctor":
        from .doctor import doctor_main

        argv = ["--hours", str(args.hours), "--seed", str(args.seed), "--gap", str(args.gap)]
        if args.dry_run:
            argv.append("--dry-run")
        if args.json:
            argv.append("--json")
        return doctor_main(argv)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
