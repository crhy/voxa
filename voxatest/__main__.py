from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .cases import load_cases, save_cases
from .config import DEFAULT_DATA_PATH, DEFAULT_FAILURE_REPORT_PATH, DEFAULT_REPORTS_DIR, load_settings
from .generate_cases import generate_cases
from .grader import GradeResult
from .report import format_results, report_paths, write_reports
from .runner import RunResult, run_case


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voxatest", description="Speak-test Voxa end to end.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate spoken test cases with the local model.")
    generate.add_argument("--count", type=int, default=250)
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
    run.add_argument("--start-at", type=int, default=1, help="1-based index of the first case to run.")
    run.add_argument(
        "--max-silent",
        type=int,
        help="Stop the run after this many consecutive cases with no reply (0 disables).",
    )

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

        speech = SpeechService()
        started = time.monotonic()
        total = len(cases)
        silent_streak = 0

        for index, case in enumerate(cases, start=1):
            if settings.run_deadline_seconds and time.monotonic() - started > settings.run_deadline_seconds:
                print("Run deadline reached before all cases completed.", flush=True)
                deadline_hit = True
                break

            result = _run_case_safely(case, settings, client, whisper, capture, speech, log)
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
            )
            status = "ERROR" if result.error else ("PASS" if result.grade.passed else "FAIL")
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
    )
    print(format_results(results, complete=complete), flush=True)
    print(f"JSON report: {paths['json_path']}", flush=True)
    print(f"Text report: {paths['text_path']}", flush=True)
    print(f"Failure report: {paths['failure_path']}", flush=True)

    if interrupted:
        return 130
    if aborted:
        return 3
    return 0 if paths["summary"]["failed"] == 0 else 1


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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
