from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from .cases import load_cases, save_cases
from .config import DEFAULT_DATA_PATH, DEFAULT_REPORTS_DIR, load_settings
from .generate_cases import generate_cases
from .report import format_results, write_reports
from .runner import run_case


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
    run.add_argument("--reports", default=str(DEFAULT_REPORTS_DIR))
    run.add_argument("--json-report")
    run.add_argument("--text-report")
    run.add_argument("--input-device", default=None)
    run.add_argument("--output-device", default=None)
    run.add_argument("--wake-word", default=None)

    return parser.parse_args(argv)


def _apply_cli_overrides(settings, args: argparse.Namespace) -> None:
    input_device = getattr(args, "input_device", None)
    output_device = getattr(args, "output_device", None)
    wake_word = getattr(args, "wake_word", None)

    if input_device is not None:
        settings.input_device = input_device
    if output_device is not None:
        settings.output_device = output_device
    if wake_word is not None:
        settings.wake_word = wake_word


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


def _run(settings, cases: list, json_report: str | None, text_report: str | None, reports_dir: str) -> int:
    if settings.output_device:
        os.environ.setdefault("GST_AUDIOSINK", settings.output_device)

    client = settings.make_client()

    try:
        from voxa.audio import AudioCapture
        from voxa.speech import SpeechService
    except ImportError as exc:
        print(f"Cannot import Voxa audio/speech modules: {exc}", file=sys.stderr)
        return 1

    main_loop, main_loop_thread = _start_glib_main_loop()
    speech = None
    try:
        whisper, whisper_error = _load_whisper(settings, print)
        if whisper_error:
            print(f"Whisper is unavailable: {whisper_error}", file=sys.stderr)
            return 1

        capture = AudioCapture()
        if settings.input_device:
            try:
                capture.list_devices()
            except Exception as exc:  # noqa: BLE001 - device discovery is a hardware boundary
                print(f"Microphone device discovery failed: {exc}", file=sys.stderr)

        speech = SpeechService()
        results = []
        started = time.monotonic()

        for case in cases:
            if settings.run_deadline_seconds and time.monotonic() - started > settings.run_deadline_seconds:
                print("Run deadline reached before all cases completed.")
                break
            results.append(run_case(case, settings, client, whisper, capture, speech, log=print))

        paths = write_reports(
            results,
            Path(reports_dir),
            json_path=Path(json_report) if json_report else None,
            text_path=Path(text_report) if text_report else None,
        )
        print(format_results(results))
        print(f"JSON report: {paths['json_path']}")
        print(f"Text report: {paths['text_path']}")

        return 0 if paths["summary"]["failed"] == 0 else 1
    finally:
        if speech is not None:
            speech.stop()
        main_loop.quit()
        main_loop_thread.join(timeout=2.0)


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
        if args.limit:
            cases = cases[: args.limit]
        if not cases:
            print(f"No test cases found in {args.cases}. Generate cases first.", file=sys.stderr)
            return 1
        return _run(settings, cases, args.json_report, args.text_report, args.reports)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
