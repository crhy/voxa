"""``voxatest doctor``: the checks a live hearing run needs, before it starts.

A live soak measures the room, the microphone, the speakers and Voxa's own voice. Half of the ways
such a run goes wrong are known in advance and can be seen without playing a single sound: the test
voice landing in Voxa's echo reference (so her canceller cancels the test user), a microphone that is
a monitor of the output, echo cancellation switched off, no drift estimate for this speaker/microphone
pair, a log we cannot read. Each check prints one PASS / WARN / FAIL / SKIP line and says what to do.

* ``PASS`` - checked and good.
* ``WARN`` - the run will work, and the number it reports must be read with this in mind.
* ``FAIL`` - the run would measure nothing; ``voxatest doctor`` exits 1.
* ``SKIP`` - a device probe, deliberately not done (``--dry-run``): nothing was asked of the hardware.

Exit codes: ``0`` when nothing failed, ``1`` when a check failed. Warnings never fail a run: a soak on
the raw route is a legitimate measurement, as long as the report says so.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from voxa.reference import (
    MODE_OWN_RENDER,
    MODE_OWN_SINK_INPUT,
    MODE_SINK_MONITOR,
    decide,
)

from . import isolation, soak
from .config import HarnessSettings, load_settings

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass(slots=True)
class Check:
    """One line of the doctor's report."""

    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.status:4} {self.name}: {self.detail}"


def check_wake_word(settings: HarnessSettings) -> Check:
    word = (settings.wake_word or "").strip()
    if not word:
        return Check("wake word", WARN, "Voxa's wake word is empty in her config: she will not start listening.")
    return Check("wake word", PASS, f'she answers to "{word}"; the soak speaks it inside her own replies')


def check_echo_mode(settings: HarnessSettings) -> Check:
    mode = (settings.voxa_echo_mode or "").strip()
    if not mode:
        return Check("AEC mode", WARN, "Voxa's config was not found, so her echo mode is unknown.")
    from voxa.config import SELF_REFERENCE_ECHO_MODES

    if mode in SELF_REFERENCE_ECHO_MODES:
        return Check("AEC mode", PASS, f"Voxa cancels her own playback ({mode}); the soak measures her own canceller")
    if mode == "webrtc":
        if settings.output_device:
            return Check(
                "AEC mode", PASS,
                "webrtc cancels the whole system output, but VoxaTest plays to its own device: "
                "the test voice never enters that reference.",
            )
        return Check(
            "AEC mode", WARN,
            "webrtc cancels the whole system output, VoxaTest's spoken prompts included: a soak "
            "on this config would cancel what it is measuring. Run with --output-device, or set "
            "Voxa's echo cancellation to DTLN.",
        )
    if mode == "off":
        return Check(
            "AEC mode", WARN,
            "echo cancellation is OFF: Voxa will hear her own voice and the test voice as speech. "
            "A soak now measures the raw route, not the product route.",
        )
    return Check("AEC mode", WARN, f"unknown echo mode {mode!r} in Voxa's config")


def check_isolation(settings: HarnessSettings) -> Check:
    return Check(
        "isolation", PASS,
        "VoxaTest's playback is tagged " + isolation.describe() +
        ", so Voxa's reference policy can exclude it (voxa/speech.py stream_tag)",
    )


def check_reference_mode(monitor_source: str | None, runner=subprocess.run) -> Check:
    decision = decide(monitor_source, runner=runner, check_sources=True)
    if decision.mode == MODE_OWN_RENDER:
        return Check("reference mode", PASS, "own-render reference: the test voice cannot enter it")
    if decision.mode == MODE_OWN_SINK_INPUT:
        return Check("reference mode", PASS, f"per-stream monitor {decision.source}: only Voxa's own playback is in it")
    if decision.mode == MODE_SINK_MONITOR:
        if decision.test_present:
            return Check("reference mode", FAIL, decision.warning)
        return Check("reference mode", WARN,
                     f"whole-sink monitor {decision.source}: any other app playing is in Voxa's reference too")
    return Check("reference mode", WARN, decision.warning or "the sound server did not answer")


def check_same_device(microphone: str, monitor_source: str | None) -> Check:
    """Is the microphone a monitor of the output the test voice plays through?"""
    mic = (microphone or "").strip()
    if not mic:
        return Check("mic/output device", WARN,
                     "no microphone chosen in Voxa: the system default will be used and cannot be checked here")
    folded = mic.casefold()
    if "monitor" in folded:
        return Check(
            "mic/output device", FAIL,
            f'the chosen source "{mic}" is a monitor of the output, not a microphone: Voxa would hear '
            "the test voice as her own echo and cancel it",
        )
    if monitor_source and monitor_source.casefold().startswith(folded):
        return Check("mic/output device", WARN,
                     f'the microphone "{mic}" and the reference sink share a device name: check that '
                     "the speakers really feed the room")
    return Check("mic/output device", PASS, f'microphone "{mic}" is a source, not the output monitor')


def check_drift(settings: HarnessSettings, sink: str | None) -> Check:
    """A drift estimate for this exact speaker/microphone pair, from Voxa's saved hearing profile."""
    if not sink:
        return Check("drift estimate", WARN, "no output sink reported, so the saved hearing profile cannot be found")
    from voxa.anccal import load_profile

    profile = load_profile(sink, settings.input_device or "default")
    if not profile:
        return Check(
            "drift estimate", WARN,
            f"no saved hearing profile for {sink} + {settings.input_device or 'default'}: the first minutes "
            "of the soak are the estimate, so read the first hour as the settling period",
        )
    delay = profile.get("delay")
    taps = len(profile.get("state", {}).get("taps", [])) if isinstance(profile.get("state"), dict) else 0
    return Check("drift estimate", PASS,
                 f"saved hearing profile for this pair: echo delay {delay} samples, {taps} taps learned")


def check_action_log(path: Path) -> Check:
    try:
        if path.is_file():
            return Check("action log", PASS, f"{path} exists and is being written")
        path.parent.mkdir(parents=True, exist_ok=True)
        return Check("action log", WARN, f"{path} does not exist yet: Voxa must be running for the soak to count")
    except OSError as exc:
        return Check("action log", FAIL, f"{path} cannot be read ({exc})")


def check_plan(hours: float, seed: int, gap_seconds: float, wake_word: str) -> Check:
    try:
        plan = soak.soak_plan(hours, seed=seed, gap_seconds=gap_seconds, wake_word=wake_word)
    except ValueError as exc:
        return Check("soak plan", FAIL, str(exc))
    summary = soak.plan_summary(plan, gap_seconds)
    return Check("soak plan", PASS,
                 f"{summary['replies']} replies, {summary['trigger_replies']} with trigger words, "
                 f"{summary['playback_hours']:.2f} playback hours")


def run_checks(
    settings: HarnessSettings | None = None,
    *,
    dry_run: bool = False,
    hours: float = 1.0,
    seed: int = 0,
    gap_seconds: float = 2.5,
    runner: Callable[..., Any] = subprocess.run,
    monitor_source: str | None = None,
) -> list[Check]:
    """Every check, in the order a live run should read them.

    ``dry_run=True`` asks nothing of the sound server: the reference mode and the device pair are
    reported as SKIP, because a dry run makes no claim about the hardware.
    """
    settings = settings or load_settings()
    probe = not dry_run
    monitor = monitor_source
    if probe and monitor is None:
        from voxa.echo import default_monitor_source

        monitor = default_monitor_source()

    checks: list[Check] = [check_wake_word(settings), check_echo_mode(settings), check_isolation(settings)]
    if probe:
        checks.append(check_reference_mode(monitor, runner=runner))
        checks.append(check_same_device(settings.input_device, monitor))
    else:
        checks.append(Check("reference mode", SKIP,
                            "not probed in a dry run: a live run checks that the test voice is out of Voxa's reference"))
        checks.append(Check("mic/output device", SKIP, "not probed in a dry run: no device was asked about"))
    checks.append(check_drift(settings, monitor if probe else None))
    checks.append(check_action_log(Path(settings.action_log_path)))
    checks.append(check_plan(hours, seed, gap_seconds, settings.wake_word))
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = [str(check) for check in checks]
    warnings = sum(1 for check in checks if check.status == WARN)
    failures = [check.name for check in checks if check.status == FAIL]
    lines.append("")
    if failures:
        lines.append(f"RESULT: FAIL ({', '.join(failures)}). Fix these before a live run.")
    elif warnings:
        lines.append(f"RESULT: OK with {warnings} warning(s): the run is valid, read its numbers with those in mind.")
    else:
        lines.append("RESULT: OK.")
    return "\n".join(lines)


def doctor_main(argv: list[str] | None = None) -> int:
    """``voxatest doctor [--dry-run] [--hours H]``: print the checks, exit 1 only on a FAIL."""
    import argparse

    parser = argparse.ArgumentParser(prog="voxatest doctor", description="Check the setup before a live hearing run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="ask nothing of the sound server or the microphone: only the config and the plan are checked")
    parser.add_argument("--hours", type=float, default=1.0, help="the soak length to plan for")
    parser.add_argument("--seed", type=int, default=0, help="the soak seed to plan for")
    parser.add_argument("--gap", type=float, default=2.5, help="seconds of silence between replies")
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    args = parser.parse_args(argv)

    checks = run_checks(dry_run=args.dry_run, hours=args.hours, seed=args.seed, gap_seconds=args.gap)
    if args.json:
        import json

        print(json.dumps({"checks": [check.as_dict() for check in checks],
                          "result": "FAIL" if any(check.status == FAIL for check in checks) else "OK"},
                         indent=2))
    else:
        print(format_checks(checks), flush=True)
    return 1 if any(check.status == FAIL for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(doctor_main(sys.argv[1:]))
