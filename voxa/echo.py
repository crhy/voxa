"""Echo cancellation: let Voxa hear the user, not the computer.

Whatever the computer plays (Voxa's own voice, music, a video) reaches the microphone and used to be heard
as speech: Voxa interrupted itself and transcribed songs. PulseAudio's ``module-echo-cancel`` (WebRTC) solves
this at the source: it is given the speaker signal and subtracts it from the microphone signal.

Two virtual devices are created:

* an echo-cancelled *microphone* (:data:`SOURCE_NAME`) that Voxa records from, and
* an *output* (:data:`SINK_NAME`) that forwards to the real speakers. Only sound played through this output
  can be subtracted, so it is made the default output while Voxa runs and everything already playing is moved
  onto it. On exit the previous default is restored and the module is unloaded.

All commands go through ``pactl`` on the host, so this works from inside the Flatpak too. Every step is
best-effort: if anything is missing (no PulseAudio, no module) Voxa simply records from the normal microphone.
"""

from __future__ import annotations

import logging
import re
import subprocess

from .agent.host import host_command

log = logging.getLogger("voxa.echo")

SOURCE_NAME = "voxa_echo_cancel_mic"
SINK_NAME = "voxa_echo_cancel_out"
# WebRTC settings: leave the hardware gain alone and clean up noise. (PulseAudio 17 rejects the module when
# given options it does not know, so only long-standing ones are used.)
AEC_ARGS = "analog_gain_control=0 digital_gain_control=1 noise_suppression=1"


def _pactl(*args: str, runner=subprocess.run) -> tuple[int, str]:
    try:
        proc = runner(host_command(["pactl", *args]), capture_output=True, text=True, check=False, timeout=6)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def parse_module_id(listing: str, source_name: str = SOURCE_NAME) -> str | None:
    """The id of an already loaded echo-cancel module that provides our microphone, from ``list short modules``."""
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "module-echo-cancel" and f"source_name={source_name}" in line:
            return parts[0]
    return None


def parse_sink_inputs(listing: str) -> list[str]:
    """Ids of the streams currently playing, from ``pactl list short sink-inputs``."""
    return [line.split("\t")[0] for line in listing.splitlines() if line.strip() and line.split("\t")[0].isdigit()]


def default_monitor_source(runner=subprocess.run) -> str | None:
    """The loopback monitor source for the default sink, e.g. ``"alsa_output.pci-0_0.monitor"``.

    This is the signal the computer plays, which the canceller in :mod:`voxa.anc` subtracts from the
    microphone. Returns ``None`` when PulseAudio cannot report a default sink.
    """
    code, sink = _pactl("get-default-sink", runner=runner)
    if code != 0 or not sink:
        return None
    return f"{sink}.monitor"


def source_for(microphone_id: str, runner=subprocess.run) -> str | None:
    """The PulseAudio source name of the microphone chosen in Voxa (matched by its serial), or None."""
    wanted = re.sub(r"[^a-z0-9]+", "_", (microphone_id or "").split(":", 1)[-1].casefold()).strip("_")
    if not wanted:
        return None
    code, listing = _pactl("list", "short", "sources", runner=runner)
    if code != 0:
        return None
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[1].endswith(".monitor") or parts[1] == SOURCE_NAME:
            continue
        if wanted in re.sub(r"[^a-z0-9]+", "_", parts[1].casefold()):
            return parts[1]
    return None


class EchoCanceller:
    """Loads, uses and removes the echo-cancelling devices. Safe to call when PulseAudio is absent."""

    def __init__(self, runner=subprocess.run) -> None:
        self._runner = runner
        self._module_id: str | None = None
        self._previous_sink: str | None = None
        self._loaded_here = False
        self.active = False

    def _run(self, *args: str) -> tuple[int, str]:
        return _pactl(*args, runner=self._runner)

    def enable(self, source_master: str | None = None) -> bool:
        """Create the devices (or adopt existing ones) and route playback through them. True when active."""
        if self.active:
            return True
        code, modules = self._run("list", "short", "modules")
        if code != 0:
            log.info("echo cancellation unavailable: pactl failed (%s)", modules)
            return False
        code, default_sink = self._run("get-default-sink")
        if code != 0 or not default_sink:
            return False
        existing = parse_module_id(modules)
        if existing is not None:
            # Left over from a run that did not exit cleanly. It may be tied to a microphone or speakers that
            # are no longer the right ones, so remove it and build a fresh one.
            self._run("unload-module", existing)
            code, default_sink = self._run("get-default-sink")
            if code != 0 or not default_sink:
                return False
        if default_sink == SINK_NAME:
            return False  # inconsistent state: do not stack another module on top
        arguments = [
            "load-module",
            "module-echo-cancel",
            "aec_method=webrtc",
            f"source_name={SOURCE_NAME}",
            f"sink_name={SINK_NAME}",
            f"sink_master={default_sink}",
            f'aec_args="{AEC_ARGS}"',  # the quotes are part of the value: it contains spaces
            "source_properties=device.description=Voxa-echo-cancelled-microphone",
            "sink_properties=device.description=Voxa-echo-cancelled-output",
        ]
        if source_master:
            arguments.append(f"source_master={source_master}")  # the microphone chosen in Voxa, not the system default
        code, module_id = self._run(*arguments)
        if code != 0 or not module_id.isdigit():
            log.info("echo cancellation unavailable: module-echo-cancel did not load (%s)", module_id)
            return False
        self._module_id = module_id
        self._loaded_here = True
        self._previous_sink = default_sink
        # Everything must play THROUGH the cancelling output, or it cannot be subtracted from the microphone.
        self._run("set-default-sink", SINK_NAME)
        _code, inputs = self._run("list", "short", "sink-inputs")
        for stream in parse_sink_inputs(inputs):
            self._run("move-sink-input", stream, SINK_NAME)
        self.active = True
        log.info("echo cancellation active (module %s, speakers %s)", self._module_id, self._previous_sink)
        return True

    def disable(self) -> None:
        """Put the sound routing back exactly as it was and remove the devices."""
        if self._module_id is None:
            return
        if self._previous_sink:
            self._run("set-default-sink", self._previous_sink)
            _code, inputs = self._run("list", "short", "sink-inputs")
            for stream in parse_sink_inputs(inputs):
                self._run("move-sink-input", stream, self._previous_sink)
        if self._loaded_here:
            self._run("unload-module", self._module_id)
        self._module_id = None
        self._loaded_here = False
        self.active = False
