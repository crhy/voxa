"""Echo cancellation: let Voxa hear the user, not the computer.

Whatever the computer plays (Voxa's own voice, music, a video) reaches the microphone and used to be heard
as speech: Voxa interrupted itself and transcribed songs. Voxa's own canceller (:mod:`voxa.anc`) solves this
by LISTENING to the loopback monitor of the real default output (:func:`default_monitor_source`) and
subtracting that signal from the microphone. It creates no devices and never changes the user's audio output.

Versions up to 0.1.5 could leave behind PulseAudio echo-cancel devices that hijacked the user's speakers.
:func:`remove_leftover_devices` finds such a leftover, hands the real output back (restoring the default
sink and moving any streams that were on our device) and unloads the module.
"""

from __future__ import annotations

import logging
import re
import subprocess

from .agent.host import host_command

log = logging.getLogger("voxa.echo")

SOURCE_NAME = "voxa_echo_cancel_mic"
SINK_NAME = "voxa_echo_cancel_out"


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


def parse_sink_master(listing: str, source_name: str = SOURCE_NAME) -> str | None:
    """The real output a leftover echo-cancel module of ours forwards to (its ``sink_master=`` argument)."""
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "module-echo-cancel" and f"source_name={source_name}" in line:
            match = re.search(r"sink_master=(\S+)", line)
            return match.group(1) if match else None
    return None


def default_monitor_source(runner=subprocess.run) -> str | None:
    """The loopback monitor source for the default sink, e.g. ``"alsa_output.pci-0_0.monitor"``.

    This is the signal the computer plays, which the canceller in :mod:`voxa.anc` subtracts from the
    microphone. Returns ``None`` when PulseAudio cannot report a default sink, and when the default is
    our own old device (never use it as the reference).
    """
    code, sink = _pactl("get-default-sink", runner=runner)
    if code != 0 or not sink or sink == SINK_NAME:
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


def remove_leftover_devices(runner=subprocess.run) -> bool:
    """Remove the echo-cancel devices an older Voxa created and hand the output back. True if one was removed."""
    code, modules = _pactl("list", "short", "modules", runner=runner)
    if code != 0:
        return False
    module_id = parse_module_id(modules)
    if module_id is None:
        return False
    master = parse_sink_master(modules)
    code, default_sink = _pactl("get-default-sink", runner=runner)
    if code == 0 and default_sink == SINK_NAME and master:
        _pactl("set-default-sink", master, runner=runner)
    if master:
        _code, sinks = _pactl("list", "short", "sinks", runner=runner)
        our_index = None
        for line in sinks.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == SINK_NAME:
                our_index = parts[0]
                break
        if our_index is not None:
            _code, inputs = _pactl("list", "short", "sink-inputs", runner=runner)
            for line in inputs.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1] == our_index:
                    _pactl("move-sink-input", parts[0], master, runner=runner)
    _pactl("unload-module", module_id, runner=runner)
    return True
