"""Which stream Voxa takes as her echo reference, and keeping the test voice out of it.

The canceller in :mod:`voxa.anc` subtracts the reference from the microphone. When the reference is
the loopback monitor of the whole output sink it holds EVERY stream playing on that sink: Voxa's own
voice, which must be cancelled, and - during a VoxaTest run - the test voice that stands in for the
user, which must never be cancelled (handoff 8.2: "keep test-fixture playback out of Voxa's
echo-reference cancellation policy when it represents the human"). Cancelling the test voice makes a
hearing measurement measure nothing.

This module picks the narrowest reference the machine offers and says plainly, in a WARNING, when the
test voice cannot be kept out of it. Preference order:

1. ``own-render`` - the reference is the samples Voxa rendered herself (roadmap task 1). Nothing that
   reaches the speakers can enter it, so the test voice is excluded by construction.
2. ``own-sink-input`` - the per-stream monitor of Voxa's own playback stream
   (``<sink>.input<index>.monitor``): only her own playback is in it, every foreign stream is skipped.
3. ``sink-monitor`` - the monitor of the whole sink. It contains the test voice, so the decision is
   reported untrusted and a WARNING is logged: the harness's numbers are invalid while that is true.

Pure Python: the pactl conversation is injected, so every branch is testable with no sound server.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger("voxa.reference")

#: application.name / media.role VoxaTest puts on its playback stream (see voxatest/isolation.py).
TEST_STREAM_TAG = "voxatest"
#: application.name PulseAudio reports for Voxa's own playback.
OWN_APPLICATION_NAME = "Voxa"
#: What the tagged stream is called in the sound server's own listing.
TEST_MEDIA_NAME = "VoxaTest test voice"
TEST_STREAM_DESCRIPTION = "VoxaTest speaks the test user through the speakers"

MODE_OWN_RENDER = "own-render"
MODE_OWN_SINK_INPUT = "own-sink-input"
MODE_SINK_MONITOR = "sink-monitor"
MODE_UNKNOWN = "unknown"

_SINK_INPUT_HEADER = re.compile(r"^Sink Input #(\d+)\s*$")
_KEY_VALUE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*=\s*(.+?)\s*$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def parse_sink_inputs(listing: str) -> list[dict]:
    """VoxaTest-relevant fields of every sink input, from ``pactl list sink-inputs``.

    Each row is ``{"index": int, "sink_index": str, "application_name": str, "media_role": str,
    "media_name": str}``. Unreadable lines are skipped; a listing with no sink inputs gives [].
    """
    rows: list[dict] = []
    current: dict | None = None
    for line in listing.splitlines():
        header = _SINK_INPUT_HEADER.match(line.strip())
        if header:
            current = {"index": int(header.group(1)), "sink_index": "", "application_name": "",
                       "media_role": "", "media_name": ""}
            rows.append(current)
            continue
        if current is None:
            continue
        match = _KEY_VALUE.match(line)
        if match is None:
            continue
        key, value = match.group(1), _unquote(match.group(2))
        if key == "Sink":
            current["sink_index"] = value
        elif key in ("application.name", "media.role", "media.name", "application.process.binary"):
            current[key.replace(".", "_")] = value
    return rows


def parse_short_sinks(listing: str) -> dict[str, str]:
    """``{index: sink name}`` from ``pactl list short sinks``."""
    out: dict[str, str] = {}
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            out[parts[0]] = parts[1]
    return out


def per_stream_monitor(sink: str, index: int) -> str:
    """The PulseAudio source carrying ONLY the playback of sink input ``index`` on ``sink``."""
    return f"{sink}.input{index}.monitor"


def test_inputs(rows: list[dict], tag: str = TEST_STREAM_TAG) -> list[dict]:
    """The streams tagged as the test voice (VoxaTest's playback)."""
    wanted = (tag or "").casefold()
    return [row for row in rows
            if wanted and (row.get("application_name", "").casefold() == wanted
                           or row.get("media_role", "").casefold() == wanted)]


def own_inputs(rows: list[dict], application: str = OWN_APPLICATION_NAME) -> list[dict]:
    """The streams that are Voxa's own playback."""
    wanted = (application or "").casefold()
    return [row for row in rows
            if wanted and row.get("application_name", "").casefold() == wanted]


@dataclass(slots=True)
class ReferenceDecision:
    """What Voxa should record as her reference, and whether the test voice is inside it."""

    source: str | None
    mode: str
    trusted: bool
    warning: str = ""
    own_input: dict | None = None
    test_present: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "mode": self.mode,
            "trusted": self.trusted,
            "warning": self.warning,
            "test_present": self.test_present,
            "own_input": self.own_input.get("index") if self.own_input else None,
            "notes": list(self.notes),
        }


TEST_IN_REFERENCE_WARNING = (
    "VoxaTest's playback is on the same output as Voxa's reference monitor and cannot be excluded: "
    "her canceller will cancel the test voice as if it were her own echo. The hearing numbers of a "
    "run taken now are not valid. Start VoxaTest's audio on its own output, or run with the "
    "own-render reference."
)


def choose_reference_source(
    sink_monitor: str | None,
    rows: list[dict],
    sources: set[str] | None = None,
    own_render: bool = False,
    sink_names: dict[str, str] | None = None,
) -> ReferenceDecision:
    """The narrowest reference available, judged from the sink inputs that exist right now.

    ``sink_monitor`` is the whole-sink loopback source (``voxa.echo.default_monitor_source()``) or
    None; ``rows`` are parsed sink inputs; ``sources`` are the source names the sound server reports
    (when given, a per-stream monitor is only chosen if it really exists); ``sink_names`` maps a sink
    input's numeric ``Sink:`` index to the sink's name.
    """
    if own_render:
        return ReferenceDecision(source=None, mode=MODE_OWN_RENDER, trusted=True,
                                 notes=["the reference is Voxa's own rendered samples"])
    if sink_monitor is None:
        return ReferenceDecision(source=None, mode=MODE_UNKNOWN, trusted=False,
                                 warning="No output monitor is available, so Voxa has no echo reference.",
                                 notes=["no default sink"])
    sink = sink_monitor[: -len(".monitor")] if sink_monitor.endswith(".monitor") else sink_monitor
    names = sink_names or {}
    mine = [row for row in own_inputs(rows) if _same_sink(row, sink, names)]
    test = [row for row in test_inputs(rows) if _same_sink(row, sink, names)]
    if mine:
        candidate = per_stream_monitor(sink, int(mine[0]["index"]))
        if sources is None or candidate in sources:
            return ReferenceDecision(source=candidate, mode=MODE_OWN_SINK_INPUT, trusted=True,
                                     own_input=mine[0], test_present=bool(test),
                                     notes=[f"only Voxa's own sink input #{mine[0]['index']} is in the reference"])
        decision = ReferenceDecision(
            source=sink_monitor, mode=MODE_SINK_MONITOR, trusted=not test,
            own_input=mine[0], test_present=bool(test),
            notes=[f"per-stream monitor {candidate} is not offered by the sound server"])
        if test:
            decision.warning = TEST_IN_REFERENCE_WARNING
            decision.notes.append("falling back to the whole-sink monitor")
        return decision
    decision = ReferenceDecision(source=sink_monitor, mode=MODE_SINK_MONITOR, trusted=not test,
                                 test_present=bool(test),
                                 notes=["Voxa's own playback stream is not listed; the whole-sink monitor is used"])
    if test:
        decision.warning = TEST_IN_REFERENCE_WARNING
    return decision


def _same_sink(row: dict, sink: str, sink_names: dict[str, str]) -> bool:
    """Is this sink input playing on ``sink``? ``sink`` may be matched by name or by numeric index."""
    index = str(row.get("sink_index", ""))
    if not index:
        return True  # the listing did not say: assume it is the output we are talking about
    if index == sink or sink_names.get(index) == sink:
        return True
    if not sink_names:
        # Indices cannot be resolved to names on this machine: assume the one output.
        return True
    return False


def sink_input_rows(runner=subprocess.run) -> list[dict]:
    """The sink inputs as the sound server reports them now ([] when pactl cannot answer)."""
    try:
        proc = runner(["pactl", "list", "sink-inputs"], capture_output=True, text=True, check=False, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return parse_sink_inputs(proc.stdout or "")


def source_names(runner=subprocess.run) -> set[str]:
    """Every source the sound server offers ([] when pactl cannot answer)."""
    try:
        proc = runner(["pactl", "list", "short", "sources"], capture_output=True, text=True, check=False, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def decide(
    sink_monitor: str | None,
    runner=subprocess.run,
    own_render: bool = False,
    check_sources: bool = True,
    resolve_sinks: bool = True,
) -> ReferenceDecision:
    """Ask the sound server and decide. Never raises: no sound server means an untrusted unknown.

    ``check_sources`` and ``resolve_sinks`` add two pactl queries each; a caller on a thread that must
    not block for long (Voxa's frame clock) turns them off and accepts the best-effort answer.
    """
    rows = sink_input_rows(runner)
    sources = source_names(runner) if check_sources and rows else None
    names = sink_names(runner) if resolve_sinks and rows else {}
    return choose_reference_source(sink_monitor, rows, sources, own_render=own_render, sink_names=names)


def sink_names(runner=subprocess.run) -> dict[str, str]:
    """``{index: sink name}`` from the sound server ({} when pactl cannot answer)."""
    try:
        proc = runner(["pactl", "list", "short", "sinks"], capture_output=True, text=True, check=False, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    return parse_short_sinks(proc.stdout or "")


def stream_properties() -> dict[str, str]:
    """The stream properties that mark playback as the test voice, never as Voxa."""
    return {
        "application.name": TEST_STREAM_TAG,
        "media.role": TEST_STREAM_TAG,
        "media.name": TEST_MEDIA_NAME,
    }


def _set(element, key: str, value) -> bool:
    try:
        if getattr(element, "find_property", None) is not None and element.find_property(key) is None:
            return False
        element.set_property(key, value)
        return True
    except Exception:  # noqa: BLE001 - a sink that refuses a property must never break playback
        return False


def apply_stream_tag(element, gst=None) -> dict[str, bool]:
    """Tag a playback element as the test voice: name, description and the stream properties.

    ``element`` is a playbin (the properties travel in its ``property-structure``, which GStreamer
    hands to the PulseAudio/PipeWire sink), an audio sink element (``properties``, which pulsesink and
    pipewiresink pass straight to the sound server), or an ``autoaudiosink`` that forwards them to its
    child. Returns what reached the element; nothing here can stop the audio.
    """
    applied: dict[str, bool] = {}
    applied["name"] = _set(element, "name", TEST_MEDIA_NAME)
    applied["description"] = _set(element, "description", TEST_STREAM_DESCRIPTION)
    structure = None
    if gst is not None:
        try:
            structure = gst.Structure.new("voxa-test-stream-properties")
            for key, value in stream_properties().items():
                structure.set_value(key, value)
        except Exception:  # noqa: BLE001 - a GStreamer without Structure: set what we can directly
            structure = None
    applied["property-structure"] = structure is not None and _set(element, "property-structure", structure)
    if not applied["property-structure"]:
        applied["properties"] = structure is not None and _set(element, "properties", structure)
        if not applied["properties"]:
            for key, value in stream_properties().items():
                applied[key] = _set(element, key, value)
        applied["role"] = _set(element, "role", TEST_STREAM_TAG)
    return applied


def log_reference(decision: ReferenceDecision) -> None:
    """Say the decision out loud: a WARNING when the test voice is inside the reference."""
    if decision.warning:
        log.warning("%s", decision.warning)
    elif decision.mode == MODE_OWN_SINK_INPUT:
        log.info("echo reference: %s (Voxa's own playback only)", decision.source)
    elif decision.mode == MODE_OWN_RENDER:
        log.info("echo reference: Voxa's own rendered samples")
    elif decision.mode == MODE_SINK_MONITOR:
        log.info("echo reference: %s (whole sink)", decision.source)
