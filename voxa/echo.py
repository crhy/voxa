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

import contextlib
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass

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


# --- roadmap task 10: device, headphone and Bluetooth changes as explicit epochs ----------------
#
# The report (research/voxa-hearing/voxa-hearing-research.md lines 84-86, 188-190) and handoff 6.2
# ("a route change, discontinuity, or new sample rate needs an explicit epoch change and recovery
# process") make the rule: on every default-device or Bluetooth-profile change Voxa FLUSHES the
# reference history, RECREATES the canceller, RE-MEASURES latency and CLASSIFIES the new output -
# speakers, headphones/headset, or Bluetooth A2DP/HFP. The acoustic path, its delay and its codec
# buffering all change with the device, so a filter that converged on one cannot be trusted on the
# next; the epoch is what makes that visible instead of letting the old filter half-cancel silently.
#
# Voxa NEVER switches a profile, a port or the default device itself (the same rule as
# :mod:`voxa.echo`'s leftover cleanup, and the header's "never change the user's audio"): the
# watcher only READS. WirePlumber already autoswitches to the headset profile when an input stream
# opens; that change is something Voxa reacts to, never something it causes.

OUTPUT_SPEAKERS = "speakers"
OUTPUT_HEADPHONES = "headphones-headset"
OUTPUT_BT_A2DP = "bt-a2dp"
OUTPUT_BT_HFP = "bt-hfp"
OUTPUT_CLASSES = (OUTPUT_SPEAKERS, OUTPUT_HEADPHONES, OUTPUT_BT_A2DP, OUTPUT_BT_HFP)

# What counts as a Bluetooth path in the device properties of both PulseAudio and PipeWire.
_BT_BUS = ("bluetooth", "bluez")
_BT_API = ("bluez", "bluez5")
# A handsfree/SCO path: it carries BOTH directions, its own codec (CVSD/mSBC) and 100-300 ms of
# link buffering, and the microphone sits in the same headset as the speaker.
# ("phone" is NOT a mark: every bluetooth headset carries device.form_factor "headphone", and
# "hands-free" is written with a hyphen far more often than "handsfree".)
_HFP_MARKS = ("hfp", "hsp", "sco", "headset", "handsfree", "hands-free", "head-unit", "headset-unit",
              "handset", "ofono", "cvsd", "msbc")
# A2DP: the music profile, one direction, SBC/LDAC/aptX buffering on top of the acoustic delay.
_A2DP_MARKS = ("a2dp", "sbc", "ldac", "aptx", "aac", "high", "music", "a2dp-sink")
# A headphone/earphone jack or a wired headset: the report's "no acoustic echo path" case.
_HEADPHONE_MARKS = ("headphone", "headset", "handset", "wired-headset", "handsfree")
_HEADPHONE_FORM_FACTORS = ("headphones", "headset", "handsfree", "earphone")
_SPEAKER_FORM_FACTORS = ("speakers", "internal-speaker", "internal-speakers", "audio-card",
                         "yamaha-rta", "hdmi", "x11-sink")

# pactl/card verbs the watcher may ever run. Anything else is a change to the user's audio and is
# refused here (test_echo_device_watch.test_watcher_never_changes_the_users_audio).
READ_ONLY_VERBS = ("list", "get-default-sink", "get-default-source", "info")

_PROP_LINE = re.compile(r'^\s*(?P<key>[A-Za-z0-9_.-]+)\s*=\s*(?P<value>"[^"]*"|\S+)\s*$')
_PORT_LINE = re.compile(r"^\s*(?P<name>[^\s:]+)\s*:\s*(?P<label>.*?)\s*"
                        r"\(type:\s*(?P<type>[^,)]+)(?:,\s*priority:\s*(?P<priority>\d+))?"
                        r"(?:,\s*(?P<avail>available|not available|availability unknown))?\)")


def parse_stanzas(listing: str, kind: str = "") -> list[dict]:
    """``pactl list sinks`` / ``list sources`` / ``list cards`` as one dict per entry.

    ``list`` of dicts with ``number``, ``state``, ``name``, ``description``, ``driver``,
    ``properties`` (dict of str), ``ports`` (list of dicts with name/label/type/priority/
    available), ``active_port``, ``active_profile`` and ``profiles``. Pure text: the fixture and
    the live output are parsed by exactly the same code, and nothing here touches the daemon.
    """
    entries: list[dict] = []
    current: dict | None = None
    section = ""
    for raw in listing.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if re.match(r"^(Sink|Source|Card|Module|Sample)\s*#", line.strip()):
            current = {"number": line.strip().split("#", 1)[1].strip(), "kind":
                       line.strip().split()[0].lower(), "properties": {}, "ports": [],
                       "profiles": [], "active_port": "", "active_profile": ""}
            entries.append(current)
            section = ""
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.endswith(":") and not line.startswith("\t"):
            section = stripped[:-1]
            continue
        if stripped.endswith(":") and line.lstrip().startswith("\t") and section != "Ports":
            section = stripped[:-1]
            continue
        if section == "Properties:" or (line.startswith("\t\t") and "=" in line):
            match = _PROP_LINE.match(line)
            if match:
                value = match.group("value")
                current["properties"][match.group("key")] = value.strip('"')
            continue
        if stripped == "Formats:" or stripped.startswith("Formats:"):
            section = "Formats"
            continue
        if section == "Ports" or (stripped.endswith(")") and ":" in stripped and "=" not in line):
            match = _PORT_LINE.match(line)
            if match:
                current["ports"].append({
                    "name": match.group("name"),
                    "label": match.group("label"),
                    "type": match.group("type").strip(),
                    "priority": int(match.group("priority") or 0),
                    "available": match.group("avail") or "",
                })
                continue
        if stripped.startswith("Active Port:"):
            current["active_port"] = stripped.split(":", 1)[1].strip()
            section = ""
            continue
        if stripped.startswith("Active Profile:"):
            current["active_profile"] = stripped.split(":", 1)[1].strip()
            section = ""
            continue
        if stripped.startswith("Profile:") or stripped.startswith("\tProfile:"):
            current["profiles"].append(stripped.split(":", 1)[1].strip())
            continue
        if stripped.startswith("Available Profiles:"):
            section = "Profiles"
            continue
        if section in ("", "Profiles") and ":" in stripped and not stripped.startswith("\t\t"):
            key, _, value = stripped.partition(":")
            if key.strip() in ("Name", "Description", "Driver", "State", "Monitor Source",
                               "Active Port", "Active Profile"):
                current[key.strip().lower().replace(" ", "_")] = value.strip()
                section = ""
    return entries


def parse_pw_dump(dump_json: str) -> list[dict]:
    """``pw-dump`` JSON as one dict per node that carries ``device.bus``/``device.form_factor``.

    PipeWire reports the same three facts the report names for a PulseAudio sink
    (``device.bus``, ``device.form_factor``, port type) in its node properties; the port itself is a
    separate ``Port`` node, whose name (``analog-output-headphones`` / ``output:analog-stereo``)
    travels here as ``active_port``.
    """
    try:
        nodes = json.loads(dump_json)
    except (ValueError, TypeError):
        return []
    out: list[dict] = []
    for node in nodes if isinstance(nodes, list) else []:
        props = ((node.get("info") or {}).get("props") or {})
        if not isinstance(props, dict):
            continue
        flat = {k: (",".join(map(str, v)) if isinstance(v, list) else str(v))
                for k, v in props.items()}
        if not any(k in flat for k in ("device.bus", "device.form_factor", "node.name",
                                       "media.class")):
            continue
        out.append({
            "number": str(node.get("id", "")),
            "kind": (flat.get("media.class") or "").lower(),
            "name": flat.get("node.name") or flat.get("media.name") or "",
            "description": flat.get("node.description") or flat.get("media.name") or "",
            "properties": flat,
            "ports": [],
            "active_port": flat.get("node.target") or _pw_port_name(flat),
            "active_profile": flat.get("device.profile.name") or "",
        })
    return out


def _pw_port_name(props: dict) -> str:
    """The PipeWire port name a node sits on (``media.name``/``node.name`` carry it)."""
    for key in ("media.name", "node.name", "port.name"):
        value = props.get(key) or ""
        if "-" in value or ":" in value:
            return value
    return ""


def _haystack(entry: dict) -> str:
    parts = [entry.get("name") or "", entry.get("description") or "",
             entry.get("active_port") or "", entry.get("active_profile") or "",
             entry.get("driver") or ""]
    parts += [port.get("name") or "" for port in entry.get("ports") or []]
    parts += [port.get("type") or "" for port in entry.get("ports") or []]
    return " ".join(parts).casefold()


def _all_text(entry: dict) -> str:
    return _haystack(entry) + " " + " ".join((entry.get("properties") or {}).keys()).casefold()


def _haystack_nofilter(entry: dict) -> str:
    """Name, description, driver, active port/profile, property KEYS - never the other ports."""
    parts = [entry.get("name") or "", entry.get("description") or "",
             entry.get("active_port") or "", entry.get("active_profile") or "",
             entry.get("driver") or ""]
    parts += (entry.get("properties") or {}).keys()
    return " ".join(parts).casefold()


def _active_port_text(entry: dict, active_port: str) -> str:
    """The ACTIVE port's own name and type (an A2DP sink lists the headset port too: ignore it)."""
    for port in entry.get("ports") or []:
        if (port.get("name") or "").casefold() == active_port:
            return f"{port.get('name') or ''} {port.get('type') or ''} {port.get('label') or ''}"
    return active_port


def is_bluetooth(entry: dict) -> bool:
    props = entry.get("properties") or {}
    if (props.get("device.bus") or "").casefold() in _BT_BUS:
        return True
    if (props.get("device.api") or "").casefold() in _BT_API:
        return True
    if (props.get("media.class") or "").casefold().startswith("a2dp"):
        return True
    haystack = _haystack(entry)
    # ofono handsets are bluetooth handsfree paths too, and PulseAudio names them ofono_*, not
    # bluez_*, so the daemon's own module/driver mark counts as much as the property does.
    return any(mark in haystack for mark in ("bluez", "bluetooth", "ofono"))


def classify_output(entry: dict | None, *, active_port: str = "", name: str = "") -> str:
    """One output as ``speakers`` / ``headphones-headset`` / ``bt-a2dp`` / ``bt-hfp``.

    Classified from ``device.bus``, ``device.form_factor`` and the ACTIVE port type, exactly the
    three facts the roadmap names - never from the device's marketing description, which lies about
    what a headphone jack actually is. ``None``/unknown classifies as speakers: that is the path
    with the most echo to cancel, so it is the safe default.
    """
    if entry is None and not name:
        return OUTPUT_SPEAKERS
    entry = dict(entry or {})
    if active_port:
        entry["active_port"] = active_port
    if name:
        entry["name"] = name
    props = entry.get("properties") or {}
    port_text = (entry.get("active_port") or "").casefold()
    form_factor = (props.get("device.form_factor") or "").casefold()
    profile = (entry.get("active_profile") or props.get("device.profile.name") or "").casefold()
    if is_bluetooth(entry):
        # Profile first: A2DP and HFP are NOT the same acoustic path (one direction with music
        # codecs vs both directions with SCO), and Voxa never switches between them itself. Only
        # the ACTIVE port and the protocol count: a card lists BOTH profiles, and PulseAudio even
        # shows the headset port on a sink that is playing A2DP right now. The protocol lives in
        # the properties for both daemons (PulseAudio's ``bluetooth.protocol``, WirePlumber's
        # ``api.bluez5.*``), so their VALUES - and never the marketing description - decide.
        blob = (_haystack_nofilter(entry) + " " + " ".join(
                    value for key, value in props.items()
                    if key.startswith("bluetooth.") or key.startswith("api.bluez5")
                    or "protocol" in key or "profile" in key)
                + " " + profile + " " + _active_port_text(entry, port_text))
        if any(mark in blob for mark in _HFP_MARKS):
            return OUTPUT_BT_HFP
        return OUTPUT_BT_A2DP
    if (form_factor in _HEADPHONE_FORM_FACTORS
            or any(mark in port_text for mark in _HEADPHONE_MARKS)
            or any(mark in profile for mark in _HEADPHONE_MARKS)
            or any(mark in port_text.split("-") for mark in ("headphones", "headset"))
            or any(port.get("type", "").casefold() in ("headphones", "headset", "handsfree")
                   for port in (entry.get("ports") or [])
                   if (port.get("name") or "").casefold() == port_text)):
        return OUTPUT_HEADPHONES
    return OUTPUT_SPEAKERS


@dataclass(frozen=True, slots=True)
class OutputTuning:
    """What the output device lets the pipeline dare to do (task 10, report lines 86, 189).

    ``delay_search_ms`` bounds the delay the delay estimator searches (the report's "A2DP widens the
    delay search to 400 ms"), ``residual_margin_db`` is task 5's residual margin the joint gate uses
    to call a frame the user, and the ``joint_ms`` / ``hold_ms`` / ``speech_threshold`` /
    ``margin_db`` fields ARE the task-5 thresholds the headphones case relaxes: with no acoustic
    echo path there is far less risk in hearing the user, so barge-in may be MORE sensitive.
    ``recovery_s`` is how long the epoch's recovery process stays open (handoff 6.2).
    """

    output_class: str = OUTPUT_SPEAKERS
    echo_path_present: bool = True
    delay_search_ms: float = 300.0
    residual_margin_db: float = 6.0
    joint_ms: float = 150.0
    hold_ms: float = 300.0
    speech_threshold: float = 0.65
    coherence_echo: float = 0.5
    erle_drop_db: float = 3.0
    recovery_s: float = 2.0
    # AEC3's own delay search starts blind; a Bluetooth link carries a big known chunk of it, so the
    # seed that goes into ``stream_delay_ms`` is 0 for a normal output and 200 ms for A2DP/HFP.
    aec3_seed_ms: float = 0.0
    # task 5's onset rule (voxa/onset.py): how far the microphone must stand over the speaker path's
    # ceiling, and for how long. The live and bench barge-in both run through these two numbers.
    onset_margin_db: float = 12.0
    onset_evidence_ms: float = 320.0

    @property
    def relaxed(self) -> bool:
        """True when this output relaxed task 5's thresholds rather than tightened them."""
        return self.output_class == OUTPUT_HEADPHONES


def output_tuning(output_class: str | None) -> OutputTuning:
    """The per-output tuning table (task 10). Defaults are today's shipped task-5 values."""
    from .onset import ONSET_EVIDENCE_MS, UNEXPLAINED_DB
    from .voicedetect import (COHERENCE_ECHO, ERLE_DROP_DB, RESIDUAL_MARGIN_DB, SUSPECT_HOLD_MS,
                              SUSPECT_JOINT_MS, SPEECH_PLAYBACK_THRESHOLD)
    base = {"speech_threshold": SPEECH_PLAYBACK_THRESHOLD, "joint_ms": SUSPECT_JOINT_MS,
            "hold_ms": SUSPECT_HOLD_MS, "residual_margin_db": RESIDUAL_MARGIN_DB,
            "coherence_echo": COHERENCE_ECHO, "erle_drop_db": ERLE_DROP_DB,
            "onset_margin_db": UNEXPLAINED_DB, "onset_evidence_ms": ONSET_EVIDENCE_MS}
    if output_class == OUTPUT_HEADPHONES:
        # No acoustic echo path: cancellation can relax and barge-in gets MORE sensitive (report
        # line 86). The joint run is shorter, the Silero playback threshold comes down, and a
        # smaller residual margin is enough to believe the user is there.
        return OutputTuning(output_class=OUTPUT_HEADPHONES, echo_path_present=False,
                            delay_search_ms=300.0, residual_margin_db=4.0, joint_ms=100.0,
                            hold_ms=200.0, speech_threshold=0.55,
                            coherence_echo=base["coherence_echo"],
                            erle_drop_db=base["erle_drop_db"],
                            onset_margin_db=9.0, onset_evidence_ms=220.0)
    if output_class in (OUTPUT_BT_A2DP, OUTPUT_BT_HFP):
        # Bluetooth adds codec buffering on top of the room: the search window widens to 400 ms and
        # the residual margin is RAISED, because a big unknown delay makes the residual prediction
        # less trustworthy - the gate needs more headroom before it calls a frame the user.
        return OutputTuning(output_class=output_class, echo_path_present=True,
                            delay_search_ms=400.0, residual_margin_db=9.0, joint_ms=200.0,
                            hold_ms=SUSPECT_HOLD_MS, speech_threshold=SPEECH_PLAYBACK_THRESHOLD,
                            coherence_echo=base["coherence_echo"],
                            erle_drop_db=base["erle_drop_db"], aec3_seed_ms=200.0,
                            onset_margin_db=15.0)
    return OutputTuning(output_class=OUTPUT_SPEAKERS, echo_path_present=True,
                        delay_search_ms=300.0, residual_margin_db=base["residual_margin_db"],
                        joint_ms=base["joint_ms"], hold_ms=base["hold_ms"],
                        speech_threshold=base["speech_threshold"],
                        coherence_echo=base["coherence_echo"],
                        erle_drop_db=base["erle_drop_db"])


@dataclass(frozen=True, slots=True)
class OutputSnapshot:
    """What the computer's output and microphone are right now, cheaply, read-only."""

    sink: str
    source: str
    output_class: str
    bt_profile: str = ""

    @property
    def key(self) -> tuple:
        """The identity that must not change unnoticed: any change is an epoch."""
        return (self.sink, self.source, self.output_class, self.bt_profile)


def _read(verb: str, *args: str, runner=subprocess.run) -> str:
    """A read-only pactl call, or "" - Voxa never changes the user's audio (header rule 3)."""
    if verb not in READ_ONLY_VERBS:
        raise ValueError(f"the device watcher may only read; refused {verb!r}")
    code, out = _pactl(verb, *args, runner=runner)
    return out if code == 0 else ""


def device_snapshot(runner=subprocess.run) -> OutputSnapshot | None:
    """The default sink, default source, output class and Bluetooth profile, as one snapshot.

    ``None`` when pactl cannot be reached at all (no PulseAudio here): the watcher then says nothing
    and the pipeline keeps its current epoch, rather than declaring the audio gone.
    """
    default_sink = _read("get-default-sink", runner=runner).strip()
    default_source = _read("get-default-source", runner=runner).strip()
    if not default_sink and not default_source:
        return None
    sinks = parse_stanzas(_read("list", "sinks", runner=runner))
    sources = parse_stanzas(_read("list", "sources", runner=runner))
    cards = parse_stanzas(_read("list", "cards", runner=runner))
    sink = next((entry for entry in sinks if entry.get("name") == default_sink), None)
    source = next((entry for entry in sources if entry.get("name") == default_source), None)
    bt_profile = ""
    for card in cards:
        profile = card.get("active_profile") or ""
        if "bluez" in profile.casefold() or is_bluetooth(card):
            bt_profile = profile or (card.get("name") or "")
            break
    return OutputSnapshot(sink=default_sink, source=default_source, bt_profile=bt_profile,
                          output_class=classify_output(sink, name=default_sink))


class DeviceWatcher:
    """Read-only watcher: a default sink/source or Bluetooth-profile change fires ``on_change``.

    PulseAudio's own events (``pulsectl.event_listen``, i.e. ``pa_context_subscribe``) are the fast
    path the roadmap asks for; when pulsectl is not importable here the watcher polls the same
    read-only pactl facts on a fixed interval, which is what makes this testable offline and needs
    no new dependency in the flatpak. Nothing in this class can change the user's audio, and it
    never switches a profile itself - WirePlumber's autoswitch to the headset profile is an event
    for Voxa to REACT to (flush the reference, recreate the canceller, re-measure latency).
    """

    def __init__(self, on_change=None, *, runner=subprocess.run, poll_seconds: float = 2.0,
                 log: logging.Logger | None = None) -> None:
        self.on_change = on_change
        self.runner = runner
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.log = log or logging.getLogger("voxa.echo")
        self.snapshot: OutputSnapshot | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self.changes = 0

    def poll_once(self) -> dict | None:
        """One read: a change of the output identity returns (new, old) and calls ``on_change``."""
        with self._lock:
            new = device_snapshot(runner=self.runner)
            if new is None:
                return None
            old = self.snapshot
            self.snapshot = new
            if old is not None and old.key == new.key:
                return None
            if old is None:
                return None
            self.changes += 1
            change = {"from": old, "to": new,
                      "reason": ("output-class" if old.output_class != new.output_class else
                                 ("bluetooth-profile" if old.bt_profile != new.bt_profile else
                                  "default-device"))}
            if self.on_change is not None:
                try:
                    self.on_change(change)
                except Exception:  # noqa: BLE001 - the watcher must never break the capture path
                    self.log.exception("device-change epoch callback failed")
            return change

    def start(self) -> None:
        """Watch in the background (daemon thread). Safe to call twice; never raises."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voxa-device-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def _loop(self) -> None:
        pulse = self._pulse_events()
        try:
            while not self._stop.wait(self._wake.wait(0.2) or self.poll_seconds):
                self.poll_once()
            if self._wake.is_set():
                self._wake.clear()
                self.poll_once()
        finally:
            if pulse is not None:
                pulse()

    def _pulse_events(self):
        """Ask PulseAudio to wake us; returns a teardown callable, or None without pulsectl."""
        try:
            import pulsectl
        except ImportError:
            return None
        try:
            ctx = pulsectl.Pulsectl("voxa-device-watch")
        except Exception:  # noqa: BLE001 - no daemon, no events: polling covers it
            return None
        thread: threading.Thread | None = None

        def wake(_event=None) -> None:
            self._wake.set()

        def listen() -> None:
            try:
                ctx.event_listen(wake)
            except Exception:  # noqa: BLE001 - the loop died; polling still covers every change
                self._stop.set()

        def teardown() -> None:
            self._stop.set()
            self._wake.set()
            if thread is not None:
                thread.join(timeout=1.0)
            with contextlib.suppress(Exception):
                ctx.close()

        try:
            thread = threading.Thread(target=listen, name="voxa-pulse-events", daemon=True)
            thread.start()
            return teardown
        except Exception:  # noqa: BLE001
            teardown()
            return None
