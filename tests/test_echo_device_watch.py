"""Roadmap task 10: the read-only device watcher, its parsers and the output classes.

Realistic fixtures only: the ``pactl list`` texts below are this machine's own output (a USB
microphone, a Realtek line-out with a NOT-available headphone port, an HDMI sink) plus the two
Bluetooth stanzas PulseAudio prints when a headset is connected in A2DP and in HFP/HSP, and the
PipeWire ``pw-dump`` nodes a WirePlumber session shows for the same devices. Nothing here touches
a sound card: the watcher is exercised through a scripted fake ``subprocess.run``.
"""

from __future__ import annotations

import json
import types

import pytest

from voxa.echo import (
    OUTPUT_BT_A2DP,
    OUTPUT_BT_HFP,
    OUTPUT_HEADPHONES,
    OUTPUT_SPEAKERS,
    DeviceWatcher,
    classify_output,
    device_snapshot,
    output_tuning,
    parse_pw_dump,
    parse_stanzas,
)

# ---------------------------------------------------------------- fixtures --------------------

# pactl list sinks, abridged from what this machine actually prints (Blue USB mic as a sink, the
# Realtek line-out whose headphone port is NOT available, and an HDMI sink).
SINKS = """\
Sink #0
\tState: SUSPENDED
\tName: alsa_output.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo
\tDescription: Blue Microphones Analog Stereo
\tDriver: module-alsa-card.c
\tMonitor Source: alsa_output.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo.monitor
\tLatency: 0 usec, configured 0 usec
\tFlags: HARDWARE HW_MUTE_CTRL HW_VOLUME_CTRL DECIBEL_VOLUME LATENCY
\tProperties:
\t\tdevice.api = "alsa"
\t\tdevice.bus = "usb"
\t\tdevice.form_factor = "microphone"
\t\tdevice.profile.name = "analog-stereo"
\t\tdevice.description = "Blue Microphones Analog Stereo"
\tPorts:
\t\tanalog-output-speaker: Speakers (type: Speaker, priority: 10000, availability unknown)
\tActive Port: analog-output-speaker
\tFormats:
\t\tpcm

Sink #1
\tState: SUSPENDED
\tName: alsa_output.pci-0000_10_00.6.analog-stereo
\tDescription: Ryzen HD Audio Controller Analog Stereo
\tDriver: module-alsa-card.c
\tProperties:
\t\tdevice.api = "alsa"
\t\tdevice.bus = "pci"
\t\tdevice.profile.name = "analog-stereo"
\t\tdevice.description = "Ryzen HD Audio Controller Analog Stereo"
\tPorts:
\t\tanalog-output-lineout: Line Out (type: Line, priority: 9000, available)
\t\tanalog-output-headphones: Headphones (type: Headphones, priority: 9900, not available)
\tActive Port: analog-output-lineout
\tFormats:
\t\tpcm

Sink #2
\tState: SUSPENDED
\tName: alsa_output.pci-0000_01_00.1.hdmi-stereo
\tDescription: AD103 High Definition Audio Controller Digital Stereo (HDMI)
\tDriver: module-alsa-card.c
\tProperties:
\t\tdevice.api = "alsa"
\t\tdevice.bus = "pci"
\t\tdevice.profile.name = "hdmi-stereo"
\tPorts:
\t\thdmi-output-0: HDMI / DisplayPort (type: HDMI, priority: 5900, available)
\tActive Port: hdmi-output-0
\tFormats:
\t\tpcm
"""

# The same machine with a Sony WH-1000XM4 plugged in: PulseAudio's bluez5 device module shows the
# A2DP sink (music, one direction) and the headset sink (HSP/HFP, both directions).
BLUEZ_SINKS = SINKS + """\
Sink #3
\tState: RUNNING
\tName: bluez_output.11_22_33_44_55_66.1
\tDescription: WH-1000XM4
\tDriver: module-bluez5-device.c
\tLatency: 20298 usec, configured 20298 usec
\tFlags: HARDWARE DECIBEL_VOLUME LATENCY
\tProperties:
\t\tbluetooth.protocol = "a2dp-sink"
\t\tbluetooth.address = "11_22_33_44_55_66"
\t\tdevice.api = "bluez5"
\t\tdevice.class = "sound"
\t\tdevice.bus = "bluetooth"
\t\tdevice.form_factor = "headphone"
\t\tdevice.icon_name = "bluetooth"
\t\tdevice.description = "WH-1000XM4"
\t\tapi.bluez5.codecs = "[SBC, 44.1 kHz, 53 kbps]"
\tPorts:
\t\toutput:a2dp-sink (type: A2DP-sink, priority: 900)
\t\toutput:hands-free-unit (type: Headset, priority: 100)
\tActive Port: output:a2dp-sink
\tFormats:
\t\tsbc

Sink #4
\tState: SUSPENDED
\tName: bluez_output.11_22_33_44_55_66.headset-unit
\tDescription: WH-1000XM4 Headset
\tDriver: module-bluez5-device.c
\tFlags: HW_MUTE_CTRL LATENCY
\tProperties:
\t\tbluetooth.protocol = "headset"
\t\tbluetooth.address = "11_22_33_44_55_66"
\t\tdevice.api = "bluez5"
\t\tdevice.class = "sound"
\t\tdevice.bus = "bluetooth"
\t\tdevice.form_factor = "headphone"
\t\tdevice.description = "WH-1000XM4 Headset"
\tPorts:
\t\theadset-output (type: Headset, priority: 100)
\tActive Port: headset-output
\tFormats:
\t\tcvsd
"""

HEADPHONE_PORT_SINK = """\
Sink #1
\tState: RUNNING
\tName: alsa_output.pci-0000_10_00.6.analog-stereo
\tDescription: Ryzen HD Audio Controller Analog Stereo
\tDriver: module-alsa-card.c
\tProperties:
\t\tdevice.api = "alsa"
\t\tdevice.bus = "pci"
\t\tdevice.profile.name = "analog-stereo"
\tPorts:
\t\tanalog-output-lineout: Line Out (type: Line, priority: 9000, not available)
\t\tanalog-output-headphones: Headphones (type: Headphones, priority: 9900, available)
\tActive Port: analog-output-headphones
"""

CARDS_BLUETOOTH = """\
Card #12
\tName: bluez_card.11_22_33_44_55_66
\tDriver: module-bluez5-device.c
\tProperties:
\t\tdevice.api = "bluez5"
\t\tdevice.class = "sound"
\t\tdevice.bus = "bluetooth"
\t\tdevice.form_factor = "headphone"
\t\tbluetooth.protocol = "a2dp-sink"
\t\tdevice.description = "WH-1000XM4"
\tActive Profile: output:a2dp-sink
"""

SOURCES = """\
Source #20
\tName: alsa_input.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo
\tDescription: Blue Microphones Analog Stereo
\tDriver: module-alsa-card.c
\tProperties:
\t\tdevice.api = "alsa"
\t\tdevice.bus = "usb"
\t\tdevice.form_factor = "microphone"
"""

# pw-dump: the WirePlumber/PipeWire nodes for the same three kinds of output.
PW_DUMP = json.dumps([
    {"id": 34, "type": "PipeWire:Interface:Node",
     "info": {"props": {
         "media.class": "Audio/Sink",
         "node.name": "a2dp_sink.float32.1",
         "node.description": "WH-1000XM4",
         "media.name": "A2DP Sink",
         "device.api": "bluez5",
         "device.bus": "bluetooth",
         "device.form_factor": "headphone",
         "api.bluez5.codec": "sbc",
         "device.profile.name": "output:a2dp-sink"}}},
    {"id": 35, "type": "PipeWire:Interface:Node",
     "info": {"props": {
         "media.class": "Audio/Sink",
         "node.name": "bluez_output.11_22_33_44_55_66.headset-unit",
         "media.name": "Headset Audio Gateway (HFP)",
         "device.api": "bluez5",
         "device.bus": "bluetooth",
         "device.form_factor": "headphone",
         "api.bluez5.connection": "client",
         "device.profile.name": "input:hfp+output:hfp"}}},
    {"id": 36, "type": "PipeWire:Interface:Node",
     "info": {"props": {
         "media.class": "Audio/Sink",
         "node.name": "alsa_output.pci-0000_10_00.6.analog-stereo",
         "node.description": "Ryzen HD Audio Controller",
         "media.name": "analog-output-headphones",
         "device.api": "alsa",
         "device.bus": "pci",
         "device.form_factor": "internal-speaker"}}},
    {"id": 37, "type": "PipeWire:Interface:Node",
     "info": {"props": {
         "media.class": "Audio/Source",
         "node.name": "alsa_input.usb-Generic_Blue_Microphones",
         "device.api": "alsa",
         "device.bus": "usb",
         "device.form_factor": "microphone"}}},
])


def make_daemon(sinks: str = SINKS, sources: str = SOURCES, cards: str = "",
                default_sink: str = "alsa_output.pci-0000_10_00.6.analog-stereo",
                default_source: str = "alsa_input.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo",
                calls: list | None = None):
    """A scripted ``subprocess.run`` for a machine with pactl; records every argv it sees."""

    def runner(argv, **_kwargs):
        if calls is not None:
            calls.append(list(argv))
        out = ""
        if "get-default-sink" in argv:
            out = default_sink
        elif "get-default-source" in argv:
            out = default_source
        elif "sinks" in argv:
            out = sinks
        elif "sources" in argv:
            out = sources
        elif "cards" in argv:
            out = cards
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    return runner


def make_raising_daemon(**kwargs):
    """A daemon that fails the moment anyone asks it to CHANGE anything."""
    inner = make_daemon(**kwargs)

    def runner(argv, **kw):
        joined = " ".join(argv)
        for mutating in ("set-default", "set-card-profile", "set-sink-port", "unload-module",
                         "load-module", "suspend", "set-sink-volume", "move-sink-input"):
            if mutating in joined:
                raise AssertionError(f"the device watcher changed the user's audio: {joined}")
        return inner(argv, **kw)

    return runner


# ---------------------------------------------------------------- parsers -------------------

def test_parse_stanzas_reads_the_real_listing() -> None:
    sinks = parse_stanzas(SINKS)
    assert [entry["name"] for entry in sinks] == [
        "alsa_output.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo",
        "alsa_output.pci-0000_10_00.6.analog-stereo",
        "alsa_output.pci-0000_01_00.1.hdmi-stereo",
    ]
    realtek = sinks[1]
    assert realtek["active_port"] == "analog-output-lineout"
    assert [(port["name"], port["type"], port["available"]) for port in realtek["ports"]] == [
        ("analog-output-lineout", "Line", "available"),
        ("analog-output-headphones", "Headphones", "not available"),
    ]
    assert realtek["properties"]["device.bus"] == "pci"
    assert sinks[0]["properties"]["device.form_factor"] == "microphone"
    # the Formats: section is NOT mistaken for properties
    assert "pcm" not in realtek["properties"]


def test_parse_stanzas_reads_active_profile_and_bluez_protocol() -> None:
    cards = parse_stanzas(CARDS_BLUETOOTH)
    assert len(cards) == 1
    assert cards[0]["active_profile"] == "output:a2dp-sink"
    assert cards[0]["properties"]["bluetooth.protocol"] == "a2dp-sink"


def test_parse_pw_dump_reads_the_wireplumber_nodes() -> None:
    nodes = parse_pw_dump(PW_DUMP)
    assert [entry["properties"]["node.name"] for entry in nodes] == [
        "a2dp_sink.float32.1",
        "bluez_output.11_22_33_44_55_66.headset-unit",
        "alsa_output.pci-0000_10_00.6.analog-stereo",
        "alsa_input.usb-Generic_Blue_Microphones",
    ]
    assert classify_output(nodes[0]) == OUTPUT_BT_A2DP
    assert classify_output(nodes[1]) == OUTPUT_BT_HFP
    # the jack the port node names is what makes this the headphone case, not the form factor
    assert classify_output(nodes[2]) == OUTPUT_HEADPHONES
    assert classify_output(nodes[3]) == OUTPUT_SPEAKERS


def test_parse_pw_dump_ignores_garbage() -> None:
    assert parse_pw_dump("not json") == []
    assert parse_pw_dump("[]") == []


# ---------------------------------------------------------------- classification ------------

def test_line_out_is_speakers_even_with_a_headphone_port_present() -> None:
    """The port exists but is NOT available: the computer plays through the line, so task 5's
    thresholds stay where they are."""
    sink = parse_stanzas(SINKS)[1]
    assert classify_output(sink) == OUTPUT_SPEAKERS


def test_the_active_headphone_port_is_the_headphone_case() -> None:
    sink = parse_stanzas(HEADPHONE_PORT_SINK)[0]
    assert classify_output(sink) == OUTPUT_HEADPHONES


def test_a_wire_headset_form_factor_is_the_headphone_case() -> None:
    entry = {"name": "alsa_output.usb-Jabra.analog-stereo", "properties":
             {"device.form_factor": "headset"}, "ports": [], "active_port": ""}
    assert classify_output(entry) == OUTPUT_HEADPHONES


def test_bluez_a2dp_and_hfp_are_separate_classes() -> None:
    sinks = parse_stanzas(BLUEZ_SINKS)
    a2dp, hfp = sinks[3], sinks[4]
    assert classify_output(a2dp) == OUTPUT_BT_A2DP
    assert classify_output(hfp) == OUTPUT_BT_HFP
    # an explicitly given active port travels with it (the watcher reads it separately)
    assert classify_output(hfp, active_port="output:hands-free-unit") == OUTPUT_BT_HFP


def test_an_ofono_hsp_handset_is_hfp_by_its_port_type() -> None:
    entry = {"name": "ofono_handset_voice_downstream_audio",
             "description": "Jabra Speak (HSP/HFP)",
             "properties": {"device.class": "sound"}, "ports":
             [{"name": "hands-free-unit", "type": "Handsfree", "priority": 100}],
             "active_port": "hands-free-unit"}
    assert classify_output(entry) == OUTPUT_BT_HFP


def test_classification_never_uses_the_marketing_description_alone() -> None:
    """A device NAMED "Headphones" but wired to a line-out port is still the speaker path: the
    report's three facts (bus, form factor, ACTIVE port) decide, not the name."""
    entry = {"name": "alsa_output.pci-lineout", "description": "Beyerdynamic Headphones",
             "properties": {}, "ports": [{"name": "analog-output-lineout", "type": "Line",
                                          "priority": 9000}],
             "active_port": "analog-output-lineout"}
    assert classify_output(entry) == OUTPUT_SPEAKERS
    assert classify_output(None) == OUTPUT_SPEAKERS


# ---------------------------------------------------------------- snapshots -----------------

def test_device_snapshot_names_the_default_output_and_its_class() -> None:
    snap = device_snapshot(runner=make_daemon())
    assert snap is not None
    assert snap.sink == "alsa_output.pci-0000_10_00.6.analog-stereo"
    assert snap.output_class == OUTPUT_SPEAKERS
    assert snap.bt_profile == ""


def test_device_snapshot_sees_the_headset_and_its_profile() -> None:
    snap = device_snapshot(runner=make_daemon(sinks=BLUEZ_SINKS, cards=CARDS_BLUETOOTH,
                                              default_sink="bluez_output.11_22_33_44_55_66.1"))
    assert snap is not None
    assert snap.output_class == OUTPUT_BT_A2DP
    assert snap.bt_profile == "output:a2dp-sink"


def test_device_snapshot_is_none_when_there_is_no_pactl() -> None:
    def dead(argv, **_kwargs):
        raise FileNotFoundError("no pactl on this machine")

    assert device_snapshot(runner=dead) is None
    assert device_snapshot(runner=make_daemon(default_sink="", default_source="")) is None


# ---------------------------------------------------------------- the watcher ---------------

def test_watcher_fires_once_per_change_and_reports_the_reason() -> None:
    daemon_a = make_daemon()
    daemon_bt = make_daemon(sinks=BLUEZ_SINKS, cards=CARDS_BLUETOOTH,
                            default_sink="bluez_output.11_22_33_44_55_66.1")
    seen: list[dict] = []
    watcher = DeviceWatcher(seen.append, runner=daemon_a, poll_seconds=0.05)

    assert watcher.poll_once() is None          # first read only learns the baseline
    assert watcher.poll_once() is None          # nothing moved
    watcher.runner = daemon_bt
    change = watcher.poll_once()
    assert change is not None
    assert change["reason"] == "output-class"
    assert change["from"].output_class == OUTPUT_SPEAKERS
    assert change["to"].output_class == OUTPUT_BT_A2DP
    assert seen and seen[0]["to"].output_class == OUTPUT_BT_A2DP
    assert watcher.changes == 1
    assert watcher.poll_once() is None          # repeats do not re-fire an epoch
    assert watcher.changes == 1


def test_watcher_calls_a_profile_change_its_own_epoch() -> None:
    a2dp = make_daemon(sinks=BLUEZ_SINKS, cards=CARDS_BLUETOOTH,
                       default_sink="bluez_output.11_22_33_44_55_66.1")
    hfp = make_daemon(sinks=BLUEZ_SINKS, default_sink="bluez_output.11_22_33_44_55_66.headset-unit")
    watcher = DeviceWatcher(runner=a2dp, poll_seconds=0.05)
    watcher.poll_once()
    watcher.runner = hfp
    change = watcher.poll_once()
    assert change is not None and change["reason"] == "output-class"
    assert change["to"].output_class == OUTPUT_BT_HFP


def test_watcher_never_changes_the_users_audio() -> None:
    """Every pactl the watcher may run is read-only; a mutating verb raises in the test daemon."""
    calls: list[list[str]] = []
    watcher = DeviceWatcher(runner=make_raising_daemon(calls=calls), poll_seconds=0.05)
    for _ in range(3):
        watcher.poll_once()
    assert calls, "the watcher read something"
    for argv in calls:
        assert {"list", "get-default-sink", "get-default-source", "info"} >= {argv[1]}


def test_read_refuses_a_mutating_verb_before_running_it() -> None:
    from voxa.echo import _read

    def runner(argv, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("a mutating verb reached the daemon")

    with pytest.raises(ValueError):
        _read("set-default-sink", "alsa_output.other", runner=runner)


def test_a_daemon_that_vanishes_leaves_the_epoch_where_it_is() -> None:
    watcher = DeviceWatcher(runner=make_daemon(), poll_seconds=0.05)
    assert watcher.poll_once() is None

    def gone(argv, **_kwargs):
        raise OSError("the pulse daemon died mid-switch")

    watcher.runner = gone
    assert watcher.poll_once() is None
    assert watcher.changes == 0


def test_a_broken_callback_cannot_kill_the_watcher() -> None:
    def boom(change: dict) -> None:
        raise RuntimeError("the epoch callback broke")

    daemon_a = make_daemon()
    daemon_bt = make_daemon(sinks=BLUEZ_SINKS, default_sink="bluez_output.11_22_33_44_55_66.1")
    watcher = DeviceWatcher(boom, runner=daemon_a, poll_seconds=0.05)
    watcher.poll_once()
    watcher.runner = daemon_bt
    assert watcher.poll_once() is not None
    assert watcher.changes == 1


# ---------------------------------------------------------------- tuning -------------------

def test_headphones_relax_task_fives_thresholds() -> None:
    speakers, phones = output_tuning(OUTPUT_SPEAKERS), output_tuning(OUTPUT_HEADPHONES)
    assert phones.echo_path_present is False
    assert phones.joint_ms < speakers.joint_ms
    assert phones.hold_ms < speakers.hold_ms
    assert phones.speech_threshold < speakers.speech_threshold
    assert phones.residual_margin_db < speakers.residual_margin_db
    assert phones.onset_margin_db < speakers.onset_margin_db
    assert phones.onset_evidence_ms < speakers.onset_evidence_ms
    assert phones.relaxed and not speakers.relaxed
    # a jack with no echo path still gets no extra delay search and no AEC3 seed
    assert phones.delay_search_ms == speakers.delay_search_ms == 300.0
    assert phones.aec3_seed_ms == speakers.aec3_seed_ms == 0.0


def test_bluetooth_widens_the_delay_search_and_raises_the_residual_margin() -> None:
    for cls in (OUTPUT_BT_A2DP, OUTPUT_BT_HFP):
        bt, speakers = output_tuning(cls), output_tuning(OUTPUT_SPEAKERS)
        assert bt.delay_search_ms == 400.0
        assert bt.residual_margin_db > speakers.residual_margin_db
        assert bt.aec3_seed_ms == 200.0
        assert bt.onset_margin_db > speakers.onset_margin_db
        assert bt.delay_search_ms > speakers.delay_search_ms
        assert not bt.relaxed
    assert output_tuning(OUTPUT_BT_HFP).recovery_s == output_tuning(OUTPUT_SPEAKERS).recovery_s


def test_unknown_output_class_takes_the_speaker_tuning() -> None:
    assert output_tuning(None).output_class == OUTPUT_SPEAKERS
    assert output_tuning("kitchen-monitor").output_class == OUTPUT_SPEAKERS
