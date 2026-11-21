"""V4: the voxatest soak (self-triggers, Poisson bound) and fixture isolation of the test voice.

Everything here is offline: no microphone, no speakers, no sound server. The soak's counting and its
one-sided 95% Poisson bound are pure arithmetic on log lines; the isolation is a property structure on
a GStreamer element, exercised with the same fake GStreamer as ``tests/test_audio_webrtc.py``; the
reference policy is judged from a scripted ``pactl`` listing. The live run itself is rhY's (it plays
sound), and the tests prove the command refuses to make a sound without ``--yes``.
"""

from __future__ import annotations

import json
import logging
import types

import numpy as np
import pytest

from voxa import reference
from voxa.audio import AudioCapture, ReferenceRing
from voxa.reference import (
    MODE_OWN_RENDER,
    MODE_OWN_SINK_INPUT,
    MODE_SINK_MONITOR,
    MODE_UNKNOWN,
    TEST_STREAM_TAG,
)
from voxa.speech import SpeechService
from voxatest import __main__ as voxatest_main
from voxatest import isolation
from voxatest import soak as soak_mod
from voxatest.config import HarnessSettings
from voxatest.doctor import FAIL, PASS, SKIP, WARN, Check, check_drift, format_checks, run_checks

WAKE = "voxa"

# ----------------------------------------------------------------- the corpus


def test_plan_is_seeded_and_fills_the_hours():
    a = soak_mod.soak_plan(1.0, seed=7)
    b = soak_mod.soak_plan(1.0, seed=7)
    assert [reply.text for reply in a] == [reply.text for reply in b]
    assert [reply.voice for reply in a] == [reply.voice for reply in b]
    summary = soak_mod.plan_summary(a, 2.5)
    # The plan must cover the requested time, not a fraction of it.
    assert summary["wall_hours"] == pytest.approx(1.0, rel=0.05)
    assert summary["replies"] == len(a) > 20


def test_plan_covers_every_trigger_family_and_the_wake_word():
    plan = soak_mod.soak_plan(2.0, seed=3, wake_word="computer")
    families = {reply.family for reply in plan}
    assert set(soak_mod.TRIGGER_FAMILIES) <= families
    joined = " ".join(reply.text for reply in plan).lower()
    for word in ("computer", "stop", "cancel", "offline", "wait", "pause"):
        assert word in joined
    assert "{wake}" not in joined


def test_replies_are_long_and_varied():
    plan = soak_mod.soak_plan(1.0, seed=1)
    assert min(len(reply.text) for reply in plan) > 120
    assert len({reply.text for reply in plan}) >= 10
    assert all(reply.est_seconds > 1.0 for reply in plan)
    assert all(reply.index == number for number, reply in enumerate(plan, start=1))


def test_plan_rejects_a_bad_length_and_a_negative_gap():
    with pytest.raises(ValueError):
        soak_mod.soak_plan(0)
    with pytest.raises(ValueError):
        soak_mod.soak_plan(1.0, gap_seconds=-1)


def test_plan_summary_names_the_voices():
    plan = soak_mod.soak_plan(0.5, seed=0, voices=("A", "B"))
    summary = soak_mod.plan_summary(plan, 2.5)
    assert summary["voices"] == ["A", "B"]
    assert summary["trigger_replies"] == sum(1 for reply in plan if reply.family in soak_mod.TRIGGER_FAMILIES)


# ---------------------------------------------------------------- the Poisson bound


def test_zero_events_in_a_hundred_hours_gives_the_handoffs_bound():
    # handoff 8.4: "Zero events in 100 hours implies an approximate one-sided 95% Poisson
    # upper bound of 0.03 events/hour".
    assert soak_mod.poisson_upper_bound(0, 100.0) == pytest.approx(0.03, abs=0.001)


def test_zero_events_in_one_hour_is_three_per_hour():
    # The brief: "reports events/hour with the one-sided 95% Poisson upper bound (3/hours when zero)".
    assert soak_mod.poisson_upper_bound(0, 1.0) == pytest.approx(3.0, abs=0.01)


def test_bound_matches_the_chi_square_for_three_events():
    # chi-square 0.95 with 2*(3+1) degrees of freedom is 15.5087; half of it over 10 hours.
    assert soak_mod.poisson_upper_bound(3, 10.0) == pytest.approx(15.5087 / 2.0 / 10.0, rel=1e-3)


def test_bound_grows_with_events_and_shrinks_with_time():
    one = soak_mod.poisson_upper_bound(1, 10.0)
    two = soak_mod.poisson_upper_bound(2, 10.0)
    long_run = soak_mod.poisson_upper_bound(1, 100.0)
    assert one < two
    assert long_run < one
    assert soak_mod.poisson_upper_bound(0, 10.0) < one


def test_bound_is_a_rate_the_observed_count_is_likely_under():
    # At the bound, P(Poisson(bound*hours) <= count) is exactly 5 %: the definition of one-sided 95 %.
    bound = soak_mod.poisson_upper_bound(4, 8.0)
    assert soak_mod.poisson_cdf(4, bound * 8.0) == pytest.approx(0.05, abs=1e-4)


def test_cdf_and_bound_argument_checks():
    assert soak_mod.poisson_cdf(0, 3.0) == pytest.approx(np.exp(-3.0), rel=1e-9)
    with pytest.raises(ValueError):
        soak_mod.poisson_upper_bound(0, 0.0)
    with pytest.raises(ValueError):
        soak_mod.poisson_upper_bound(-1, 1.0)
    with pytest.raises(ValueError):
        soak_mod.poisson_upper_bound(1, 1.0, confidence=1.5)


# ---------------------------------------------------------------- the counting

ACTION_RECORDS = [
    {"time": "10:00:01", "heard": "stop the music", "route": "rule", "tool": "media_stop", "ok": True},
    {"time": "10:00:04", "heard": "", "route": "rule", "tool": "timer", "ok": True},
    {"time": "10:00:07", "heard": "", "route": "none", "tool": "", "ok": None},
]

BARGE_LINES = [
    "10:00:02 yield generation=3 onset_audio_ms=120.0 margin_db=2.50",
    "10:00:05 INTERRUPT",
    "10:00:06 pause frames=40",
    "10:00:08 gate-open level=0.41 baseline=0.10",
    "10:00:09 offline-interrupt heard='offline'",
    "not an event",
]


def test_barge_lines_parse_into_typed_events():
    event = soak_mod.parse_barge_line(BARGE_LINES[0])
    assert event["event"] == "yield"
    assert event["generation"] == 3
    assert event["onset_audio_ms"] == pytest.approx(120.0)
    assert event["margin_db"] == pytest.approx(2.5)
    assert soak_mod.parse_barge_line('{"event": "tts-stop", "t_s": 1.5}')[ "event"] == "tts-stop"
    assert soak_mod.parse_barge_line("10:00:05 INTERRUPT")["event"] == "INTERRUPT"
    assert soak_mod.parse_barge_line("nonsense") is None
    assert soak_mod.parse_barge_line("") is None
    assert soak_mod.parse_barge_line("{not json") is None


def test_self_pause_names_and_the_microphone_pause_is_excluded():
    assert soak_mod.is_self_pause({"event": "yield"})
    assert soak_mod.is_self_pause({"event": "INTERRUPT"})
    assert soak_mod.is_self_pause({"event": "tts-stop"})
    assert soak_mod.is_self_pause({"event": "offline-interrupt"})
    # A "pause" event in the runner marks a microphone segment boundary, not an output pause:
    # counting it would measure the segmentation, not the self-interruption (voxa/bench/metrics.py).
    assert not soak_mod.is_self_pause({"event": "pause"})
    assert not soak_mod.is_self_pause({"event": "gate-open"})
    assert not soak_mod.is_self_pause({})


def test_count_self_events_counts_both_sources_and_reports_the_bound():
    events = [soak_mod.parse_barge_line(line) for line in BARGE_LINES]
    result = soak_mod.count_self_events(ACTION_RECORDS, events, hours=2.0, replies_spoken=100)
    assert result["self_command_count"] == 2  # the heard line and the tool line
    assert result["self_pause_count"] == 3  # yield, INTERRUPT, offline-interrupt
    assert result["self_event_count"] == 5
    assert result["events_per_hour"] == pytest.approx(2.5)
    assert result["self_pause_per_hour"] == pytest.approx(1.5)
    assert result["self_command_rate"] == pytest.approx(0.02)
    assert result["poisson_upper_95_per_hour"] >= result["events_per_hour"]
    assert [item["tool"] for item in result["self_commands"]] == ["media_stop", "timer"]


def test_a_clean_run_reports_the_bound_and_says_it_proves_nothing_forever():
    result = soak_mod.count_self_events([], [], hours=100.0, replies_spoken=500)
    assert result["self_event_count"] == 0
    assert result["events_per_hour"] == 0.0
    assert result["poisson_upper_95_per_hour"] == pytest.approx(0.03, abs=0.001)
    line = soak_mod.format_bound(result)
    assert "upper bound" in line and "0.0300" in line
    assert "not proof of zero risk" in line


def test_count_self_events_needs_a_positive_time_and_a_reply_count_is_optional():
    with pytest.raises(ValueError):
        soak_mod.count_self_events([], [], hours=0.0)
    result = soak_mod.count_self_events(ACTION_RECORDS, [], hours=1.0, replies_spoken=0)
    assert result["self_command_rate"] is None  # no denominator: null, never a fake 0


def test_a_record_with_no_words_and_no_tool_is_not_a_self_command():
    assert not soak_mod.is_self_command_record({"heard": "", "tool": ""})
    assert soak_mod.is_self_command_record({"heard": "stop", "tool": ""})
    assert soak_mod.is_self_command_record({"heard": "", "tool": "media_stop"})


# ---------------------------------------------------------------- the live loop


def write_lines(path, lines):
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def append_text(path, text):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def test_run_soak_speaks_the_plan_and_counts_what_the_logs_gain(tmp_path):
    action = tmp_path / "actions.jsonl"
    barge = tmp_path / "barge.log"
    action.write_text("")
    barge.write_text("")
    plan = soak_mod.soak_plan(0.01, seed=0, gap_seconds=0.0)
    logs = soak_mod.SoakLogs(action_log=action, barge_log=barge)
    spoken: list[str] = []
    seen: list[dict] = []

    def speak(reply):
        spoken.append(reply.text)
        if len(spoken) == 2:
            write_lines(action, [json.dumps({"time": "10:00:02", "heard": "stop", "route": "rule",
                                             "tool": "media_stop"})])
            write_lines(barge, ["10:00:02 yield generation=1 onset_audio_ms=100.0"])

    result = soak_mod.run_soak(plan, logs, hours=0.01, speak=speak, gap_seconds=0.0,
                               on_event=seen.append, clock=lambda: 0.0, sleep=lambda _s: None,
                               poll_seconds=0.0)
    assert len(spoken) == len(plan)
    assert result["self_command_count"] == 1
    assert result["self_pause_count"] == 1
    assert result["replies_spoken"] == len(plan)
    assert result["plan"]["replies"] == len(plan)
    assert [item["event"] for item in seen if item["event"] == "soak_reply"] == ["soak_reply"] * len(plan)


def test_the_logs_are_tailed_from_where_the_run_starts(tmp_path):
    action = tmp_path / "actions.jsonl"
    barge = tmp_path / "barge.log"
    write_lines(action, [json.dumps({"heard": "old", "tool": "timer"})])
    write_lines(barge, ["10:00:00 INTERRUPT"])
    logs = soak_mod.SoakLogs(action_log=action, barge_log=barge)
    logs.start()
    assert logs.poll() == []
    write_lines(action, [json.dumps({"heard": "new", "tool": "media_stop"})])
    fresh = logs.poll()
    assert [item["kind"] for item in fresh] == ["self-command"]
    assert logs.poll() == []


def test_a_half_written_line_is_not_counted_until_it_is_complete(tmp_path):
    action = tmp_path / "actions.jsonl"
    action.write_text("")
    logs = soak_mod.SoakLogs(action_log=action, barge_log=tmp_path / "barge.log")
    logs.start()
    append_text(action, json.dumps({"heard": "stop", "tool": "x"}))  # no newline yet
    assert logs.poll() == []
    append_text(action, "\n")
    assert [item["kind"] for item in logs.poll()] == ["self-command"]


def test_ctrl_c_ends_the_soak_and_the_observed_counts_survive(tmp_path):
    action = tmp_path / "actions.jsonl"
    action.write_text("")
    barge = tmp_path / "barge.log"
    barge.write_text("")
    plan = soak_mod.soak_plan(0.01, seed=0, gap_seconds=0.0)
    logs = soak_mod.SoakLogs(action_log=action, barge_log=barge)

    def speak(reply):
        write_lines(barge, ["10:00:00 INTERRUPT"])
        raise KeyboardInterrupt

    result = soak_mod.run_soak(plan, logs, hours=0.01, speak=speak, gap_seconds=0.0,
                               sleep=lambda _s: None, poll_seconds=0.0)
    assert result["interrupted"] is True
    assert result["self_pause_count"] == 1
    assert result["replies_spoken"] == 1


def test_barge_log_path_sits_beside_the_action_log(tmp_path):
    assert soak_mod.barge_log_path(tmp_path / "actions.jsonl").name == "barge.log"


def test_format_plan_names_the_families_and_the_bound():
    plan = soak_mod.soak_plan(0.5, seed=0)
    text = soak_mod.format_plan(plan, soak_mod.plan_summary(plan))
    assert "Soak plan" in text and "Trigger-word replies" in text
    assert "replies" in text


# ---------------------------------------------------------------- fixture isolation


def test_the_tag_is_the_same_words_the_product_looks_for():
    assert isolation.APPLICATION_NAME == TEST_STREAM_TAG == "voxatest"
    assert isolation.playback_properties() == reference.stream_properties()
    assert isolation.property_names() == ["application.name", "media.name", "media.role"]
    assert "application.name=voxatest" in isolation.describe()


class FakeStructure:
    def __init__(self, name):
        self.name = name
        self.values: dict[str, object] = {}

    def set_value(self, key, value):
        self.values[key] = value


class FakeGst:
    Structure = types.SimpleNamespace(new=lambda name: FakeStructure(name))


class Element:
    """A GStreamer element that only knows the properties a real player/sink knows."""

    def __init__(self, name, knows):
        self.name = name
        self.knows = set(knows)
        self.props: dict[str, object] = {}

    def find_property(self, key):
        return object() if key in self.knows else None

    def set_property(self, key, value):
        if key not in self.knows:
            raise TypeError(f"{self.name} has no property {key}")
        self.props[key] = value


def test_a_playbin_gets_the_tag_in_its_property_structure():
    playbin = Element("playbin", {"uri", "name", "description", "property-structure"})
    applied = isolation.tag_playback(playbin, FakeGst)
    assert applied["property-structure"] is True
    assert applied["name"] is True and applied["description"] is True
    structure = playbin.props["property-structure"]
    assert structure.values["application.name"] == "voxatest"
    assert structure.values["media.role"] == "voxatest"
    assert isolation.tag_isolation(applied) is True


def test_a_sink_element_gets_the_tag_in_its_properties():
    sink = Element("pulsesink", {"name", "description", "role", "properties"})
    applied = isolation.tag_playback(sink, FakeGst)
    assert applied["property-structure"] is False
    assert applied["properties"] is True
    assert sink.props["role"] == "voxatest"
    assert sink.props["properties"].values["media.name"] == "VoxaTest test voice"
    assert isolation.tag_isolation(applied) is True


def test_a_player_that_knows_nothing_still_plays_and_reports_the_tag_as_missing():
    plain = Element("alsasink", {"device"})
    applied = isolation.tag_playback(plain, FakeGst)
    assert isolation.tag_isolation(applied) is False
    assert applied["name"] is False and applied["properties"] is False


def test_no_gstreamer_module_still_tags_the_sink_directly():
    sink = Element("pulsesink", {"name", "description", "role", "application.name", "media.role", "media.name"})
    applied = isolation.tag_playback(sink, None)
    assert applied["application.name"] is True and applied["media.role"] is True
    assert isolation.tag_isolation(applied) is True


def test_the_harness_tags_its_speech_and_voxa_does_not():
    tagged = SpeechService(stream_tag=isolation.APPLICATION_NAME)
    untagged = SpeechService()
    element = Element("pulsesink", {"name", "description", "role", "properties"})
    tagged._tag_stream(element, FakeGst)
    assert element.props["properties"].values["application.name"] == "voxatest"
    element2 = Element("pulsesink", {"name", "description", "role", "properties"})
    untagged._tag_stream(element2, FakeGst)
    assert element2.props == {}  # Voxa's own playback is never tagged: it IS the reference


def test_the_harness_builds_its_speech_service_with_the_tag():
    source = voxatest_main.__file__
    from pathlib import Path

    text = Path(source).read_text(encoding="utf-8")
    assert "stream_tag=isolation.APPLICATION_NAME" in text


def test_is_test_stream_uses_the_same_rule_as_the_reference_policy():
    assert isolation.is_test_stream({"application_name": "voxatest", "media_role": ""})
    assert isolation.is_test_stream({"application_name": "Voxa", "media_role": "voxatest"})
    assert not isolation.is_test_stream({"application_name": "Voxa", "media_role": "multimedia"})


# ---------------------------------------------------------------- the reference policy

PACTL_SINK_INPUTS = """Sink Input #3213
\tDriver: protocol-native.c
\tOwner Module: 13
\tSink: 1
\tClient: 39
\tCorked: no
\tMuted: no
\tProperties:
\t\tapplication.name = "Voxa"
\t\tmedia.role = "multimedia"
\t\tmedia.name = "Audio Stream"
\t\tapplication.process.binary = "voxa"
\t\tnode.name = "gst_autosink0.stream_0"
Sink Input #3220
\tDriver: protocol-native.c
\tSink: 1
\tClient: 44
\tProperties:
\t\tapplication.name = "voxatest"
\t\tmedia.role = "voxatest"
\t\tmedia.name = "VoxaTest test voice"
"""

PACTL_NO_TEST = """Sink Input #3213
\tSink: 1
\tProperties:
\t\tapplication.name = "Voxa"
\t\tmedia.role = "multimedia"
"""

PACTL_OTHER_APP = """Sink Input #3220
\tSink: 1
\tProperties:
\t\tapplication.name = "VLC"
\t\tmedia.role = "music"
"""


def test_sink_inputs_are_parsed_by_name():
    rows = reference.parse_sink_inputs(PACTL_SINK_INPUTS)
    assert len(rows) == 2
    assert rows[0]["index"] == 3213 and rows[0]["application_name"] == "Voxa"
    assert rows[1]["application_name"] == "voxatest" and rows[1]["media_role"] == "voxatest"
    assert rows[1]["media_name"] == "VoxaTest test voice"
    assert reference.parse_sink_inputs("") == []
    assert reference.test_inputs(rows)[0]["index"] == 3220
    assert reference.own_inputs(rows)[0]["index"] == 3213


def test_the_per_stream_monitor_is_named_after_the_sink_and_the_stream():
    assert reference.per_stream_monitor("alsa_output.usb-Blue.analog-stereo", 3213) == \
        "alsa_output.usb-Blue.analog-stereo.input3213.monitor"


def test_the_own_sink_input_reference_excludes_the_test_voice():
    rows = reference.parse_sink_inputs(PACTL_SINK_INPUTS)
    sink = "alsa_output.usb-Blue.analog-stereo"
    sources = {f"{sink}.monitor", reference.per_stream_monitor(sink, 3213)}
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, sources)
    assert decision.mode == MODE_OWN_SINK_INPUT
    assert decision.source == reference.per_stream_monitor(sink, 3213)
    assert decision.trusted is True
    assert decision.warning == ""
    assert decision.test_present is True  # the test voice plays on the sink, and is not in the reference


def test_a_whole_sink_monitor_with_the_test_voice_is_refused_with_a_warning(caplog):
    rows = reference.parse_sink_inputs(
        PACTL_SINK_INPUTS.replace('application.name = "Voxa"', 'application.name = "VLC"'))
    sink = "alsa_output.usb-Blue.analog-stereo"
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, {f"{sink}.monitor"})
    assert decision.mode == MODE_SINK_MONITOR
    assert decision.trusted is False
    assert "cannot be excluded" in decision.warning
    with caplog.at_level(logging.WARNING):
        reference.log_reference(decision)
    assert any("test voice" in record.message for record in caplog.records)


def test_no_test_voice_on_the_sink_gives_a_clean_narrow_reference():
    rows = reference.parse_sink_inputs(PACTL_NO_TEST)
    sink = "alsa_output.usb-Blue.analog-stereo"
    sources = {f"{sink}.monitor", reference.per_stream_monitor(sink, 3213)}
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, sources)
    assert decision.mode == MODE_OWN_SINK_INPUT
    assert decision.test_present is False and decision.trusted is True


def test_a_foreign_app_alone_is_a_trusted_sink_monitor():
    rows = reference.parse_sink_inputs(PACTL_OTHER_APP)
    sink = "alsa_output.usb-Blue.analog-stereo"
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, {f"{sink}.monitor"})
    assert decision.mode == MODE_SINK_MONITOR
    assert decision.trusted is True and decision.test_present is False


def test_a_per_stream_monitor_the_sound_server_does_not_offer_falls_back_and_warns():
    rows = reference.parse_sink_inputs(PACTL_SINK_INPUTS)
    sink = "alsa_output.usb-Blue.analog-stereo"
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, {f"{sink}.monitor"})
    assert decision.mode == MODE_SINK_MONITOR
    assert decision.trusted is False
    assert "per-stream monitor" in " ".join(decision.notes)


def test_no_monitor_at_all_is_an_untrusted_unknown():
    decision = reference.choose_reference_source(None, [])
    assert decision.mode == MODE_UNKNOWN and decision.trusted is False


def test_decide_survives_a_sound_server_that_cannot_answer():
    def runner(argv, **_kwargs):
        raise OSError("no pactl")

    decision = reference.decide("alsa_output.real.monitor", runner=runner)
    assert decision.mode == MODE_SINK_MONITOR and decision.trusted is True


def test_decide_asks_the_sound_server_once_per_query():
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        out = PACTL_SINK_INPUTS if "sink-inputs" in argv and "short" not in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    decision = reference.decide("alsa_output.real.monitor", runner=runner, check_sources=False,
                               resolve_sinks=False)
    assert decision.mode == MODE_OWN_SINK_INPUT
    assert calls == [["pactl", "list", "sink-inputs"]]


# ------------------------------------------------- the test voice never enters the reference


def test_the_own_render_reference_never_sees_the_test_voice():
    """The rule the whole programme rests on: Voxa cancels her own render, never the test user."""
    sink = "alsa_output.usb-Blue.analog-stereo"
    rows = reference.parse_sink_inputs(PACTL_SINK_INPUTS)  # the test voice IS playing on the sink
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, {f"{sink}.monitor"},
                                                own_render=True)
    assert decision.mode == MODE_OWN_RENDER
    assert decision.trusted is True
    assert decision.warning == ""

    capture = AudioCapture()
    capture.use_own_reference()
    assert capture.reference_source is None
    voxas_samples = np.array([0.25, -0.25, 0.5, -0.5] * 40, dtype=np.float32)
    capture.feed_reference(voxas_samples)
    ring = capture.reference_ring
    assert ring.total == len(voxas_samples)  # the only writer of the reference is Voxa's own render
    assert np.allclose(ring.window(0, len(voxas_samples)), voxas_samples)
    assert np.all(ring.window(len(voxas_samples), 40) == 0.0)  # nothing else has entered it


def test_the_capture_never_opens_a_monitor_pipeline_in_own_reference_mode():
    created: list[str] = []

    class Factory:
        def make(self, name):
            created.append(name)
            return None

    gst = types.SimpleNamespace(ElementFactory=Factory())
    capture = AudioCapture()
    capture.own_reference = True
    capture._start_reference(gst)
    assert created == []  # no pulsesrc, no reference pipeline: the reference is her own render


def test_the_capture_logs_the_warning_when_the_test_voice_cannot_be_excluded(caplog):
    sink = "alsa_output.usb-Blue.analog-stereo"
    rows = reference.parse_sink_inputs(
        PACTL_SINK_INPUTS.replace('application.name = "Voxa"', 'application.name = "VLC"'))
    decision = reference.choose_reference_source(f"{sink}.monitor", rows, {f"{sink}.monitor"})

    class Element:
        def __init__(self, name):
            self.name = name

        def set_property(self, *_args):
            return None

        def find_property(self, _key):
            return object()

        def connect(self, *_args, **_kwargs):
            return True

        def link(self, _other):
            return True

    class Pipeline:
        def __init__(self, name):
            self.name = name

        def add(self, _element):
            return None

        def set_state(self, _state):
            return "ok"

    gst = types.SimpleNamespace(
        ElementFactory=types.SimpleNamespace(make=lambda name: Element(name)),
        Caps=types.SimpleNamespace(from_string=lambda _caps: None),
        Pipeline=types.SimpleNamespace(new=lambda name: Pipeline(name)),
        State=types.SimpleNamespace(PLAYING="playing"),
        SECOND=1000,
    )
    capture = AudioCapture()
    capture.reference_source = f"{sink}.monitor"
    capture.reference_decision = decision
    with caplog.at_level(logging.WARNING):
        capture._start_reference(gst)
    assert any("test voice" in record.message for record in caplog.records)


def test_the_capture_moves_to_the_narrow_source_the_policy_chose():
    sink = "alsa_output.usb-Blue.analog-stereo"
    rows = reference.parse_sink_inputs(PACTL_SINK_INPUTS)
    decision = reference.choose_reference_source(f"{sink}.monitor", rows,
                                                {f"{sink}.monitor", reference.per_stream_monitor(sink, 3213)})

    class Element:
        def __init__(self, name):
            self.name = name
            self.props: dict[str, object] = {}

        def set_property(self, key, value):
            self.props[key] = value

        def find_property(self, _key):
            return object()

        def connect(self, *_args, **_kwargs):
            return True

        def link(self, _other):
            return True

    pipelines: list[str] = []

    class Pipeline:
        def __init__(self, name):
            self.name = name
            pipelines.append(name)

        def add(self, _element):
            return None

        def set_state(self, _state):
            return "ok"

    made: list[Element] = []
    gst = types.SimpleNamespace(
        ElementFactory=types.SimpleNamespace(make=lambda name: made.append(Element(name)) or made[-1]),
        Caps=types.SimpleNamespace(from_string=lambda _caps: None),
        Pipeline=types.SimpleNamespace(new=lambda name: Pipeline(name)),
        State=types.SimpleNamespace(PLAYING="playing"),
        SECOND=1000,
    )
    capture = AudioCapture()
    capture.reference_source = f"{sink}.monitor"
    capture.reference_decision = decision
    capture._start_reference(gst)
    assert capture.reference_source == reference.per_stream_monitor(sink, 3213)
    assert made[0].props["device"] == reference.per_stream_monitor(sink, 3213)


def test_mix_chunk_cancels_voxas_own_render_and_leaves_the_test_voice_alone():
    """The canceller subtracts what is in the reference; the test voice is not in it, so it survives."""
    ring = ReferenceRing(4000)
    voxas = np.array([0.2 * np.sin(index / 9.0) for index in range(1600)], dtype=np.float32)
    ring.append(voxas)
    echo = voxas * 0.5
    test_voice = np.array([0.3] * 1600, dtype=np.float32)
    mic = echo + test_voice
    from voxa.anc import EchoCanceller

    canceller = EchoCanceller(sample_rate=16000, delay=0)
    cleaned = canceller.process(mic.copy(), ring.window(0, 1600))
    residual = cleaned - test_voice
    assert float(np.sqrt(np.mean(residual**2))) < float(np.sqrt(np.mean(mic**2)))
    assert float(np.mean(cleaned**2)) > 0.0  # the test voice is still there, not cancelled


# ---------------------------------------------------------------- voxatest doctor


def settings(**overrides) -> HarnessSettings:
    settings = HarnessSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def test_a_dry_run_asks_nothing_of_the_hardware_and_fails_nothing():
    checks = run_checks(settings(wake_word="voxa", voxa_echo_mode="voxa"), dry_run=True)
    names = [check.name for check in checks]
    assert "reference mode" in names and "mic/output device" in names
    assert [check.status for check in checks if check.name in ("reference mode", "mic/output device")] == [SKIP, SKIP]
    assert not any(check.status == FAIL for check in checks)
    assert any(check.status == PASS for check in checks)


def test_the_doctor_command_passes_a_dry_run():
    assert voxatest_main.main(["doctor", "--dry-run"]) == 0


def test_echo_off_is_a_warning_because_the_run_measures_the_raw_route():
    assert check_echo_mode_off().status == WARN


def check_echo_mode_off() -> Check:
    from voxatest.doctor import check_echo_mode

    return check_echo_mode(settings(voxa_echo_mode="off"))


def test_a_monitor_as_the_microphone_fails_the_run():
    from voxatest.doctor import check_same_device

    check = check_same_device("alsa_output.usb-Blue.monitor", "alsa_output.usb-Blue.monitor")
    assert check.status == FAIL
    assert "monitor" in check.detail
    assert check_same_device("alsa_input.usb-Blue", "alsa_output.usb-Blue.monitor").status == PASS
    assert check_same_device("", "alsa_output.usb-Blue.monitor").status == WARN


def test_a_test_voice_inside_the_reference_fails_the_doctor():
    from voxatest.doctor import check_reference_mode

    sink = "alsa_output.usb-Blue.analog-stereo"

    def runner(argv, **_kwargs):
        out = (PACTL_SINK_INPUTS.replace('application.name = "Voxa"', 'application.name = "VLC"')
               if "sink-inputs" in argv and "short" not in argv else "")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    check = check_reference_mode(f"{sink}.monitor", runner=runner)
    assert check.status == FAIL
    assert "cannot be excluded" in check.detail


def test_the_narrow_reference_passes_the_doctor():
    from voxatest.doctor import check_reference_mode

    sink = "alsa_output.usb-Blue.analog-stereo"

    def runner(argv, **_kwargs):
        if "sink-inputs" in argv and "short" not in argv:
            out = PACTL_SINK_INPUTS
        elif "short" in argv and "sources" in argv:
            out = f"1\t{sink}.monitor\n2\t{reference.per_stream_monitor(sink, 3213)}\n"
        else:
            out = ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    check = check_reference_mode(f"{sink}.monitor", runner=runner)
    assert check.status == PASS
    assert "input3213.monitor" in check.detail


def test_a_saved_hearing_profile_gives_the_drift_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from voxa.anccal import profile_path

    path = profile_path("alsa_output.usb-Blue.analog-stereo", "alsa_input.usb-Blue")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"delay": 480, "state": {"taps": [0.1] * 8}}))
    check = check_drift(settings(input_device="alsa_input.usb-Blue"), "alsa_output.usb-Blue.analog-stereo")
    assert check.status == PASS
    assert "480 samples" in check.detail


def test_no_profile_is_an_honest_warning_not_a_missing_number(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    check = check_drift(settings(input_device="alsa_input.usb-Blue"), "alsa_output.usb-Blue.analog-stereo")
    assert check.status == WARN
    assert "no saved hearing profile" in check.detail
    assert check_drift(settings(), None).status == WARN


def test_the_format_names_the_result_and_counts_the_warnings():
    checks = [Check("a", PASS, "fine"), Check("b", WARN, "read the numbers with this in mind")]
    text = format_checks(checks)
    assert "PASS a: fine" in text
    assert "RESULT: OK with 1 warning" in text
    assert "RESULT: FAIL" not in text
    assert "RESULT: FAIL" in format_checks([Check("b", FAIL, "no")])


def test_the_plan_check_reports_the_trigger_replies():
    from voxatest.doctor import check_plan

    check = check_plan(0.1, 0, 2.5, "voxa")
    assert check.status == PASS and "trigger words" in check.detail
    assert check_plan(0, 0, 2.5, "voxa").status == FAIL


# ---------------------------------------------------------------- the soak command


def test_the_soak_command_plans_without_making_a_sound(capsys):
    code = voxatest_main.main(["soak", "--hours", "0.1", "--plan"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Soak plan" in out and "upper bound" in out


def test_the_soak_command_refuses_to_speak_without_yes(capsys):
    code = voxatest_main.main(["soak", "--hours", "0.1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--yes" in out
    assert "Tell rhY" in out


def test_the_soak_command_rejects_a_bad_length(capsys):
    assert voxatest_main.main(["soak", "--hours", "0"]) == 2


def test_the_soak_report_carries_the_numbers_for_check(tmp_path):
    result = soak_mod.count_self_events(ACTION_RECORDS, [], hours=2.0, replies_spoken=10)
    result["plan"] = {"replies": 10}
    payload = {"soak": result, "comprehension": {
        key: value for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)}}
    path = tmp_path / "soak.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert voxatest_main.main(["check", str(path), "self_event_count<=5", "self_command_count<=2"]) == 0
    assert voxatest_main.main(["check", str(path), "self_event_count<=1"]) == 1
