"""V3: the voxatest hearing suite - scheduling, schema and metric math.

No microphone, no speakers, no network and no language model: the recogniser is a stub, Voxa's and
the test user's voices are a stub TTS, and the scheduling is tested on a fake clock. What runs for
real is the arithmetic the suite is built on - where the user starts, what the pipeline's events
mean, and what the seven run keys add up to.
"""

import json
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from voxatest import comprehension, hearing, offline
from voxatest.cases import TestCase
from voxatest.config import HarnessSettings
from voxatest.hearing import (
    EXPECTS,
    LiveConductor,
    build_case_clip,
    expected_request,
    format_hearing,
    is_hearing_case,
    measure_case,
    measure_live_stop_latency,
    percentile,
    plan_live,
    resolve_barge_at,
    schedule_case,
    score_hearing,
    speak_timed,
    validate_hearing,
    word_retention,
)
from voxatest.offline import run_hearing_offline_case
from voxatest.report import write_reports
from voxatest.runner import RunResult

SUITE = Path(__file__).resolve().parent.parent / "voxatest" / "data" / "test_cases_hearing.json"
SAMPLE = Path(__file__).resolve().parent.parent / "voxatest" / "data" / "samples" / "hello-sample.wav"
SR = 16000
SAMPLE_RATE = 22050


# --------------------------------------------------------------------------- stubs


def speech_samples(seconds: float) -> np.ndarray:
    """The repo's own spoken sample, resampled to 16 kHz and cut to a length: real speech, not a tone."""
    with wave.open(str(SAMPLE), "rb") as handle:
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    raw = raw.astype(np.float32) / 32768.0
    out = np.interp(np.linspace(0, len(raw) - 1, int(seconds * SR)), np.arange(len(raw)), raw)
    return out.astype(np.float32)


def fake_edge_voice(text: str, voice: str, rate: int, tmp_dir: Path):
    """Stub TTS with word timings: one 0.25 s word per word, from the repo's own spoken sample."""
    tokens = [token.strip(",.;:!?") for token in text.split() if token.strip(",.;:!?")]
    chunks, words = [], []
    position = 0.0
    for token in tokens:
        samples = speech_samples(0.25)
        chunks.append(samples)
        words.append({"word": token, "onset_s": position, "offset_s": position + 0.25})
        position += 0.25
    return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), words


def fake_natural_voice(text: str, voice: str, rate: int, tmp_dir: Path):
    samples, words = fake_edge_voice(text, voice, rate, tmp_dir)
    return samples, words, "stub-tts"


class StubASR:
    """A recogniser that answers from a script, with the attributes the pipeline calls."""

    def __init__(self, text: str = ""):
        self.fixed = text
        self.model_name = "stub-whisper"
        self.calls = 0
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        self.calls += 1
        seconds = len(pcm) / 2 / SR
        self.audio_seconds += seconds
        return {"text": self.fixed, "words": [{"word": word, "onset_s": 0.0, "offset_s": 0.2}
                                              for word in self.fixed.split()],
                "no_speech_prob": 0.01, "audio_seconds": round(seconds, 3), "asr_seconds": 0.0,
                "segments": 1 if self.fixed else 0}

    def transcribe(self, pcm: bytes, language: str = "en", hint: str = "") -> str:
        return self.fixed


@pytest.fixture
def voices(monkeypatch):
    """No network, no eSpeak, no ffmpeg anywhere in these tests."""
    monkeypatch.setattr(hearing.bench_render, "edge_voice", fake_edge_voice)
    monkeypatch.setattr(offline, "natural_voice", fake_natural_voice)
    return True


# --------------------------------------------------------------------------- schema


def case(**fields) -> TestCase:
    raw = {"id": "c", "category": "hearing-bargein", "prompt": "stop",
           "expected_contains": [], "expected_keywords": [], "rubric": "",
           "expected_tool": "", "expected_args": {}, "after": "", "verify": {},
           "capability": 0, "setup": [], "vm_only": False}
    raw.update(fields)
    return TestCase.from_dict(raw)


def test_the_new_hearing_fields_survive_a_round_trip():
    the_case = case(id="h1", prompt="stop", reply="The weather is sunny.", barge_at="middle",
                    level_db=-5.5, voice="en-US-JennyNeural", pause_at=0.4, pause_s=1.2,
                    expect="yield", cell="quiet")
    raw = the_case.to_dict()
    assert raw["reply"] == "The weather is sunny."
    assert raw["barge_at"] == "middle" and raw["level_db"] == -5.5
    assert raw["voice"] == "en-US-JennyNeural"
    assert raw["pause_at"] == 0.4 and raw["pause_s"] == 1.2
    assert raw["expect"] == "yield" and raw["cell"] == "quiet"
    again = TestCase.from_dict(raw)
    assert (again.expect, again.cell, again.level_db, again.pause_s) == ("yield", "quiet", -5.5, 1.2)


def test_a_plain_case_keeps_the_old_shape():
    the_case = case(id="plain", prompt="open gmail")
    assert the_case.expect == "" and the_case.reply == "" and the_case.barge_at is None
    assert not is_hearing_case(the_case)
    assert validate_hearing(the_case) == ""


def test_a_voxa_only_hearing_case_needs_no_prompt():
    the_case = case(id="solo", prompt="", expect="keep", reply="Say stop to end this reply.")
    assert validate_hearing(the_case) == ""
    assert is_hearing_case(the_case)


def test_an_empty_prompt_is_still_an_error_for_an_ordinary_case():
    with pytest.raises(ValueError):
        TestCase.from_dict({"prompt": "", "category": "general"})


def test_a_yield_case_must_say_something():
    assert "must say what the user says" in validate_hearing(case(prompt="", expect="yield"))


@pytest.mark.parametrize("fields,why", [
    ({"expect": "shout"}, "expect must be one of"),
    ({"expect": "yield", "level_db": 90.0}, "level_db out of"),
    ({"expect": "yield", "level_db": "loud"}, "level_db must be a number"),
    ({"expect": "keep", "pause_s": 9.0}, "pause_s out of"),
    ({"expect": "keep", "pause_s": "long"}, "pause_s must be a number"),
    ({"expect": "keep", "pause_at": 3.0}, "pause_at must be a fraction"),
    ({"expect": "keep", "pause_at": "middle"}, "pause_at must be a number"),
    ({"expect": "yield", "barge_at": "everywhere"}, "barge_at must be a number"),
    ({"expect": "yield", "barge_at": -2.0}, "barge_at cannot be negative"),
])
def test_bad_hearing_cases_are_refused(fields, why):
    assert why in validate_hearing(case(**fields))


def test_a_hearing_case_with_a_broken_number_is_refused_by_name():
    the_case = case(expect="yield", level_db="loud")
    assert "level_db must be a number" in validate_hearing(the_case)


def test_the_bench_suites_are_not_hearing_cases():
    from voxatest.offline import load_offline_cases

    for name in ("test_cases_smoke.json", "command_cases.json"):
        for the_case in load_offline_cases(SUITE.parent / name):
            assert not is_hearing_case(the_case), f"{name}:{the_case.id} must not be a hearing case"


# --------------------------------------------------------------------------- scheduling


@pytest.mark.parametrize("value,reply,seconds,name", [
    (None, 4.0, hearing.USER_ONLY_ONSET_SECONDS, "user_only"),
    ("", 4.0, hearing.USER_ONLY_ONSET_SECONDS, "user_only"),
    (0.5, 4.0, 2.0, "fraction:0.5"),
    (0.25, 8.0, 2.0, "fraction:0.25"),
    (2.5, 4.0, 2.5, "seconds:2.5"),
    ("onset", 4.0, 0.0, "onset"),
    ("first_syllable", 4.0, pytest.approx(0.24), "first_syllable"),
    ("middle", 4.0, 2.0, "middle"),
    ("last_word", 4.0, 3.4, "last_word"),
    ("after_cancel", 4.0, 4.0 + hearing.AFTER_CANCEL_SECONDS, "after_cancel"),
    ("after_reply", 4.0, 4.0 + hearing.AFTER_REPLY_GAP_SECONDS, "after_reply"),
])
def test_barge_at_is_a_fraction_below_one_and_seconds_from_one_up(value, reply, seconds, name):
    got, position = resolve_barge_at(value, reply)
    assert got == seconds and position == name


def test_a_reply_case_starts_inside_the_reply_and_a_user_only_case_breathes_first():
    over = case(prompt="stop", reply="The weather is sunny.", barge_at="middle", expect="yield")
    plan = schedule_case(over, reply_seconds=4.0, user_seconds=0.5)
    assert plan.onset_s == 2.0 and not plan.wake and plan.reply_end_s == 4.0
    assert plan.user_end_s == pytest.approx(2.5)

    solo = case(prompt="open the browser", expect="answer", barge_at=None)
    plan = schedule_case(solo, reply_seconds=0.0, user_seconds=1.0, wake_seconds=0.6)
    assert plan.wake and plan.onset_s == pytest.approx(hearing.USER_ONLY_ONSET_SECONDS
                                                       + 0.6 + hearing.WAKE_GAP_SECONDS)
    assert plan.user_end_s == pytest.approx(plan.onset_s + 1.0)


def test_a_yield_case_never_wakes_her_first():
    plan = schedule_case(case(prompt="stop", expect="yield"), reply_seconds=0.0,
                         user_seconds=0.5, wake_seconds=0.6)
    assert not plan.wake and plan.onset_s == hearing.USER_ONLY_ONSET_SECONDS


def test_the_deliberate_pause_lands_where_the_case_puts_it():
    the_case = case(prompt="open the browser and go to my music page", expect="keep",
                    pause_at=0.5, pause_s=1.2)
    plan = schedule_case(the_case, reply_seconds=0.0, user_seconds=2.0, wake_seconds=0.5)
    assert plan.pause_start_s == pytest.approx(plan.onset_s + 1.0)
    assert plan.pause_end_s == pytest.approx(plan.pause_start_s + 1.2)
    assert plan.user_end_s == pytest.approx(plan.onset_s + 2.0 + 1.2)


def test_no_pause_means_no_pause_window():
    plan = schedule_case(case(prompt="stop", expect="yield"), reply_seconds=3.0)
    assert plan.pause_start_s is None and plan.pause_end_s is None


def test_the_live_plan_makes_her_read_the_reply_and_splits_at_the_pause():
    the_case = case(prompt="open the browser and go to my music page", expect="keep",
                    pause_at=0.5, pause_s=1.0)
    plan = plan_live(the_case, reply_seconds=0.0)
    assert plan.lead_prompt == "" and plan.wake_first
    assert plan.words == "open the browser and" and plan.tail_words == "go to my music page"
    assert plan.tail_at_s == pytest.approx(plan.speak_at_s + 1.0)

    over = case(prompt="stop", reply="The weather is sunny.", barge_at="middle", expect="yield")
    plan = plan_live(over, reply_seconds=4.0)
    assert plan.lead_prompt.startswith("say ") and "weather" in plan.lead_prompt
    assert plan.speak_at_s == pytest.approx(2.0) and not plan.wake_first
    assert plan.words == "stop" and plan.tail_words == ""


def test_the_live_plan_reads_a_fraction_of_the_reply_as_seconds():
    plan = plan_live(case(prompt="stop", reply="x", barge_at=0.75, expect="yield"),
                     reply_seconds=4.0)
    assert plan.speak_at_s == pytest.approx(3.0)
    assert plan.position == "fraction:0.75"
    named = plan_live(case(prompt="stop", reply="x", barge_at="after_cancel", expect="answer"),
                      reply_seconds=2.0)
    assert named.speak_at_s == pytest.approx(2.0 + hearing.AFTER_CANCEL_SECONDS)


class FakeClock:
    """A clock the test advances: the scheduler's sleeps are bookkeeping, not wall time."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_speak_timed_waits_for_the_reply_and_then_uses_the_fake_clock():
    plan = hearing.LivePlan(lead_prompt="", speak_at_s=1.5, words="stop",
                            tail_words="and go to my music page", tail_at_s=3.0)
    clock = FakeClock(10.0)
    spoken: list[str] = []
    times = speak_timed(spoken.append, plan, reply_start_s=10.0, clock=clock, sleep=clock.sleep)
    assert spoken == ["stop", "and go to my music page"]
    assert times[0] == pytest.approx(11.5) and times[1] == pytest.approx(13.0)
    assert clock.now == pytest.approx(13.0)


def test_speak_timed_speaks_at_once_when_the_moment_has_passed():
    plan = hearing.LivePlan(lead_prompt="", speak_at_s=0.0, words="stop")
    clock = FakeClock(5.0)
    spoken: list[str] = []
    speak_timed(spoken.append, plan, reply_start_s=5.0, clock=clock, sleep=clock.sleep)
    assert spoken == ["stop"] and clock.now == 5.0


def test_the_conductor_starts_the_clock_at_the_first_sustained_loud_frame():
    plan = hearing.LivePlan(lead_prompt="", speak_at_s=1.0, words="stop")
    conductor = LiveConductor(plan=plan, clock=FakeClock(0.0))
    fired = [conductor.note(0.01, t / 50.0) for t in range(0, 100)]
    assert not any(fired), "20 ms frames of noise are not a reply"
    fired = [conductor.note(0.4, 2.0 + t / 50.0) for t in range(0, 9)]
    assert not any(fired), "160 ms of loud is not yet a reply"
    fired = [conductor.note(0.4, 2.0 + t / 50.0) for t in range(9, 14)]
    assert conductor.started_s is not None and any(fired)
    assert conductor.due == [pytest.approx(conductor.started_s + 1.0)]
    assert not conductor.missed_start


def test_the_conductor_gives_up_and_says_so_when_she_never_starts():
    plan = hearing.LivePlan(lead_prompt="", speak_at_s=0.5, words="stop")
    conductor = LiveConductor(plan=plan, clock=FakeClock(0.0), timeout_s=1.0)
    fired = [conductor.note(0.001, t / 50.0) for t in range(0, 60)]
    assert conductor.started_s is not None and conductor.missed_start
    assert any(fired) and conductor.due == [pytest.approx(conductor.started_s + 0.5)]


def test_the_conductor_stops_after_the_reply_was_found():
    plan = hearing.LivePlan(lead_prompt="", speak_at_s=0.0, words="stop")
    conductor = LiveConductor(plan=plan, clock=FakeClock(0.0))
    [conductor.note(0.4, t / 50.0) for t in range(0, 20)]
    later = [conductor.note(0.4, 1.0 + t / 50.0) for t in range(0, 20)]
    assert not any(later)


def test_live_stop_latency_is_the_quiet_after_the_user_starts():
    levels = [(index / 50.0, 0.5) for index in range(0, 100)]  # 2 s of her voice
    levels += [(2.0 + index / 50.0, 0.001) for index in range(0, 40)]  # then silence
    assert measure_live_stop_latency(levels, onset_s=1.0) == pytest.approx(1280.0)
    assert measure_live_stop_latency(levels[:50], onset_s=1.0) is None
    assert measure_live_stop_latency(levels, onset_s=5.0) is None


LOUD = [(index / 50.0, 0.4) for index in range(0, 100)]
QUIET = [(2.0 + index / 50.0, 0.001) for index in range(0, 100)]


def test_the_live_measure_reads_the_recording_as_the_offline_one_reads_the_events():
    the_case = case(id="live-1", prompt="stop", reply="The weather is sunny.",
                    barge_at="middle", expect="yield")
    block = hearing.measure_live(the_case, spoken_at_s=1.5, tail_at_s=None,
                                 levels=LOUD + QUIET, transcript="stop", wake_word="voxa",
                                 reply_seconds=2.0)
    assert block["measured"] == "live" and block["overlap"]
    assert block["yielded"] and block["stop_latency_ms"] == pytest.approx(780.0)
    assert block["request_complete"] and block["word_retention"] == 1.0
    assert block["self_interrupt_count"] == 0


def test_the_live_measure_counts_a_reply_she_stopped_herself():
    the_case = case(id="live-2", prompt="", reply="Say stop to end this reply.", expect="keep")
    cut_short = [(index / 50.0, 0.4) for index in range(0, 40)] + [(0.8 + index / 50.0, 0.001)
                                                                   for index in range(0, 60)]
    block = hearing.measure_live(the_case, spoken_at_s=None, tail_at_s=None,
                                 levels=cut_short, transcript="", wake_word="voxa",
                                 reply_seconds=2.0)
    assert block["self_interrupt_count"] == 1 and not block["kept_turn"]
    assert block["yielded"] is False and block["word_retention"] is None


def test_the_live_measure_calls_an_answer_during_the_pause_premature():
    the_case = case(id="live-3", prompt="open the browser and go to my music page", expect="keep",
                    pause_at=0.5, pause_s=1.0)
    levels = LOUD + [(2.0 + index / 50.0, 0.4) for index in range(0, 100)]
    block = hearing.measure_live(the_case, spoken_at_s=1.0, tail_at_s=3.0, levels=levels,
                                 transcript="open the browser", wake_word="voxa", reply_seconds=0.0)
    assert block["premature"] is True and block["kept_turn"] is False
    quiet_after = [(index / 50.0, 0.4 if index < 50 else 0.001) for index in range(0, 150)]
    calm = hearing.measure_live(the_case, spoken_at_s=1.0, tail_at_s=3.0, levels=quiet_after,
                                transcript="open the browser", wake_word="voxa", reply_seconds=0.0)
    assert calm["premature"] is False and calm["kept_turn"] is True


# --------------------------------------------------------------------------- metric math


@pytest.mark.parametrize("expected,got,want", [
    ("stop", "stop", 1.0),
    ("stop", "Stop.", 1.0),
    ("open the browser", "open the browser", 1.0),
    ("open the browser", "open the", 0.6667),
    ("open the browser", "", 0.0),
    ("the high is 72 degrees", "the high is seventy two degrees", 1.0),
    ("yes", "yes you", 0.0),
    ("", "anything", None),
])
def test_word_retention_counts_the_words_that_arrived(expected, got, want):
    got_value = word_retention(expected, got)
    if want is None:
        assert got_value is None
    else:
        assert got_value == pytest.approx(want, abs=1e-4)


def test_percentile_interpolates_and_reports_nothing_for_nothing():
    assert percentile([], 50) is None
    assert percentile([100.0], 95) == 100.0
    values = [100.0, 200.0, 300.0, 400.0]
    assert percentile(values, 50) == 250.0
    assert percentile(values, 0) == 100.0
    assert percentile(values, 100) == 400.0


def truth(**fields) -> dict:
    base = {"id": "c", "scenario": "voxatest-hearing", "expect": "yield",
            "overlap_position": "middle", "pair": None,
            "playback": {"text": "The weather is sunny.", "offset_s": 4.0, "onset_s": 0.0, "words": []},
            "user": {"text": "stop", "onset_s": 2.0, "offset_s": 2.5, "words": [],
                     "pause_s": None, "pause_start_s": None, "pause_end_s": None}}
    base.update(fields)
    return base


def event(name: str, t_s: float, **fields) -> dict:
    return {"event": name, "t_s": t_s, "frame": int(t_s * 50), **fields}


def test_a_stop_after_the_user_starts_is_a_yield_with_a_latency():
    the_case = case(prompt="stop", reply="The weather is sunny.", expect="yield")
    events = [event("tts-start", 0.02), event("tts-stop", 2.35, reason="barge-in-onset")]
    measures = measure_case(the_case, truth(), events, ["stop"], "voxa")
    assert measures["yielded"] is True
    assert measures["stop_latency_ms"] == pytest.approx(350.0)
    assert measures["word_retention"] == 1.0 and measures["request_complete"]
    assert measures["self_interrupt_count"] == 0 and measures["overlap"]


def test_no_stop_means_no_yield_and_no_latency():
    the_case = case(prompt="stop", reply="The weather is sunny.", expect="yield")
    measures = measure_case(the_case, truth(), [event("tts-start", 0.02)], [], "voxa")
    assert measures["yielded"] is False and measures["stop_latency_ms"] is None
    assert measures["word_retention"] == 0.0 and not measures["request_complete"]


def test_a_stop_with_no_user_speech_is_a_self_interruption():
    the_case = case(prompt="", reply="Say stop to end this reply.", expect="keep")
    solo = truth(user=None, playback={"text": "Say stop to end this reply.", "offset_s": 3.0,
                                      "onset_s": 0.0, "words": []})
    measures = measure_case(the_case, solo, [event("tts-start", 0.02),
                                             event("tts-stop", 1.5, reason="barge-in-onset")], [], "voxa")
    assert measures["self_interrupt_count"] == 1 and not measures["kept_turn"]
    assert measures["yielded"] is False and measures["word_retention"] is None


def test_a_stop_before_the_user_starts_is_her_own_voice_stopping_her():
    the_case = case(prompt="stop", reply="The weather is sunny.", expect="yield")
    measures = measure_case(the_case, truth(), [event("tts-stop", 1.0),
                                                event("tts-stop", 2.4)], ["stop"], "voxa")
    assert measures["self_interrupt_count"] == 1 and measures["yielded"] is True
    assert measures["stop_latency_ms"] == pytest.approx(400.0)


def test_a_request_inside_the_deliberate_pause_is_a_premature_answer():
    the_case = case(prompt="open the browser and go to my music page", expect="keep",
                    pause_at=0.5, pause_s=1.0)
    timed = truth(user={"text": "open the browser", "onset_s": 0.3, "offset_s": 3.0, "words": [],
                        "pause_s": 1.0, "pause_start_s": 1.3, "pause_end_s": 2.3},
                  playback=None)
    early = measure_case(the_case, timed, [event("request", 1.8, text="open the browser")],
                         ["open the browser"], "voxa")
    assert early["premature"] is True and not early["kept_turn"]
    late = measure_case(the_case, timed, [event("request", 3.4, text="open the browser")],
                        ["open the browser"], "voxa")
    assert late["premature"] is False


def test_a_reply_started_inside_the_pause_is_premature_but_its_tail_is_not():
    the_case = case(prompt="open the browser and go to my music page", expect="keep",
                    pause_at=0.5, pause_s=1.0)
    timed = truth(user={"text": "open the browser", "onset_s": 0.3, "offset_s": 3.0, "words": [],
                        "pause_s": 1.0, "pause_start_s": 1.3, "pause_end_s": 2.3},
                  playback={"text": "Opening.", "offset_s": 1.0, "onset_s": 0.0, "words": []})
    assert measure_case(the_case, timed, [event("tts-start", 1.6, text="Opening.")], [], "voxa")["premature"]
    assert not measure_case(the_case, timed, [event("tts-start", 1.1, text="Opening.")], [], "voxa")["premature"]


def test_a_backchannel_that_does_not_stop_her_is_a_kept_turn():
    the_case = case(prompt="yes", reply="The weather is sunny.", expect="keep")
    measures = measure_case(the_case, truth(user={"text": "yes", "onset_s": 2.0, "offset_s": 2.3,
                                                  "words": [], "pause_s": None,
                                                  "pause_start_s": None, "pause_end_s": None}),
                            [event("tts-start", 0.02), event("tts-end", 4.0)], [], "voxa")
    assert measures["kept_turn"] is True and measures["self_interrupt_count"] == 0
    assert measures["expect"] == "keep" and not measures["yielded"]


def test_the_expected_request_drops_the_wake_word():
    assert expected_request(case(prompt="voxa open the browser", expect="keep"), "voxa") == "open the browser"
    assert expected_request(case(prompt="stop", expect="yield"), "voxa") == "stop"


def scoring(**overrides) -> dict:
    row = {"expect": "yield", "cell": "normal", "overlap": True, "yielded": True,
           "stop_latency_ms": 200.0, "word_retention": 1.0, "request_complete": True,
           "premature": False, "self_interrupt_count": 0, "kept_turn": True, "pause_s": None}
    row.update(overrides)
    return row


def test_the_run_keys_are_the_seven_the_task_fixes():
    rows = [
        scoring(),                                                          # yielded, whole
        scoring(yielded=False, stop_latency_ms=None, word_retention=0.0, request_complete=False),
        scoring(expect="keep", yielded=False, kept_turn=True, word_retention=None),
        scoring(expect="keep", yielded=True, kept_turn=False, self_interrupt_count=2,
                word_retention=None),
        scoring(expect="answer", overlap=False, yielded=False, word_retention=0.5,
                request_complete=False),
        scoring(expect="keep", pause_s=1.0, premature=True, kept_turn=False, word_retention=None),
    ]
    block = score_hearing(rows)
    assert block["interrupt_recall"] == pytest.approx(0.5)
    assert block["keep_accuracy"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert block["stop_latency_p50_ms"] == 200.0 and block["stop_latency_p95_ms"] == 200.0
    assert block["premature_rate"] == pytest.approx(1.0)
    assert block["self_interrupt_count"] == 2
    assert block["request_word_retention"] == pytest.approx((1.0 + 0.0 + 0.5) / 3.0, abs=1e-4)
    assert block["cases"] == 6 and block["yield_cases"] == 2 and block["pause_cases"] == 1
    assert set(hearing.KEYS) <= set(block)


def test_a_run_with_no_hearing_case_reports_nothing():
    assert score_hearing([]) == {}
    assert score_hearing([{"expect": ""}]) == {}


def test_every_key_comes_with_its_counts_and_its_cells():
    rows = [scoring(cell="quiet"), scoring(cell="normal"),
            scoring(expect="keep", cell="quiet", kept_turn=False, yielded=True, word_retention=None)]
    block = score_hearing(rows)
    assert block["by_cell"]["quiet"]["cases"] == 2
    assert block["by_cell"]["normal"]["cases"] == 1
    assert block["by_overlap"]["overlap"]["cases"] == 3
    assert block["by_expect"]["keep"]["keep_accuracy"] == 0.0
    assert block["by_cell"]["quiet"]["interrupt_recall"] == pytest.approx(1.0)


def test_a_missing_family_is_null_not_zero():
    block = score_hearing([scoring(expect="answer", overlap=False, yielded=False,
                                   stop_latency_ms=None)])
    assert block["interrupt_recall"] is None and block["keep_accuracy"] is None
    assert block["premature_rate"] is None and block["stop_latency_p50_ms"] is None
    assert block["self_interrupt_count"] == 0 and block["request_word_retention"] == 1.0


def test_the_text_report_prints_every_key():
    block = score_hearing([scoring(), scoring(expect="keep", kept_turn=True, word_retention=None)])
    lines = format_hearing(block)
    joined = "\n".join(lines)
    for key in hearing.KEYS:
        assert key in joined
    assert lines[0].startswith("Hearing suite")
    assert format_hearing({}) == []


# --------------------------------------------------------------------------- the suite file


@pytest.fixture(scope="module")
def suite_cases() -> list[TestCase]:
    from voxatest.cases import load_cases

    return load_cases(SUITE)


def test_the_hearing_suite_has_at_least_sixty_cases(suite_cases):
    assert len(suite_cases) >= 60
    assert all(is_hearing_case(the_case) for the_case in suite_cases)
    assert not [the_case.id for the_case in suite_cases if validate_hearing(the_case)]


def test_the_suite_covers_every_cell_the_matrix_names(suite_cases):
    categories = {the_case.category for the_case in suite_cases}
    for family in ("hearing-bargein", "hearing-answer", "hearing-repeat", "hearing-pause",
                   "hearing-continuation", "hearing-quiet", "hearing-backchannel",
                   "hearing-self", "hearing-selfword", "hearing-tail"):
        assert family in categories, f"the suite is missing {family}"
    positions = {the_case.barge_at for the_case in suite_cases}
    for position in ("onset", "first_syllable", "middle", "last_word", "after_cancel"):
        assert position in positions, f"no case interrupts at {position}"
    answers = {the_case.prompt for the_case in suite_cases if the_case.expect == "answer"}
    for words in ("yes", "no", "that one", "no, the other one"):
        assert words in answers, f"no short answer {words!r}"
    repeats = [the_case for the_case in suite_cases
               if the_case.category == "hearing-repeat"
               and the_case.prompt.strip(". ") == the_case.reply.strip(". ")]
    assert repeats, "nobody repeats Voxa's exact words"
    stop_words = [the_case for the_case in suite_cases
                  if the_case.category == "hearing-selfword" and "stop" in the_case.prompt
                  and "stop" in the_case.reply]
    assert stop_words, "no case says stop while she says stop"
    pauses = [the_case for the_case in suite_cases if the_case.pause_s is not None]
    assert pauses and all(0.5 <= float(the_case.pause_s) <= 1.5 for the_case in pauses)
    assert any(the_case.level_db is not None and float(the_case.level_db) <= -10.0
               for the_case in suite_cases), "no quiet user"
    solos = [the_case for the_case in suite_cases if not the_case.prompt.strip()]
    assert solos, "no Voxa-only case"
    assert {the_case.expect for the_case in suite_cases} == set(EXPECTS)


def test_the_suite_declares_its_cells(suite_cases):
    cells = {the_case.cell for the_case in suite_cases}
    assert "normal" in cells and "quiet" in cells
    quiet = [the_case for the_case in suite_cases if the_case.cell == "quiet"]
    assert all(float(the_case.level_db) < 0.0 for the_case in quiet if the_case.level_db is not None)


# --------------------------------------------------------------------------- the run


def test_one_hearing_case_offline_reports_the_five_measures(voices, tmp_path):
    the_case = case(id="h-one", prompt="stop", reply="The weather is sunny.",
                    barge_at="middle", expect="yield", level_db=5.0)
    measures = hearing.run_hearing_case(the_case, HarnessSettings(), StubASR(text="stop"), None,
                                        0, tmp_path, echo_mode="off", wake_word="voxa",
                                        log=lambda _line: None)
    assert measures["expect"] == "yield" and measures["echo_mode"] == "off"
    assert measures["user_onset_s"] == pytest.approx(0.5 * measures["clip"]["reply_seconds"], abs=0.2)
    assert measures["expected_request"] == "stop"
    assert set(hearing.KEYS) != set(measures)  # the per-case block is not the run block
    for key in ("yielded", "stop_latency_ms", "request_complete", "premature", "self_interrupt_count"):
        assert key in measures


def test_the_clip_is_a_bench_shaped_truth_with_the_user_inside_the_reply(voices, tmp_path):
    the_case = case(id="h-clip", prompt="stop", reply="The weather is sunny today.",
                    barge_at="middle", expect="yield", level_db=5.0)
    clip = build_case_clip(the_case, tmp_path, 3)
    truth_file = clip.truth
    assert truth_file["scenario"] == "voxatest-hearing"
    assert truth_file["playback"]["offset_s"] > 0.0
    assert 0.0 < truth_file["user"]["onset_s"] < truth_file["playback"]["offset_s"]
    assert clip.mic.shape == clip.ref.shape
    assert clip.mic.dtype == np.float32 and clip.ref.dtype == np.float32
    assert np.max(np.abs(clip.mic)) <= 1.0
    assert truth_file["echo"]["delay_ms"] > 0.0


def test_the_same_seed_builds_the_same_clip(voices, tmp_path):
    the_case = case(id="h-seed", prompt="stop", reply="The weather is sunny today.",
                    barge_at="middle", expect="yield", level_db=-5.0)
    first = build_case_clip(the_case, tmp_path, 11)
    second = build_case_clip(the_case, tmp_path, 11)
    assert np.array_equal(first.mic, second.mic)
    assert np.array_equal(first.ref, second.ref)
    assert first.truth == second.truth
    other = build_case_clip(the_case, tmp_path, 12)
    assert not np.array_equal(first.mic, other.mic), "the seed must decide the room"


def test_the_offline_run_grades_a_hearing_case_and_carries_its_measures(voices, tmp_path, monkeypatch):
    the_case = case(id="h-run", prompt="open the browser", reply="The weather is sunny today.",
                    barge_at="after_reply", expect="answer", level_db=5.0)
    asr = StubASR(text="open the browser")
    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": asr)
    results, block = offline.run_offline([the_case], HarnessSettings(),
                                          offline.parse_variants("clean"), 0, tmp_path,
                                          asr=asr, registry={})
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RunResult)
    assert result.hearing and result.hearing["expect"] == "answer"
    assert result.heard == result.hearing["request_text"]
    assert result.grade.method == "hearing"
    assert block["hearing_cases"] == 1
    assert block["hearing"]["cases"] == 1
    assert set(hearing.KEYS) <= set(block["hearing"])


class ScriptedRunner:
    """A pipeline that replays a script of events: the integration test's stand-in for ClipRunner."""

    events: list[dict] = []

    def __init__(self, asr=None, echo_mode="", wake_word="", settings=None):
        ScriptedRunner.last = self

    def run(self, mic, ref, truth):
        self.events = list(SCRIPTED_EVENTS)
        return {"requests": ["open the browser"], "turns": [{"text": "open the browser",
                                                              "kind": "armed", "frame": 10}],
                "counts": {"suspect": 1, "barge_in": 0, "own_voice": 0, "ignored": 0,
                           "exit": 0, "turns": 1, "hold": 0},
                "transcript_texts": ["open the browser"], "seconds": 4.0}


SCRIPTED_EVENTS = [
    {"event": "tts-start", "t_s": 0.02, "frame": 1},
    {"event": "tts-end", "t_s": 1.42, "frame": 71},
    {"event": "request", "t_s": 2.6, "frame": 130, "text": "open the browser"},
]


class RecordingSpeech:
    """Speakers that remember every breath the harness spoke, in order."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, rate, voice, *, on_started, on_done, on_error):
        self.spoken.append(text)
        on_started()
        on_done()

    def stop(self):
        return None


class LevelCapture:
    """A microphone that replays a level script: her voice, then the room going quiet."""

    def __init__(self, loud_frames: int = 30, quiet_frames: int = 120):
        self.loud_frames = loud_frames
        self.quiet_frames = quiet_frames
        self.started = 0
        self.stopped = 0
        self.on_audio: Callable[[bytes, float], None] | None = None

    def start(self, device_id, on_audio, on_level, on_error):
        self.started += 1
        self.on_audio = on_audio
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self):
        for _ in range(self.loud_frames):
            time.sleep(0.02)
            self.on_audio(b"\0\0" * 160, 0.4)
        for _ in range(self.quiet_frames):
            time.sleep(0.02)
            self.on_audio(b"\0\0" * 160, 0.001)

    def stop(self):
        self.stopped += 1


def test_the_live_path_speaks_the_barge_in_when_her_reply_starts(monkeypatch):
    import queue as queue_module

    from voxatest.runner import run_case

    def drain_stream(audio_queue, stop_event, **_kwargs):
        """The listening phase of the real pipeline: keep pulling frames until the case is over."""
        while not stop_event.is_set():
            try:
                if audio_queue.get(timeout=0.1) is None:
                    break
            except queue_module.Empty:
                continue
        return iter([])

    monkeypatch.setattr("voxatest.runner.segment_stream", drain_stream)
    the_case = case(id="live-run", prompt="stop", reply="The weather is sunny today.",
                    barge_at="middle", expect="yield", level_db=5.0)
    settings = HarnessSettings(wake_word="voxa", settle_seconds=0.0, speak_timeout_seconds=1.0,
                               reply_wait_seconds=3.0, end_silence_seconds=0.2, max_reply_seconds=3.0)
    speech = RecordingSpeech()
    capture = LevelCapture()
    result = run_case(the_case, settings, None, None, capture, speech, log=lambda _line: None)
    assert result.transcript == ""
    assert any("weather" in text for text in speech.spoken), speech.spoken
    assert any("stop" in text for text in speech.spoken), speech.spoken
    assert result.hearing is not None and result.hearing["expect"] == "yield"
    assert result.hearing["measured"] == "live"
    assert result.hearing["user_onset_s"] is not None
    assert capture.started == 1 and capture.stopped == 1


def test_a_delivered_request_is_a_complete_answer(voices, tmp_path, monkeypatch):
    the_case = case(id="h-ok", prompt="open the browser", reply="The weather is sunny today.",
                    barge_at="after_reply", expect="answer", level_db=5.0)
    monkeypatch.setattr(hearing, "ClipRunner", ScriptedRunner)
    result = run_hearing_offline_case(the_case, HarnessSettings(), StubASR(text="open the browser"),
                                      {}, offline.Variant("clean"), 0, tmp_path, echo_mode="off",
                                      wake_word="voxa", log=lambda _line: None)
    assert result.hearing["request_text"] == "open the browser"
    assert result.hearing["word_retention"] == 1.0 and result.hearing["request_complete"]
    assert result.grade.passed and result.grade.method == "hearing"
    assert result.heard == "open the browser"


def test_a_hearing_case_the_pipeline_never_answered_fails_the_case(voices, tmp_path):
    the_case = case(id="h-fail", prompt="open the browser", reply="The weather is sunny today.",
                    barge_at="middle", expect="answer", level_db=5.0)
    result = run_hearing_offline_case(the_case, HarnessSettings(), StubASR(text=""), None,
                                      offline.Variant("clean"), 0, tmp_path, echo_mode="off",
                                      wake_word="voxa", log=lambda _line: None)
    assert not result.grade.passed and "incomplete" in result.grade.notes


def test_the_grade_wants_the_decision_the_case_asked_for(voices, tmp_path):
    the_case = case(id="h-grade", prompt="stop", reply="The weather is sunny today.",
                    barge_at="middle", expect="yield", level_db=5.0)
    measures = {"expect": "yield", "yielded": True, "stop_latency_ms": 250.0,
                "word_retention": 1.0, "request_complete": True, "kept_turn": True,
                "self_interrupt_count": 0, "cell": "normal", "echo_mode": "off",
                "user_onset_s": 1.0, "clip": {"level_db": 5.0}}
    assert offline.hearing_grade(the_case, measures).passed
    measures["yielded"] = False
    grade = offline.hearing_grade(the_case, measures)
    assert not grade.passed and "did not stop" in grade.notes


def test_the_comprehension_block_carries_the_hearing_keys_for_check():
    rows = [{"case_id": "a", "category": "hearing-bargein", "prompt": "stop", "heard": "stop",
             "passed": True, "expected_tool": "", "hearing": scoring()},
            {"case_id": "b", "category": "hearing-answer", "prompt": "yes", "heard": "yes",
             "passed": True, "expected_tool": "",
             "hearing": scoring(expect="answer", overlap=False, yielded=False,
                                stop_latency_ms=None)}]
    block = comprehension.score_run(rows, "voxa")
    for key in hearing.KEYS:
        assert key in block, f"{key} is not in the comprehension block"
    report = {"comprehension": block}
    value, missing = comprehension._resolve(report, "request_word_retention")
    assert missing == "" and value == pytest.approx(1.0)
    value, missing = comprehension._resolve(report, "self_interrupt_count")
    assert missing == "" and value == 0


def test_a_run_without_hearing_cases_keeps_the_comprehension_block_as_it_was(voices):
    rows = [{"case_id": "a", "category": "general", "prompt": "open gmail", "heard": "open gmail",
             "passed": True, "expected_tool": ""}]
    block = comprehension.score_run(rows, "voxa")
    assert not (set(hearing.KEYS) & set(block))
    assert block["hearing_word_accuracy"] == 1.0


def test_the_report_files_carry_the_hearing_section(voices, tmp_path):
    the_case = case(id="h-rep", prompt="stop", reply="The weather is sunny today.",
                    barge_at="middle", expect="yield", level_db=5.0)
    asr = StubASR(text="stop")
    result = run_hearing_offline_case(the_case, HarnessSettings(), asr, {}, offline.Variant("clean"),
                                      0, tmp_path, echo_mode="off", wake_word="voxa",
                                      log=lambda _line: None)
    paths = write_reports([result], tmp_path / "reports",
                          json_path=tmp_path / "run.json", text_path=tmp_path / "run.txt",
                          failure_path=tmp_path / "fail.md")
    payload = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
    assert set(hearing.KEYS) <= set(payload["comprehension"])
    assert payload["results"][0]["hearing"]["expect"] == "yield"
    text = Path(paths["text_path"]).read_text(encoding="utf-8")
    assert "Hearing suite" in text and "request_word_retention" in text
