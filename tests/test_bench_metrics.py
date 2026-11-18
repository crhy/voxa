"""Tests for the B3 metrics and the gate report (voxa/bench/metrics.py, report.py, gates.toml).

Everything is hand-built: one truth dict and one list of events per toy clip. No WAV, no faster-whisper,
no sound device, no network, no sleeping. Each case states the exact number the metric must produce,
which is roadmap B3's gate ("exact values on hand-built toy cases"). The cases include the ones the
roadmap names - a deleted first word, an echo word inside the TTS window, a trigger with no onset, a
back-channel that resumes, a premature cut-off, a duplicated frame, a lost frame, a matched pair
decided wrong - and the rest of the metric set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxa.bench.metrics import METRIC_NAMES, align, clip_facts, compute, edit_distance, normalise, number_to_words, wer

# --------------------------------------------------------------------------- toy clips

def frame(index: int, erle_db: float = 0.0, mic_level_db: float = -40.0, clean_level_db: float = -40.0,
          ref_power: float = 0.0, muted: bool = False) -> dict:
    """One 20 ms frame of the runner's event stream."""
    return {
        "event": "frame",
        "frame": index,
        "samples": [index * 320, (index + 1) * 320],
        "t_s": round((index + 1) * 0.02, 3),
        "erle_db": erle_db,
        "mic_level_db": mic_level_db,
        "clean_level_db": clean_level_db,
        "ref_power": ref_power,
        "muted": muted,
        "speaking": ref_power > 0.0,
    }


def frames(count: int, **fields) -> list[dict]:
    return [frame(index, **fields) for index in range(count)]


def timed(text: str, onset: float, offset: float) -> list[dict]:
    """A word list with on/offsets, the shape truth.json and the transcript events use."""
    return [{"word": word, "onset_s": onset, "offset_s": offset} for word in text.split()]


def truth(cid: str = "toy-0000-x-00", scenario: str = "fst", expect: str = "none",
          playback: dict | None = None, user: dict | None = None, echo: dict | None = None,
          pair: dict | None = None, position: str | None = None) -> dict:
    return {"id": cid, "scenario": scenario, "expect": expect, "overlap_position": position,
            "playback": playback, "user": user, "echo": echo, "pair": pair}


def playing(text: str, onset: float = 0.0, offset: float = 3.6) -> dict:
    return {"text": text, "onset_s": onset, "offset_s": offset, "words": timed(text, onset, offset)}


def speaking(text: str, onset: float = 1.0, offset: float = 2.0) -> dict:
    return {"text": text, "onset_s": onset, "offset_s": offset, "words": timed(text, onset, offset)}


def clip(clip_truth: dict, events: list[dict], summary: dict | None = None) -> dict:
    return clip_facts(clip_truth, events, summary or {})


def value_of(facts: list[dict], name: str) -> float | None:
    return compute(facts, seed=0)[name]["value"]


def row_of(facts: list[dict], name: str) -> dict:
    return compute(facts, seed=0)[name]


# --------------------------------------------------------------------------- text and alignment

def test_text_is_normalised_and_numbers_become_words() -> None:
    assert normalise("Volume Down!") == "volume down"
    assert normalise("The library opens at 9 in the morning.") == "the library opens at nine in the morning"
    assert normalise("a high of 72 degrees") == "a high of seventy two degrees"
    assert normalise("don\u2019t stop \u2014 it's 105!") == "dont stop its one hundred five"
    assert number_to_words("0") == "zero"
    assert number_to_words("13") == "thirteen"
    assert number_to_words("72") == "seventy two"
    assert number_to_words("105") == "one hundred five"
    assert number_to_words("1999") == "one nine nine nine"
    assert number_to_words("3.5") == "three point five"


def test_word_level_levenshtein_and_alignment() -> None:
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["open", "the", "map"], ["the", "map"]) == 1
    assert edit_distance(["yes"], ["blue"]) == 1
    assert edit_distance(["the", "library", "opens"], ["the", "library", "opens", "at", "nine"]) == 2

    assert align(["a", "b", "c"], ["a", "c"])[0] == ["match", "del", "match"]
    assert align(["a", "b", "c"], ["a", "x", "c"])[0] == ["match", "sub", "match"]
    assert align(["a"], ["a", "b", "c"]) == (["match"], 2)
    assert wer(["a", "b", "c", "d"], ["a", "x", "d"]) == pytest.approx(0.5)  # 1 sub + 1 del / 4
    assert wer([], ["a"]) is None


# --------------------------------------------------------------------------- the named toy cases

def test_a_deleted_first_word_counts_once_everywhere_it_should() -> None:
    """Roadmap case 1: the user's first word never arrives."""
    facts = [clip(
        truth(scenario="cold", expect="keep", position="cold", user=speaking("open the map", 1.0, 2.0)),
        frames(120) + [
            {"event": "pause", "t_s": 2.4, "segment_seconds": 1.4},
            {"event": "transcript", "t_s": 2.4, "text": "the map", "words": timed("the map", 0.0, 1.0)},
            {"event": "endpoint", "t_s": 2.4, "text": "the map", "complete": True},
            {"event": "request", "t_s": 2.4, "text": "the map", "kind": "wake"},
        ])]
    assert value_of(facts, "first_word_trunc_cold") == 1.0
    assert value_of(facts, "user_word_deletion_rate") == pytest.approx(1.0 / 3.0)
    assert value_of(facts, "first_word_trunc_bargein") is None  # no barge-in in this clip


def test_an_echo_word_inside_the_tts_window_is_a_leak_and_one_outside_is_not() -> None:
    """Roadmap case 2: one played TTS word comes back inside +/- 1.5 s, the other does not."""
    facts = [clip(
        truth(scenario="fst", playback=playing("the library", 0.0, 3.6)),
        frames(180) + [
            {"event": "clip-transcript", "t_s": 3.6, "text": "library",
             "words": [{"word": "library", "onset_s": 2.9, "offset_s": 3.2},
                       {"word": "the", "onset_s": 6.0, "offset_s": 6.2}]},
        ])]
    # "library" ends at 1.0 s in the script; 2.9 s is inside 1.0 + 1.5. "the" at 6.0 s is not.
    assert value_of(facts, "echo_leak_word_rate") == 0.5
    assert value_of(facts, "fst_transcript_rate") == 0.0  # the product path heard nothing


def test_a_trigger_with_no_user_onset_is_a_false_bargein_per_hour() -> None:
    """Roadmap case 3: Voxa's own playback opens the level gate and the words decide an interrupt."""
    facts = [clip(
        truth(scenario="fst", playback=playing("the weather is clear", 0.0, 3.6)),
        frames(180) + [{"event": "barge-in-confirm", "frame": 50, "t_s": 1.0, "heard": "the weather"},
                       {"event": "tts-stop", "frame": 50, "t_s": 1.0, "heard": "the weather"}])]
    assert value_of(facts, "false_bargein_per_hour") == pytest.approx(1.0 / (3.6 / 3600.0))
    assert value_of(facts, "interrupt_stop_rate") == 1.0
    assert value_of(facts, "self_pause_per_hour") == pytest.approx(1.0 / (3.6 / 3600.0))
    assert value_of(facts, "self_command_rate") == 0.0  # no request reached the assistant


def test_a_backchannel_keeps_playing_and_a_stop_breaks_the_resume_rate() -> None:
    """Roadmap case 4: "yes" over a reply must not take the floor."""
    quiet = clip(
        truth(scenario="short", expect="backchannel", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("yes", 1.0, 1.3)),
        frames(180, ref_power=0.001))
    assert value_of([quiet], "backchannel_resume_rate") == 1.0

    broken = clip(
        truth(scenario="short", expect="backchannel", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("yes", 1.0, 1.3)),
        frames(180, ref_power=0.001) + [{"event": "tts-stop", "frame": 70, "t_s": 1.4, "heard": "yes"}])
    assert value_of([broken], "backchannel_resume_rate") == 0.0


def test_a_premature_cut_off_is_measured_as_a_negative_endpoint_latency() -> None:
    """Roadmap case 5: the endpoint fires before the user's last word ends."""
    facts = [clip(
        truth(scenario="nst", expect="keep", user=speaking("what time is it", 1.0, 2.0)),
        frames(120) + [{"event": "endpoint", "t_s": 1.5, "text": "what time", "complete": True}])]
    assert value_of(facts, "premature_cutoff_rate") == 1.0
    assert value_of(facts, "ep50_ms") == pytest.approx(-500.0)
    assert value_of(facts, "ep90_ms") == pytest.approx(-500.0)

    late = [clip(
        truth(scenario="nst", expect="keep", user=speaking("what time is it", 1.0, 2.0)),
        frames(150) + [{"event": "endpoint", "t_s": 2.4, "text": "what time is it", "complete": True}])]
    assert value_of(late, "premature_cutoff_rate") == 0.0
    assert value_of(late, "ep50_ms") == pytest.approx(400.0)


def test_a_duplicated_frame_identifier_breaks_turn_retention() -> None:
    """Roadmap case 6: the accepted span counts one frame twice."""
    events = frames(60)
    events[30]["frame"] = 29  # frame 29 appears twice, and 30 is gone
    facts = [clip(
        truth(scenario="dt_middle", expect="keep", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("stop the music", 0.7, 1.6)),
        events + [{"event": "suspect", "frame": 28, "t_s": 0.58},
                  {"event": "barge-in-confirm", "frame": 40, "t_s": 0.82, "heard": "stop"},
                  {"event": "request", "t_s": 0.9, "text": "stop", "kind": "wake"}])]
    assert value_of(facts, "turn_retention_rate") == 0.0


def test_a_lost_frame_identifier_breaks_turn_retention() -> None:
    """Roadmap case 7: a frame of the accepted span never reached the pipeline."""
    events = frames(60)
    del events[35]  # frame 35 is missing from the stream
    facts = [clip(
        truth(scenario="dt_middle", expect="keep", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("stop the music", 0.7, 1.6)),
        events + [{"event": "suspect", "frame": 34, "t_s": 0.7},
                  {"event": "barge-in-confirm", "frame": 40, "t_s": 0.82, "heard": "stop"},
                  {"event": "request", "t_s": 0.9, "text": "stop", "kind": "wake"}])]
    assert value_of(facts, "turn_retention_rate") == 0.0


def test_a_matched_pair_decided_wrong_on_one_side_fails_the_pair() -> None:
    """Roadmap case 8: the same words said by Voxa alone and by the user over her."""
    voxa_side = clip(
        truth(scenario="pair", expect="none", pair={"id": "stop", "side": "voxa"},
              playback=playing("stop", 0.0, 0.9)),
        frames(60, ref_power=0.001) + [{"event": "request", "t_s": 1.2, "text": "stop", "kind": "armed"}])
    user_side = clip(
        truth(scenario="pair", expect="yield", pair={"id": "stop", "side": "user"},
              playback=playing("the weather is clear", 0.0, 3.6), user=speaking("stop", 1.0, 1.3)),
        frames(180, ref_power=0.001) + [{"event": "tts-stop", "frame": 60, "t_s": 1.4, "heard": "stop"},
                                        {"event": "request", "t_s": 1.5, "text": "stop", "kind": "wake"}])
    assert value_of([voxa_side, user_side], "pair_accuracy") == 0.0
    assert value_of([user_side], "interrupt_stop_rate") is None  # only Voxa-only clips count there

    clean_voxa_side = clip(
        truth(scenario="pair", expect="none", pair={"id": "stop", "side": "voxa"},
              playback=playing("stop", 0.0, 0.9)),
        frames(60, ref_power=0.001))
    assert value_of([clean_voxa_side, user_side], "pair_accuracy") == 1.0


# --------------------------------------------------------------------------- the rest of the set

def test_turn_retention_needs_the_words_promoted_exactly_once() -> None:
    events = frames(60)
    facts = [clip(
        truth(scenario="dt_middle", expect="keep", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("stop the music", 0.7, 1.6)),
        events + [{"event": "suspect", "frame": 34, "t_s": 0.7},
                  {"event": "barge-in-confirm", "frame": 40, "t_s": 0.82, "heard": "stop"},
                  {"event": "request", "t_s": 0.9, "text": "stop the music", "kind": "wake"}])]
    assert value_of(facts, "turn_retention_rate") == 1.0

    twice = clip(
        truth(scenario="dt_middle", expect="keep", playback=playing("the weather is clear", 0.0, 3.6),
              user=speaking("stop the music", 0.7, 1.6)),
        events + [{"event": "suspect", "frame": 34, "t_s": 0.7},
                  {"event": "barge-in-confirm", "frame": 40, "t_s": 0.82, "heard": "stop"},
                  {"event": "request", "t_s": 0.9, "text": "stop", "kind": "wake"},
                  {"event": "request", "t_s": 1.0, "text": "stop", "kind": "wake"}])
    assert value_of([twice], "turn_retention_rate") == 0.0


def test_a_missed_attempt_and_a_duck_are_both_measured() -> None:
    """An attempt needs >= 2 words over >= 0.5 s; a duck within 1.0 s of the onset is not missed."""
    attempt = truth(scenario="dt_middle", expect="keep",
                    playback=playing("the weather is clear", 0.0, 3.6),
                    user=speaking("stop the music", 1.0, 1.9))
    late = clip(attempt, frames(180, ref_power=0.001) + [{"event": "tts-stop", "frame": 90, "t_s": 1.35,
                "heard": "stop the"}])
    assert value_of([late], "missed_bargein_rate") == 0.0
    assert value_of([late], "duck_latency_p50_ms") == pytest.approx(350.0)
    assert value_of([late], "duck_latency_p90_ms") == pytest.approx(350.0)
    assert value_of([late], "stop_latency_p95_ms") == pytest.approx(350.0)

    never = clip(attempt, frames(180, ref_power=0.001))
    assert value_of([never], "missed_bargein_rate") == 1.0
    assert value_of([never], "duck_latency_p50_ms") is None

    too_short = clip(truth(scenario="short", expect="yield",
                          playback=playing("the weather is clear", 0.0, 3.6),
                          user=speaking("stop", 1.0, 1.2)),
                     frames(180, ref_power=0.001))
    assert value_of([too_short], "missed_bargein_rate") is None  # one word, 0.2 s: not an attempt


def test_the_delay_the_canceller_estimates_is_compared_with_the_delay_injected() -> None:
    facts = [clip(
        truth(scenario="fst", echo={"delay_samples": 1600, "drift_ppm": 0.0}),
        frames(200) + [{"event": "delay-estimate", "frame": 150, "t_s": 3.0, "delay_samples": 1616,
                        "delay_ms": 101.0}])]
    assert value_of(facts, "delay_est_err_ms") == pytest.approx(1.0)  # 16 samples at 16 kHz


def test_drift_loses_erle_against_a_steady_echo_path() -> None:
    drifting = clip(truth(scenario="fst", echo={"drift_ppm": -150.0}, playback=playing("x y", 0.0, 6.0)),
                   [frame(index, erle_db=10.0, ref_power=0.001) for index in range(300)])
    steady = clip(truth(scenario="fst", echo={"drift_ppm": 0.0}, playback=playing("x y", 0.0, 6.0)),
                  [frame(index, erle_db=20.0, ref_power=0.001) for index in range(300)])
    assert value_of([drifting, steady], "drift_erle_loss_db") == pytest.approx(10.0)
    assert value_of([drifting], "drift_erle_loss_db") is None  # no steady clip: nothing to compare
    mild = clip(truth(scenario="fst", echo={"drift_ppm": -60.0}, playback=playing("x y", 0.0, 6.0)),
                [frame(index, erle_db=17.0, ref_power=0.001) for index in range(300)])
    assert value_of([drifting, steady, mild], "drift_erle_loss_db") == pytest.approx(8.5)  # mild = steady


def test_reconverge_measures_the_time_back_to_15_db_after_the_delay_re_estimate() -> None:
    events = [frame(index, erle_db=20.0 if index >= 212 else 5.0, ref_power=0.001) for index in range(300)]
    facts = [clip(truth(scenario="fst", echo={"delay_samples": 1600}),
                  events + [{"event": "delay-estimate", "frame": 150, "t_s": 3.0, "delay_samples": 1600}])]
    assert value_of(facts, "reconverge_s") == pytest.approx(4.26 - 3.0)


def test_near_end_attenuation_is_the_pipeline_applied_to_the_user_during_double_talk() -> None:
    dt = clip(truth(scenario="dt_middle", expect="keep",
                    playback=playing("the weather is clear", 0.0, 3.6),
                    user=speaking("stop the music", 1.0, 1.9)),
              [frame(index, mic_level_db=-30.0, clean_level_db=-34.0, ref_power=0.001)
               for index in range(50, 95)])
    nst = clip(truth(scenario="nst", expect="keep", user=speaking("stop the music", 1.0, 1.9)),
               [frame(index, mic_level_db=-30.0, clean_level_db=-30.0, ref_power=0.0)
                for index in range(50, 95)])
    assert value_of([dt, nst], "nearend_atten_dt_db") == pytest.approx(4.0)
    assert value_of([nst], "nearend_atten_dt_db") is None  # no double-talk clip to compare


def test_erle_is_scored_on_far_end_frames_after_two_seconds_only() -> None:
    facts = [clip(truth(scenario="fst", playback=playing("the weather is clear", 0.0, 6.0)),
              [frame(index, erle_db=1.0 if index < 100 else 25.0, ref_power=0.001) for index in range(300)]
              + [frame(index, erle_db=99.0, ref_power=0.0) for index in range(300, 320)])]
    assert value_of(facts, "erle_fst_median_db") == 25.0  # the first 2 s and the silent tail are out
    assert value_of(facts, "erle_fst_p10_db") == 25.0


def test_protected_words_short_answers_and_interrupt_commands_must_survive() -> None:
    kept = clip(truth(scenario="short", expect="yield",
                      playback=playing("the weather is clear", 0.0, 3.6), user=speaking("stop", 1.0, 1.2)),
                frames(180, ref_power=0.001) +
                [{"event": "pause", "t_s": 1.4, "segment_seconds": 0.4},
                 {"event": "transcript", "t_s": 1.4, "text": "stop", "words": timed("stop", 0.0, 0.2)}])
    assert value_of([kept], "protected_word_keep_rate") == 1.0

    lost = clip(truth(scenario="short", expect="yield",
                      playback=playing("the weather is clear", 0.0, 3.6), user=speaking("stop", 1.0, 1.2)),
                frames(180, ref_power=0.001))
    assert value_of([lost], "protected_word_keep_rate") == 0.0


def test_recall_and_deletion_read_the_words_that_reached_the_asr_stage() -> None:
    heard = clip(truth(scenario="nst", expect="keep", user=speaking("the library opens at nine", 1.0, 2.0)),
                 frames(120) + [{"event": "transcript", "t_s": 2.4, "text": "the library opens at",
                                 "words": timed("the library opens at", 0.0, 1.0)}])
    assert value_of([heard], "user_recall_nst") == pytest.approx(0.8)
    assert value_of([heard], "user_word_deletion_rate") == pytest.approx(0.2)

    silent = clip(truth(scenario="nst", expect="keep", user=speaking("the library opens at nine", 1.0, 2.0)),
                  frames(120))
    assert value_of([silent], "user_recall_nst") == 0.0
    assert value_of([silent], "first_word_trunc_bargein") is None


def test_a_double_talk_clip_costs_wer_points_against_near_end_only_clips() -> None:
    dt = clip(truth(scenario="dt_middle", expect="keep",
                    playback=playing("the weather is clear", 0.0, 3.6),
                    user=speaking("the library opens at nine", 1.0, 2.0)),
              frames(180, ref_power=0.001) +
              [{"event": "transcript", "t_s": 2.4, "text": "the library opens at",
                "words": timed("the library opens at", 0.0, 1.0)}])
    nst = clip(truth(scenario="nst", expect="keep", user=speaking("the library opens at nine", 1.0, 2.0)),
               frames(120) + [{"event": "transcript", "t_s": 2.4, "text": "the library opens at nine",
                               "words": timed("the library opens at nine", 0.0, 1.0)}])
    assert value_of([dt, nst], "dt_wer_penalty_pts") == pytest.approx(20.0)


def test_every_stop_of_her_own_stream_counts_per_hour_of_playback() -> None:
    facts = [clip(truth(scenario="fst", playback=playing("the weather is clear", 0.0, 3.6)),
              frames(180, ref_power=0.001) +
              [{"event": "pause", "t_s": 1.2, "segment_seconds": 0.4},  # a segment boundary, not a stop
               {"event": "tts-stop", "t_s": 1.4, "frame": 70, "heard": "the weather"},
               {"event": "tts-stop", "t_s": 2.4, "frame": 120, "heard": "the weather"}])]
    assert value_of(facts, "self_pause_per_hour") == pytest.approx(2.0 / (3.6 / 3600.0))
    assert value_of(facts, "interrupt_stop_rate") == 1.0
    assert value_of(facts, "false_bargein_per_hour") == 0.0  # no trigger: only the stops are counted


def test_a_voxa_only_clip_that_becomes_a_command_is_the_serious_failure() -> None:
    facts = [clip(truth(scenario="fst", playback=playing("say stop to end this reply", 0.0, 3.6)),
              frames(180, ref_power=0.001) + [{"event": "pause", "t_s": 2.0, "segment_seconds": 0.6},
                                              {"event": "transcript", "t_s": 2.0, "text": "stop",
                                               "words": timed("stop", 0.0, 0.2)},
                                              {"event": "request", "t_s": 2.0, "text": "stop",
                                               "kind": "armed"}])]
    assert value_of(facts, "self_command_rate") == 1.0
    assert value_of(facts, "fst_transcript_rate") == 1.0


def test_cpu_core_share_reads_the_runner_s_own_clock() -> None:
    facts = [clip(truth(scenario="nst"), frames(10),
                {"wall_seconds": 2.0, "asr_processing_seconds": 1.0})]
    assert value_of(facts, "cpu_core_share") == pytest.approx(0.5)


def test_metrics_this_stage_cannot_measure_are_null_and_never_fail() -> None:
    rows = compute([], seed=0)
    for name in ("align_err_p90_ms", "erle_music_loss_db", "transport_drop_count"):
        assert rows[name]["value"] is None
        assert rows[name]["ci"] == [None, None]
        assert rows[name]["reason"]


# --------------------------------------------------------------------------- the report and check

def toy_facts() -> list[dict]:
    return [
        clip(truth(scenario="fst", playback=playing("the weather is clear", 0.0, 3.6)),
             frames(180, ref_power=0.001)),
        clip(truth(scenario="nst", expect="keep", user=speaking("the library opens", 1.0, 2.0)),
             frames(120) + [{"event": "transcript", "t_s": 2.4, "text": "the library opens",
                             "words": timed("the library opens", 0.0, 1.0)},
                            {"event": "endpoint", "t_s": 2.4, "text": "the library opens",
                             "complete": True},
                            {"event": "request", "t_s": 2.4, "text": "the library opens", "kind": "wake"}]),
    ]


def test_the_report_carries_every_metric_name_the_brief_fixes(tmp_path=None) -> None:
    from voxa.bench.report import build_report

    results = [{"id": "toy-0000-fst-00", "scenario": "fst", "counts": {"barge_in": 0, "suspect": 0,
                                                                      "own_voice": 0, "ignored": 0,
                                                                      "exit": 0},
                "requests": [], "seconds": 3.6, "wall_seconds": 1.0, "asr_calls": 1,
                "asr_audio_seconds": 3.6, "asr_processing_seconds": 0.5, "degraded": None,
                "delay_estimated_samples": 0}]
    report = build_report("smoke", 0, "voxa", results, toy_facts(), 9.6, 2.0, "base")
    assert sorted(report["metrics"]) == sorted(METRIC_NAMES)
    assert len(METRIC_NAMES) == 33
    for _name, row in report["metrics"].items():
        assert set(row) >= {"value", "ci", "threshold", "pass"}
        assert row["pass"] is True  # B3 sets no thresholds yet
        assert row["threshold"] is None
    assert report["pass"] is True
    assert report["by_scenario"]["nst"]["clips"] == 1
    assert report["by_scenario"]["nst"]["metrics"]["user_recall_nst"]["value"] == 1.0
    assert report["run"]["clips"] == 1


def test_gates_toml_lists_every_metric_with_no_threshold_yet() -> None:
    from voxa.bench import gates

    assert gates.thresholds() == {}  # B3: every metric listed, none limited yet
    listed = gates._load(gates.GATES_PATH).get("metric", [])
    assert sorted(entry["name"] for entry in listed) == sorted(METRIC_NAMES)
    assert gates.evaluate(gates.GATES_PATH, {"metrics": compute(toy_facts(), seed=0)}) == []


def test_check_prints_value_and_ci_and_exits_one_on_a_failure_a_gap_or_a_null(tmp_path: Path) -> None:
    from voxa.bench import report as report_module

    rows = compute(toy_facts(), seed=0)
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"suite": "smoke", "echo_mode": "voxa", "pass": True, "metrics": rows}))

    assert report_module.check([str(path), "user_recall_nst>=0"]) == 0
    assert report_module.check([str(path), "ep50_ms<=0"]) == 1  # the endpoint is 400 ms late
    assert report_module.check([str(path), "user_recall_nst>1"]) == 1
    assert report_module.check([str(path), "transport_drop_count<=0"]) == 1  # null: not a pass
    assert report_module.check([str(path), "not_a_metric>=0"]) == 1  # missing from the report
    assert report_module.check([str(path), "ep50_ms<1000", "user_recall_nst==1"]) == 0
    assert report_module.check([str(path), "user_recall_nst => 1"]) == 1  # unparsable


def test_a_threshold_in_gates_toml_fails_the_run(tmp_path: Path) -> None:
    from voxa.bench import gates

    path = tmp_path / "gates.toml"
    path.write_text('[[metric]]\nname = "user_recall_nst"\ndirection = ">="\nthreshold = 0.95\n')
    assert gates.thresholds(path) == {"user_recall_nst": (">=", 0.95)}
    failures = gates.evaluate(path, {"metrics": compute(toy_facts(), seed=0)})
    assert failures == []  # the toy clip recalls every word

    rows = compute(toy_facts(), seed=0)
    rows["user_recall_nst"]["value"] = 0.5
    assert gates.evaluate(path, {"metrics": rows}) == ['user_recall_nst>=0.95: user_recall_nst=0.5']
