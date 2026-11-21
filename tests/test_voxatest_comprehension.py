"""V1: voxatest scores COMPREHENSION - what Voxa heard vs what was said, and which layer failed.

The fixtures below are written run results, never a microphone: they cover every layer
(``not_heard`` / ``misheard`` / ``misrouted`` / ``misanswered`` / ``ok``), the confusion list, the
Comprehension section of every report, the ``comprehension`` and ``check`` commands, and the rule
that the text normalisation is the bench's own - imported, never copied.
"""

import json

import pytest

from voxa.agent.actionlog import ActionLog, ActionRecord
from voxa.bench import metrics as bench_metrics
from voxatest import __main__ as voxatest_main
from voxatest import comprehension
from voxatest import runner as runner_module
from voxatest.cases import TestCase
from voxatest.comprehension import (
    LAYERS,
    best_record_accuracy,
    case_line,
    check,
    classify_layer,
    comprehension_main,
    confusion_pairs,
    format_comprehension,
    format_comprehension_markdown,
    hearing_accuracy,
    rescore,
    score_case,
    score_run,
    strip_wake,
    tools_ran,
)
from voxatest.config import HarnessSettings
from voxatest.grader import GradeResult
from voxatest.report import format_failure_report, format_results, write_reports
from voxatest.runner import RunResult

WAKE = "voxa"
PROMPT = "set a timer for five minutes"
MISHEARD = "set a time for five minutes"  # one word of six: accuracy 5/6


class FakeCapture:
    def start(self, device_id, on_audio, on_level, on_error):
        return None

    def stop(self):
        return None


class FakeWhisper:
    def transcribe(self, _pcm, _language):
        return "Timer set."


class FakeSpeech:
    def speak(self, text, rate, voice, *, on_started, on_done, on_error):
        on_done()


def fixture(
    case_id="c1", category="timers", prompt=PROMPT, heard=PROMPT, *,
    passed=True, expected_tool="set_timer", tools=(), action_tool="", error="",
):
    """One written run result, in the shape a voxatest JSON report stores."""
    record = None
    if action_tool:
        record = {"heard": heard, "route": "tool", "tool": action_tool, "ok": True, "speech": "Timer set."}
    return RunResult(
        case_id, category, prompt, "Timer set.",
        GradeResult(100 if passed else 0, passed, "graded", "action_log"),
        error, 1.0, record, expected_tool, heard, 0, list(tools), [heard] if heard else [],
    ).to_dict()


def every_layer():
    """One fixture per layer, in the order the signal travels."""
    return [
        fixture("ok", tools=["set_timer"], action_tool="set_timer"),
        fixture("not_heard", heard="", tools=[]),
        fixture("misheard", heard=MISHEARD, passed=False, tools=["set_timer"], action_tool="set_timer"),
        fixture("misrouted", passed=False, tools=["start_timer"]),
        fixture("misanswered", passed=False, tools=["set_timer"], action_tool="set_timer"),
    ]


def run_result(case_id="ok", heard=PROMPT, *, passed=True, expected_tool="set_timer", tool="set_timer"):
    """The same fixture as the object ``write_reports`` takes."""
    record = {"heard": heard, "route": "tool", "tool": tool, "ok": True, "speech": "Timer set."} if tool else None
    return RunResult(
        case_id, "timers", PROMPT, "Timer set.",
        GradeResult(100 if passed else 0, passed, "graded", "action_log"),
        "", 1.0, record, expected_tool, heard, 0, [tool] if tool else [], [heard] if heard else [],
    )


# --------------------------------------------------------------------------- hearing accuracy


def test_hearing_accuracy_is_one_when_every_word_was_heard():
    assert hearing_accuracy(PROMPT, "set a timer for five minutes!", WAKE) == 1.0


def test_hearing_accuracy_ignores_the_wake_word_on_both_sides():
    # voxatest speaks "voxa, <prompt>", so the wake word is not part of what the case asked for.
    assert hearing_accuracy(PROMPT, f"Voxa, {PROMPT}", WAKE) == 1.0
    assert hearing_accuracy(f"{WAKE} {PROMPT}", PROMPT, WAKE) == 1.0
    assert hearing_accuracy(PROMPT, f"voxa {PROMPT}", "") < 1.0  # asked not to strip it: it is a word


def test_hearing_accuracy_spells_numbers_the_way_the_bench_does():
    # "7" and "seven" are the same word only because the bench normalisation spells numbers out.
    assert hearing_accuracy("set a timer for 7 minutes", "set a timer for seven minutes", WAKE) == 1.0
    assert hearing_accuracy("set a timer for 7 minutes", "set a timer for six minutes", WAKE) < 1.0


def test_hearing_accuracy_counts_deletions_and_insertions():
    assert hearing_accuracy(PROMPT, "set timer", WAKE) == pytest.approx(1 / 3)
    assert hearing_accuracy(PROMPT, f"{PROMPT} / open the door", WAKE) == pytest.approx(0.5)
    assert hearing_accuracy(PROMPT, "", WAKE) == 0.0  # heard nothing: zero, never negative


def test_hearing_accuracy_without_a_prompt_is_none():
    assert hearing_accuracy("", "anything at all", WAKE) is None


def test_strip_wake_removes_every_occurrence_of_the_wake_word():
    assert strip_wake([word for word in f"voxa {PROMPT} voxa".split()], ["voxa"]) == PROMPT.split()
    assert strip_wake(["hey", "voxa", "set"], ["hey", "voxa"]) == ["set"]
    assert strip_wake(["set", "a", "timer"], []) == ["set", "a", "timer"]


def test_normalisation_is_the_benchs_own_imported_functions():
    # The brief forbids a second copy of the bench's text handling: voxatest scores on the same one.
    assert comprehension.word_list is bench_metrics.word_list
    assert comprehension.wer is bench_metrics.wer
    assert comprehension.align is bench_metrics.align
    assert comprehension.normalise is bench_metrics.normalise


def test_best_record_accuracy_explains_a_window_that_logged_more_than_the_prompt():
    assert best_record_accuracy(PROMPT, [PROMPT, "open the door"], WAKE) == 1.0
    assert best_record_accuracy(PROMPT, [], WAKE) is None


# --------------------------------------------------------------------------- confusions


def test_confusion_pairs_lists_the_words_voxa_swapped():
    assert confusion_pairs(PROMPT, MISHEARD, WAKE) == [("timer", "time")]
    assert confusion_pairs(PROMPT, PROMPT, WAKE) == []
    # A deleted word is a deletion, not a confusion: nothing was heard in its place.
    assert confusion_pairs(PROMPT, "set timer", WAKE) == []


def test_confusion_pairs_survive_the_wake_word_and_numbers():
    assert confusion_pairs("set 3 timers", "voxa set two timers", WAKE) == [("three", "two")]


# --------------------------------------------------------------------------- layers


def test_every_layer_is_named_by_the_first_stage_that_failed():
    expected = {layer: layer for layer in LAYERS}
    for row in every_layer():
        accuracy = hearing_accuracy(row["prompt"], row["heard"], WAKE)
        assert classify_layer(row, accuracy, WAKE) == expected[row["case_id"]]


def test_a_case_that_heard_nothing_is_not_heard_even_when_a_tool_ran():
    row = fixture("late", heard="", tools=["set_timer"], action_tool="set_timer")
    assert classify_layer(row, hearing_accuracy(PROMPT, "", WAKE), WAKE) == "not_heard"


def test_a_harness_failure_is_reported_as_not_heard_and_keeps_its_error():
    scored = score_case(fixture("late", heard="", error="Microphone capture failed."), WAKE)
    assert scored["layer"] == "not_heard"
    assert scored["error"] == "Microphone capture failed."


def test_heard_right_but_the_wrong_tool_is_misrouted():
    row = fixture("late", passed=False, tools=["start_timer"])
    assert classify_layer(row, 1.0, WAKE) == "misrouted"


def test_a_case_without_an_expected_tool_cannot_be_misrouted():
    row = fixture("late", prompt="what time is it", heard="what time is it", expected_tool="", tools=[])
    assert classify_layer(row, 1.0, WAKE) == "ok"


def test_tools_ran_reads_the_action_record_of_an_older_report():
    assert tools_ran(fixture("late", tools=[], action_tool="set_timer")) == ["set_timer"]
    assert tools_ran({"heard": PROMPT}) == []


# --------------------------------------------------------------------------- run scoring


def test_score_run_reports_the_four_keys_the_brief_fixes():
    block = score_run(every_layer(), WAKE)
    for key in ("hearing_word_accuracy", "routed_tool_accuracy", "layer_counts", "confusions"):
        assert key in block
    assert block["layer_counts"] == {layer: 1 for layer in LAYERS}


def test_score_run_averages_hearing_and_routing_over_the_right_cases():
    block = score_run(every_layer(), WAKE)
    # ok 1.0, not_heard 0.0, misheard 5/6, misrouted 1.0, misanswered 1.0.
    assert block["hearing_word_accuracy"] == pytest.approx(round((1 + 0 + 5 / 6 + 1 + 1) / 5, 4))
    assert block["routed_tool_accuracy"] == pytest.approx(0.6)  # 3 of the 5 ran the expected tool
    assert block["routed_cases"] == 5
    assert block["cases_scored"] == 5


def test_routed_tool_accuracy_ignores_cases_that_name_no_tool():
    block = score_run([fixture("chat", prompt="what time is it", heard="what time is it",
                               expected_tool="", tools=[])], WAKE)
    assert block["routed_tool_accuracy"] is None
    assert block["routed_cases"] == 0


def test_score_run_groups_accuracy_per_category():
    rows = every_layer() + [fixture("chat", category="math", prompt="two plus two",
                                    heard="two plus two", expected_tool="", tools=[])]
    block = score_run(rows, WAKE)
    assert block["by_category"]["math"]["cases"] == 1
    assert block["by_category"]["math"]["hearing_word_accuracy"] == 1.0
    assert block["by_category"]["timers"]["cases"] == 5
    assert block["by_category"]["timers"]["layer_counts"]["misheard"] == 1


def test_confusion_list_is_sorted_by_count_and_capped_at_the_top_rows():
    rows = [fixture(f"p{index}", heard=MISHEARD, passed=False, tools=["set_timer"]) for index in range(3)]
    rows.append(fixture("w", prompt="set a timer for nine minutes", heard="set a timer for wine minutes",
                        passed=False, tools=["set_timer"]))
    assert score_run(rows, WAKE, top=1)["confusions"] == [{"said": "timer", "heard": "time", "count": 3}]
    assert len(score_run(rows, WAKE, top=30)["confusions"]) == 2


def test_confusion_list_is_empty_when_every_word_was_heard():
    assert score_run([fixture()], WAKE)["confusions"] == []


def test_score_run_of_an_empty_run_is_honest_about_having_no_numbers():
    block = score_run([], WAKE)
    assert block["hearing_word_accuracy"] is None
    assert block["routed_tool_accuracy"] is None
    assert block["layer_counts"] == {layer: 0 for layer in LAYERS}
    assert block["confusions"] == []


def test_score_case_keeps_the_detail_a_human_needs():
    scored = score_case(every_layer()[2], WAKE)
    assert scored["case_id"] == "misheard"
    assert scored["hearing_word_accuracy"] == pytest.approx(0.8333)
    assert scored["hearing_word_accuracy_best_record"] == pytest.approx(0.8333)
    assert scored["layer"] == "misheard"
    assert scored["routed"] is True
    assert scored["tools_ran"] == ["set_timer"]
    assert scored["confusions"] == [["timer", "time"]]


# --------------------------------------------------------------------------- reports


def test_json_report_carries_the_comprehension_block(tmp_path):
    paths = write_reports([run_result()], tmp_path / "reports", wake_word=WAKE)
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    block = payload["comprehension"]

    assert block["hearing_word_accuracy"] == 1.0
    assert block["routed_tool_accuracy"] == 1.0
    assert block["layer_counts"]["ok"] == 1
    assert block["cases"][0]["case_id"] == "ok"


def test_text_report_has_a_comprehension_section_and_a_layer_per_case():
    text = format_results([run_result(), run_result("mis", heard=MISHEARD, passed=True)], wake_word=WAKE)

    assert "Comprehension" in text
    assert "hearing_word_accuracy" in text
    assert "routed_tool_accuracy" in text
    assert "CONFUSIONS" in text
    assert "timer -> time" in text
    assert "comprehension: layer=misheard hearing=0.8333 routed=yes" in text


def test_failure_markdown_has_a_comprehension_section():
    markdown = format_failure_report([run_result("mis", heard=MISHEARD, passed=False)], wake_word=WAKE)

    assert "## Comprehension" in markdown
    assert "| Category | Cases | Hearing accuracy | Routed tool accuracy |" in markdown
    assert "| timers | 1 | 0.8333 | 1.0000 |" in markdown
    assert "| timer | time | 1 |" in markdown


def test_comprehension_section_says_so_when_nothing_was_swapped():
    assert any("none" in line for line in format_comprehension(score_run([fixture()], WAKE)))
    assert "None: every word was heard as it was said." in format_comprehension_markdown(score_run([fixture()], WAKE))


def test_case_line_names_the_layer_and_the_hearing_accuracy():
    assert case_line(fixture(tools=["set_timer"]), WAKE) == "comprehension: layer=ok hearing=1.0000 routed=yes"
    assert "routed=n/a" in case_line(fixture(expected_tool="", tools=[]), WAKE)


# --------------------------------------------------------------------------- commands


def test_comprehension_command_rescores_a_written_report(tmp_path, capsys):
    paths = write_reports([run_result()], tmp_path / "reports", wake_word=WAKE)
    out = tmp_path / "rescored.json"

    assert comprehension_main(["--report", str(paths["json_path"]), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    written = json.loads(out.read_text(encoding="utf-8"))

    assert "Comprehension" in printed
    assert "hearing_word_accuracy" in printed
    assert written["comprehension"]["layer_counts"]["ok"] == 1
    assert written["results"][0]["case_id"] == "ok"


def test_comprehension_command_prints_json_when_asked(tmp_path, capsys):
    paths = write_reports([run_result()], tmp_path / "reports", wake_word=WAKE)

    assert comprehension_main(["--report", str(paths["json_path"]), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["hearing_word_accuracy"] == 1.0
    assert payload["layer_counts"]["ok"] == 1


def test_comprehension_command_reports_an_unreadable_report(tmp_path):
    assert comprehension_main(["--report", str(tmp_path / "missing.json")]) == 2


def test_comprehension_command_rejects_a_report_without_results(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"summary": {}}), encoding="utf-8")

    assert comprehension_main(["--report", str(path)]) == 2


def test_rescore_keeps_the_rest_of_the_report(tmp_path):
    paths = write_reports([], tmp_path / "reports", wake_word=WAKE)
    report = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    rescored = rescore(report, WAKE)

    assert rescored["summary"] == report["summary"]
    assert rescored["comprehension"]["cases_scored"] == 0


def test_check_passes_conditions_on_a_scored_report(tmp_path, capsys):
    paths = write_reports([run_result()], tmp_path / "reports", wake_word=WAKE)

    code = check([str(paths["json_path"]), "hearing_word_accuracy>=1.0", "layer_counts.ok==1",
                  "by_category.timers.hearing_word_accuracy>=0.9"])
    printed = capsys.readouterr().out

    assert code == 0
    assert "hearing_word_accuracy = 1.0000 OK" in printed
    assert "layer_counts.ok = 1 OK" in printed


def test_check_fails_a_condition_that_does_not_hold(tmp_path, capsys):
    paths = write_reports([run_result("mis", heard=MISHEARD, passed=False)], tmp_path / "reports", wake_word=WAKE)

    assert check([str(paths["json_path"]), "hearing_word_accuracy>=1.0"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_fails_on_a_missing_metric_a_null_and_an_unparsable_condition(tmp_path, capsys):
    paths = write_reports([], tmp_path / "reports", wake_word=WAKE)

    assert check([str(paths["json_path"]), "hearing_word_accuracy>=0.5"]) == 1  # null: nothing was scored
    assert check([str(paths["json_path"]), "layer_counts.nope>=1"]) == 1  # not a layer
    assert check([str(paths["json_path"]), "hearing_word_accuracy~0.5"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_check_reports_an_unreadable_report(tmp_path):
    assert check([str(tmp_path / "missing.json"), "hearing_word_accuracy>=0.5"]) == 2


def test_cli_wires_comprehension_and_check(tmp_path, capsys):
    paths = write_reports([run_result()], tmp_path / "reports", wake_word=WAKE)
    capsys.readouterr()

    assert voxatest_main.main(["comprehension", "--report", str(paths["json_path"])]) == 0
    assert "Comprehension" in capsys.readouterr().out
    assert voxatest_main.main(["check", str(paths["json_path"]), "routed_tool_accuracy>=1.0"]) == 0
    assert voxatest_main.main(["check", str(paths["json_path"]), "routed_tool_accuracy>=1.1"]) == 1


# --------------------------------------------------------------------------- the runner side


def test_run_result_records_every_tool_and_every_heard_record(tmp_path, monkeypatch):
    log_path = tmp_path / "actions.jsonl"

    def fake_segment_stream(*_args, **_kwargs):
        # Voxa acts while the case is live: the harness reads the log after the listening window.
        ActionLog(log_path).append(
            ActionRecord(time="2026-11-19T00:00:00", heard=PROMPT, route="tool", tool="set_timer", ok=True),
        )
        ActionLog(log_path).append(
            ActionRecord(time="2026-11-19T00:00:01", heard="open the door", route="tool", tool="open_app", ok=True),
        )
        return iter([b"hello"])

    monkeypatch.setattr(runner_module, "segment_stream", fake_segment_stream)

    case = TestCase(id="one", category="timers", prompt=PROMPT, expected_tool="set_timer")
    settings = HarnessSettings(wake_word=WAKE, settle_seconds=0.0, speak_timeout_seconds=1.0,
                               reply_wait_seconds=1.0, action_log_path=log_path)

    result = runner_module.run_case(case, settings, object(), FakeWhisper(), FakeCapture(), FakeSpeech(),
                                    log=lambda _message: None)
    block = score_run([result.to_dict()], WAKE)

    assert result.heard == f"{PROMPT} / open the door"
    assert result.heard_records == [PROMPT, "open the door"]
    assert result.observed_tools == ["set_timer", "open_app"]
    assert result.to_dict()["observed_tools"] == ["set_timer", "open_app"]
    # The whole heard text carries three words nobody asked for, so the honest total is 0.5 ...
    assert block["hearing_word_accuracy"] == 0.5
    assert block["layer_counts"]["misheard"] == 1
    # ... while the per-record diagnostic shows the prompt itself was heard word for word.
    assert block["cases"][0]["hearing_word_accuracy_best_record"] == 1.0
    assert block["routed_tool_accuracy"] == 1.0


def test_run_result_defaults_to_empty_lists_when_the_log_stayed_empty():
    result = RunResult("one", "general", "Say hello", "Hello.", GradeResult(90, True, "ok", "model"))

    assert result.observed_tools == []
    assert result.heard_records == []
    assert result.to_dict()["observed_tools"] == []
    assert result.to_dict()["heard_records"] == []
