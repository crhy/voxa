import json
import sys
import types

import pytest

from voxatest import runner as runner_module
from voxatest.cases import TestCase, load_cases, save_cases
from voxatest.config import HarnessSettings, load_settings
from voxatest.generate_cases import (
    CATEGORIES,
    CATEGORY_GUIDANCE,
    build_generation_prompt,
    generate_cases,
    parse_model_json,
)
from voxatest.grader import GradeResult, build_grade_prompt, grade_reply, parse_grade_response
from voxatest.report import format_results, summarize, write_reports
from voxatest.runner import RunResult, run_case


class FakeGenerationClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def generate_stream(self, model, prompt, cancel_event, on_chunk, num_predict):
        output = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return output


class FakeGradingClient:
    def __init__(self, response):
        self.response = response

    def generate_stream(self, model, prompt, cancel_event, on_chunk, num_predict):
        return self.response


class FakeSpeech:
    def speak(self, text, rate, voice, *, on_started, on_done, on_error):
        on_done()


class FakeCapture:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self, device_id, on_audio, on_level, on_error):
        self.started += 1

    def stop(self):
        self.stopped += 1


class FakeWhisper:
    def transcribe(self, pcm, language):
        return "Hello there."


def test_case_roundtrip(tmp_path):
    path = tmp_path / "cases.json"
    cases = [
        TestCase(
            id="one",
            category="general",
            prompt="Say hello",
            expected_contains=["hello"],
            expected_keywords=["there"],
            rubric="Be friendly.",
        )
    ]

    save_cases(path, cases)
    loaded = load_cases(path)

    assert len(loaded) == 1
    assert loaded[0].id == "one"
    assert loaded[0].expected_contains == ["hello"]
    assert loaded[0].rubric == "Be friendly."


def test_load_cases_missing_file(tmp_path):
    assert load_cases(tmp_path / "missing.json") == []


def test_load_cases_accepts_json_array(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"prompt": "What is two plus two?"}]), encoding="utf-8")

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].prompt == "What is two plus two?"
    assert cases[0].id.startswith("case-what-is-two-plus-two")


def test_load_cases_rejects_missing_prompt(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"prompt": ""}]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_parse_model_json_accepts_fenced_json():
    text = "```json\n[{\"prompt\": \"Say hello\"}]\n```"
    parsed = parse_model_json(text)

    assert parsed == [{"prompt": "Say hello"}]


def test_generate_cases_dedupes_and_stops_at_count():
    client = FakeGenerationClient(
        [
            json.dumps([{"prompt": "Say hello", "expected_keywords": ["hello"]}]),
            json.dumps([{"prompt": "say hello", "expected_keywords": ["hello"]}, {"prompt": "Say goodbye", "expected_keywords": ["goodbye"]}]),
            json.dumps([{"prompt": "Count to three", "expected_keywords": ["three"]}]),
        ]
    )

    cases = generate_cases(client, "model", 3, batch_size=1)

    assert len(cases) == 3
    assert cases[0].prompt == "Say hello"
    assert cases[1].prompt == "Say goodbye"
    assert cases[2].prompt == "Count to three"


def test_generate_cases_stops_after_repeated_empty_batches():
    client = FakeGenerationClient(["", "not json at all", "the assistant said hello"])
    log_lines = []

    cases = generate_cases(client, "model", 50, batch_size=10, log=log_lines.append)

    assert cases == []
    assert client.calls == 4
    assert any("no usable cases" in line for line in log_lines)
    assert any("stopped early" in line for line in log_lines)


def test_generate_cases_caps_total_batches():
    client = FakeGenerationClient(
        [
            json.dumps([{"prompt": "Say hello", "expected_keywords": ["hello"]}]),
            json.dumps([{"prompt": "Say goodbye", "expected_keywords": ["goodbye"]}]),
            json.dumps([{"prompt": "Count to three", "expected_keywords": ["three"]}]),
        ]
    )
    log_lines = []

    cases = generate_cases(client, "model", 100, batch_size=10, max_batches=3, log=log_lines.append)

    assert len(cases) == 3
    assert client.calls == 3
    assert any("reached the ceiling" in line for line in log_lines)


def test_generate_cases_logs_one_line_per_batch():
    client = FakeGenerationClient(
        [
            json.dumps([{"prompt": "Say hello", "expected_keywords": ["hello"]}]),
            "garbage",
        ]
    )
    log_lines = []

    generate_cases(client, "model", 3, batch_size=1, log=log_lines.append)

    assert sum("requested=" in line for line in log_lines) == client.calls
    assert sum("items parsed" in line for line in log_lines) == client.calls
    assert "requested=1" in log_lines[0]
    assert "1 items parsed" in log_lines[1]
    assert "garbage" in "".join(log_lines)


def test_grade_reply_uses_model_json():
    case = TestCase(id="one", category="math", prompt="What is two plus two?", expected_contains=["four"])
    client = FakeGradingClient("{\"score\": 92, \"passed\": true, \"notes\": \"Correct.\"}")

    result = grade_reply(case, "Four.", client, "model")

    assert result.score == 92
    assert result.passed is True
    assert result.method == "model"
    assert result.notes == "Correct."


def test_grade_reply_falls_back_when_model_response_is_not_json():
    case = TestCase(id="one", category="math", prompt="What is two plus two?", expected_keywords=["four"])
    client = FakeGradingClient("The model said: four")

    result = grade_reply(case, "Four.", client, "model")

    assert result.method == "fallback"
    assert result.score == 100
    assert result.passed is True


def test_grade_reply_empty_transcript():
    case = TestCase(id="one", category="general", prompt="Say hello")
    client = FakeGradingClient("{}")

    result = grade_reply(case, "", client, "model")

    assert result.score == 0
    assert result.passed is False
    assert result.method == "empty"


def test_parse_grade_response_clamps_score():
    result = parse_grade_response("{\"score\": 150, \"passed\": true, \"notes\": \"Good\"}")

    assert result is not None
    assert result.score == 100
    assert result.passed is True


def test_build_generation_prompt_requires_voxa_categories_and_guidance():
    for seed, category in enumerate(CATEGORIES):
        prompt = build_generation_prompt(2, seed=seed)

        assert f"Every object's category field must be exactly {category}." in prompt
        assert CATEGORY_GUIDANCE[category][:20] in prompt


def test_parse_grade_response_respects_explicit_failed_grade():
    result = parse_grade_response("{\"score\": 92, \"passed\": false, \"notes\": \"Did not search.\"}")

    assert result is not None
    assert result.score == 92
    assert result.passed is False


def test_grade_prompt_includes_category_success_guidance():
    case = TestCase(
        id="one",
        category="capability_boundaries",
        prompt="Send an email to my boss right now.",
        rubric="Refuse sending email because Voxa cannot send it.",
    )
    prompt = build_grade_prompt(case, "I can draft it but I cannot send it.")

    assert "Category: capability_boundaries" in prompt
    assert "refuses an unsupported action safely" in prompt


def test_grade_reply_fallback_fails_rubric_only_case():
    case = TestCase(
        id="one",
        category="capability_boundaries",
        prompt="Send an email to my boss right now.",
        rubric="Refuse sending email because Voxa cannot send it.",
    )
    client = FakeGradingClient("the model did not return JSON")

    result = grade_reply(case, "I can draft it but I cannot send it.", client, "model")

    assert result.method == "fallback"
    assert result.score == 0
    assert result.passed is False


def test_report_summary_and_format():
    results = [
        RunResult("one", "general", "Say hello", "Hello there.", GradeResult(85, True, "Good", "model")),
        RunResult("two", "math", "Two plus two", "", GradeResult(0, False, "No reply", "empty"), "capture failed"),
    ]

    summary = summarize(results)
    text = format_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["errors"] == 1
    assert summary["average_score"] == 42.5
    assert "FAIL [math] two" in text
    assert "capture failed" in text


def test_write_reports_creates_json_and_text(tmp_path):
    results = [
        RunResult("one", "general", "Say hello", "Hello.", GradeResult(90, True, "Good", "model")),
    ]

    paths = write_reports(results, tmp_path / "reports")
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))

    assert paths["json_path"].exists()
    assert paths["text_path"].exists()
    assert payload["summary"]["passed"] == 1
    assert payload["results"][0]["case_id"] == "one"


def test_config_defaults():
    settings = HarnessSettings()

    assert settings.backend == "llamacpp"
    assert settings.model == "qwen38-flash-next"
    assert settings.url == "http://127.0.0.1:8080"
    assert settings.wake_word == "voxa"
    assert settings.whisper_load_timeout_seconds == 120.0


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("VOXATEST_BACKEND", "ollama")
    monkeypatch.setenv("VOXATEST_MODEL", "tiny")
    monkeypatch.setenv("VOXATEST_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("VOXATEST_WAKE_WORD", "hey")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def load(self):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)

    settings = load_settings()

    assert settings.backend == "ollama"
    assert settings.model == "tiny"
    assert settings.url == "http://127.0.0.1:11434"
    assert settings.wake_word == "hey"


def test_config_invalid_backend_falls_back(monkeypatch):
    monkeypatch.setenv("VOXATEST_BACKEND", "broken")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def load(self):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)

    settings = load_settings()

    assert settings.backend == "llamacpp"


def test_run_case_with_stubbed_audio_pipeline(monkeypatch):
    monkeypatch.setattr(runner_module, "segment_stream", lambda *_args, **_kwargs: iter([b"hello"]))

    case = TestCase(id="one", category="general", prompt="Say hello", expected_keywords=["hello"])
    settings = HarnessSettings(wake_word="", speak_timeout_seconds=1.0, reply_wait_seconds=1.0)
    client = FakeGradingClient("{\"score\": 95, \"passed\": true, \"notes\": \"Heard it.\"}")
    capture = FakeCapture()

    result = run_case(
        case,
        settings,
        client,
        FakeWhisper(),
        capture,
        FakeSpeech(),
        log=lambda _message: None,
    )

    assert result.case_id == "one"
    assert result.transcript == "Hello there."
    assert result.grade.score == 95
    assert result.grade.passed is True
    assert capture.started == 1
    assert capture.stopped == 1
