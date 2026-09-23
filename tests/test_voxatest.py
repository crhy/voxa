import json
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from voxatest import __main__ as voxatest_main
from voxatest import runner as runner_module
from voxatest.cases import TestCase, load_cases, save_cases
from voxatest.config import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    DEFAULT_URL,
    HarnessSettings,
    _env_number,
    choose_voxa_config,
    default_failure_report_path,
    default_reports_dir,
    load_settings,
    voxa_config_candidates,
)
from voxatest.generate_cases import (
    CATEGORIES,
    CATEGORY_GUIDANCE,
    build_generation_prompt,
    generate_cases,
    parse_model_json,
)
from voxatest.grader import GradeResult, build_grade_prompt, grade_reply, parse_grade_response
from voxatest.report import (
    PROGRESS_RE,
    format_failure_report,
    format_results,
    parse_progress_line,
    report_paths,
    summarize,
    write_reports,
)
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

    def list_models(self):
        return ["qwen38-flash-next"]


class FakeSpeech:
    def speak(self, text, rate, voice, *, on_started, on_done, on_error):
        on_done()

    def stop(self):
        return None


class FailingSpeech:
    def speak(self, text, rate, voice, *, on_started, on_done, on_error):
        on_error("speaker unavailable")


class FakeMainLoop:
    def __init__(self):
        self.ran = threading.Event()
        self.quit_called = False

    def run(self):
        self.ran.set()

    def quit(self):
        self.quit_called = True


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


def test_write_reports_creates_git_friendly_failure_snapshot(tmp_path):
    results = [
        RunResult("pass", "general", "Say hello", "Hello.", GradeResult(90, True, "Good", "model")),
        RunResult(
            "fail",
            "math",
            "Two plus two",
            "Five",
            GradeResult(10, False, "Wrong answer", "model"),
            "capture warning",
        ),
    ]
    failure_path = tmp_path / "VOXATEST_FAILURES.md"

    paths = write_reports(results, tmp_path / "reports", failure_path=failure_path)
    report = failure_path.read_text(encoding="utf-8")

    assert paths["failure_path"] == failure_path
    assert "## fail" in report
    assert "Two plus two" in report
    assert "Wrong answer" in report
    assert "capture warning" in report
    assert "## pass" not in report
    assert "timestamp" not in report.casefold()


def test_failure_report_records_clean_latest_run():
    result = RunResult("pass", "general", "Say hello", "Hello.", GradeResult(90, True, "Good", "model"))

    report = format_failure_report([result])

    assert "Failed: 0" in report
    assert "No failures were reported in the latest run." in report


def test_write_reports_uses_unique_names_for_concurrent_runs(tmp_path):
    first = write_reports([], tmp_path)
    second = write_reports([], tmp_path)

    assert first["json_path"] != second["json_path"]
    assert first["text_path"] != second["text_path"]
    assert len(list(tmp_path.glob("run_*.json"))) == 2


def test_report_paths_honors_explicit_paths(tmp_path):
    json_path = tmp_path / "chosen.json"
    text_path = tmp_path / "chosen.txt"

    resolved = report_paths(tmp_path / "reports", json_path=json_path, text_path=text_path)

    assert resolved == (json_path, text_path)


def test_report_paths_defaults_to_unique_run_files(tmp_path):
    json_path, text_path = report_paths(tmp_path)

    assert json_path.name.startswith("run_") and json_path.suffix == ".json"
    assert text_path.name.startswith("run_") and text_path.suffix == ".txt"
    assert json_path != text_path


def test_reports_mark_partial_runs(tmp_path):
    results = [
        RunResult(
            "one",
            "general",
            "Say hello",
            "Hello.",
            GradeResult(90, True, "Good", "model"),
            duration_seconds=1.25,
        )
    ]
    failure_path = tmp_path / "VOXATEST_FAILURES.md"

    paths = write_reports(results, tmp_path / "reports", failure_path=failure_path, complete=False)
    payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    text = paths["text_path"].read_text(encoding="utf-8")
    report = failure_path.read_text(encoding="utf-8")

    assert payload["complete"] is False
    assert text.splitlines()[0] == "PARTIAL RUN (interrupted or still running)"
    assert "- Status: partial run" in report
    assert "(1.2s)" in format_results(results)


def test_parse_progress_line_reads_harness_status_lines():
    line = "[3/12] PASS wake-hello score 90 (1.2s)"
    assert parse_progress_line(line) == (3, 12, "PASS")
    assert parse_progress_line("[1/1] FAIL a1 score 0 (0.0s)") == (1, 1, "FAIL")
    assert parse_progress_line("[0/5] ERROR crash-case score 0 (0.1s)") == (0, 5, "ERROR")
    assert parse_progress_line("  [7/9] PASS   padded  ") == (7, 9, "PASS")


def test_parse_progress_line_rejects_non_progress_lines():
    for line in (
        "Voxa settings: /etc/vox.yaml — wake word \"hey\", model llama at http://127.0.0.1:11434",
        "Empty transcript for wake-hello; retrying (attempt 2 of 3).",
        "[3/12] PASSED wake-hello",
        "[3/12] SKIP wake-hello",
        "Total: 12",
        "",
        "1/2 PASS",
        "done [1/2] PASS case-one",
    ):
        assert parse_progress_line(line) is None
        assert PROGRESS_RE.match(line) is None


def test_progress_regex_matches_every_line_the_run_loop_prints(tmp_path, monkeypatch, capsys):
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    monkeypatch.setattr(
        voxatest_main,
        "run_case",
        lambda *_args, **_kwargs: RunResult(
            "one", "general", "Say hello", "Hello.", GradeResult(90, True, "Good", "model"), duration_seconds=0.5
        ),
    )

    code = voxatest_main._run(
        HarnessSettings(retries=0, settle_seconds=0.0),
        [TestCase(id="one", category="general", prompt="Say hello"), TestCase(id="two", category="general", prompt="Say again")],
        str(tmp_path / "run.json"),
        str(tmp_path / "run.txt"),
        tmp_path / "reports",
        tmp_path / "failures.md",
    )
    assert code == 0

    progress = [parse_progress_line(line) for line in capsys.readouterr().out.splitlines()]
    assert [entry for entry in progress if entry] == [(1, 2, "PASS"), (2, 2, "PASS")]


def test_default_reports_dir_is_separate_and_honors_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXATEST_REPORTS_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert default_reports_dir() == tmp_path / "state" / "voxatest" / "reports"

    monkeypatch.setenv("VOXATEST_REPORTS_DIR", str(tmp_path / "custom"))
    assert default_reports_dir() == tmp_path / "custom"


def test_default_failure_report_path_honors_override(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXATEST_FAILURE_REPORT", raising=False)
    assert default_failure_report_path() == Path("VOXATEST_FAILURES.md")

    target = tmp_path / "failures.md"
    monkeypatch.setenv("VOXATEST_FAILURE_REPORT", str(target))
    assert default_failure_report_path() == target


def test_config_defaults():
    settings = HarnessSettings()

    assert settings.backend == "llamacpp"
    assert settings.model == "qwen38-flash-next"
    assert settings.url == "http://127.0.0.1:8080"
    assert settings.wake_word == "voxa"
    assert settings.whisper_load_timeout_seconds == 120.0
    assert settings.run_deadline_seconds == 0.0
    assert settings.max_reply_seconds == 90.0
    assert settings.settle_seconds == 1.5
    assert settings.retries == 1


def test_config_numeric_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXATEST_RUN_DEADLINE", "600")
    monkeypatch.setenv("VOXATEST_END_SILENCE", "2.5")
    monkeypatch.setenv("VOXATEST_REPLY_WAIT", "30")
    monkeypatch.setenv("VOXATEST_RETRIES", "3")
    monkeypatch.setenv("VOXATEST_RETRIES", "3")
    monkeypatch.setenv("VOXATEST_MAX_SILENT", "2")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, _path=None):
            pass

        def load(self, **_kwargs):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(tmp_path / "voxa_config.json"))
    (tmp_path / "voxa_config.json").write_text("{}", encoding="utf-8")

    settings = load_settings()

    assert settings.run_deadline_seconds == 600.0
    assert settings.end_silence_seconds == 2.5
    assert settings.reply_wait_seconds == 30.0
    assert settings.retries == 3
    assert settings.max_consecutive_no_reply == 2


def test_config_invalid_numeric_env_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXATEST_RETRIES", "many")
    monkeypatch.setenv("VOXATEST_REPLY_WAIT", "1.5.5")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, _path=None):
            pass

        def load(self, **_kwargs):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(tmp_path / "voxa_config.json"))
    (tmp_path / "voxa_config.json").write_text("{}", encoding="utf-8")

    settings = load_settings()

    assert settings.retries == 1
    assert settings.reply_wait_seconds == 20.0
    assert _env_number("VOXATEST_UNSET", 7.5, float) == 7.5


def test_config_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXATEST_BACKEND", "ollama")
    monkeypatch.setenv("VOXATEST_MODEL", "tiny")
    monkeypatch.setenv("VOXATEST_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("VOXATEST_WAKE_WORD", "hey")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, _path=None):
            pass

        def load(self, **_kwargs):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(tmp_path / "voxa_config.json"))
    (tmp_path / "voxa_config.json").write_text("{}", encoding="utf-8")

    settings = load_settings()

    assert settings.backend == "ollama"
    assert settings.model == "tiny"
    assert settings.url == "http://127.0.0.1:11434"
    assert settings.wake_word == "hey"


def test_config_invalid_backend_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXATEST_BACKEND", "broken")

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, _path=None):
            pass

        def load(self, **_kwargs):
            return None

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(tmp_path / "voxa_config.json"))
    (tmp_path / "voxa_config.json").write_text("{}", encoding="utf-8")

    settings = load_settings()

    assert settings.backend == "llamacpp"


@pytest.mark.parametrize(
    ("backend", "expected_model", "expected_url"),
    [
        ("ollama", "ollama-test", "http://ollama.test:11434"),
        ("llamacpp", "llamacpp-test", "http://llamacpp.test:8080"),
    ],
)
def test_config_follows_voxa_backend(monkeypatch, tmp_path, backend, expected_model, expected_url):
    for name in ("VOXATEST_BACKEND", "VOXATEST_MODEL", "VOXATEST_URL", "FLATPAK_ID"):
        monkeypatch.delenv(name, raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(config_path))

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, _path=None):
            pass

        def load(self, **_kwargs):
            return types.SimpleNamespace(
                ai_backend=backend,
                ollama_model="ollama-test",
                ollama_url="http://ollama.test:11434",
                llamacpp_model="llamacpp-test",
                llamacpp_url="http://llamacpp.test:8080",
                microphone_id="",
                wake_word="voxa",
                whisper_model="base",
                language="en",
                tts_rate=180,
                tts_voice="voice",
                voice_threshold=450,
                silence_ms=900,
            )

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)

    settings = load_settings()

    assert settings.backend == backend
    assert settings.model == expected_model
    assert settings.url == expected_url
    assert settings.voxa_config_path == config_path


def test_voxa_config_candidate_ordering_and_dedup(monkeypatch, tmp_path):
    for name in ("VOXATEST_VOXA_CONFIG", "XDG_CONFIG_HOME", "FLATPAK_ID"):
        monkeypatch.delenv(name, raising=False)

    override = tmp_path / "override.json"
    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(override))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    candidates = voxa_config_candidates()
    assert candidates[0] == override
    assert candidates[1] == tmp_path / "xdg" / "voxa" / "config.json"
    assert candidates[2] == Path.home() / ".config" / "voxa" / "config.json"
    assert candidates[3] == Path.home() / ".var" / "app" / "io.github.crhy.voxa" / "config" / "voxa" / "config.json"
    assert len(candidates) == len(set(candidates))

    monkeypatch.setenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    assert len(voxa_config_candidates()) == 3


def test_choose_voxa_config_prefers_newest_file(monkeypatch, tmp_path):
    for name in ("VOXATEST_VOXA_CONFIG", "XDG_CONFIG_HOME", "FLATPAK_ID"):
        monkeypatch.delenv(name, raising=False)

    host = tmp_path / "host" / "config.json"
    flatpak = tmp_path / "flatpak" / "config.json"
    for path in (host, flatpak):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    os.utime(host, (1000, 1000))
    os.utime(flatpak, (2000, 2000))

    candidates = [host, flatpak]
    assert choose_voxa_config(candidates) == flatpak
    assert choose_voxa_config(list(reversed(candidates))) == flatpak

    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(host))
    assert choose_voxa_config(candidates) == host

    monkeypatch.setenv("VOXATEST_VOXA_CONFIG", str(tmp_path / "missing.json"))
    assert choose_voxa_config(candidates) is None
    assert choose_voxa_config([]) is None


def test_load_settings_uses_most_recently_modified_voxa_config(monkeypatch, tmp_path):
    for name in ("VOXATEST_VOXA_CONFIG", "XDG_CONFIG_HOME", "FLATPAK_ID", "VOXATEST_BACKEND"):
        monkeypatch.delenv(name, raising=False)

    older = tmp_path / "host-config.json"
    newer = tmp_path / "flatpak-config.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    monkeypatch.setattr(
        "voxatest.config.voxa_config_candidates",
        lambda: [older, newer],
    )

    fake_module = types.ModuleType("voxa.config")

    class FakeConfigStore:
        def __init__(self, path):
            self.path = path

        def load(self, **_kwargs):
            return types.SimpleNamespace(
                ai_backend="llamacpp",
                microphone_id="",
                wake_word="voxa" if self.path == newer else "computer",
                whisper_model="base",
                language="en",
                tts_rate=180,
                tts_voice="en-US-AriaNeural",
                voice_threshold=450,
                silence_ms=900,
                ollama_model="",
                ollama_url="",
                llamacpp_model="",
                llamacpp_url="",
            )

    fake_module.ConfigStore = FakeConfigStore
    monkeypatch.setitem(sys.modules, "voxa.config", fake_module)

    settings = load_settings()
    assert settings.voxa_config_path == newer
    assert settings.wake_word == "voxa"


def test_reports_and_failure_paths_inside_flatpak(monkeypatch):
    monkeypatch.delenv("VOXATEST_REPORTS_DIR", raising=False)
    monkeypatch.delenv("VOXATEST_FAILURE_REPORT", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("FLATPAK_ID", "io.github.crhy.voxa")
    assert default_reports_dir() == Path.home() / "Documents" / "VoxaTest" / "reports"
    assert default_failure_report_path() == Path.home() / "Documents" / "VoxaTest" / "VOXATEST_FAILURES.md"

    monkeypatch.delenv("FLATPAK_ID")
    assert default_reports_dir() == Path.home() / ".local" / "state" / "voxatest" / "reports"
    assert default_failure_report_path() == Path("VOXATEST_FAILURES.md")

    monkeypatch.setenv("VOXATEST_REPORTS_DIR", "/somewhere/reports")
    monkeypatch.setenv("VOXATEST_FAILURE_REPORT", "/somewhere/failures.md")
    monkeypatch.setenv("FLATPAK_ID", "io.github.crhy.voxa")
    assert default_reports_dir() == Path("/somewhere/reports")
    assert default_failure_report_path() == Path("/somewhere/failures.md")


def test_load_settings_without_voxa_config_keeps_defaults(monkeypatch, tmp_path):
    for name in ("VOXATEST_BACKEND", "VOXATEST_MODEL", "VOXATEST_URL", "VOXATEST_WAKE_WORD", "FLATPAK_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "voxatest.config.voxa_config_candidates",
        lambda: [tmp_path / "does-not-exist.json"],
    )

    settings = load_settings()

    assert settings.voxa_config_path is None
    assert settings.backend == DEFAULT_BACKEND
    assert settings.model == DEFAULT_MODEL
    assert settings.url == DEFAULT_URL


def test_start_glib_main_loop_runs_callbacks(monkeypatch):
    loop = FakeMainLoop()
    repository = types.ModuleType("gi.repository")
    repository.GLib = types.SimpleNamespace(MainLoop=lambda: loop)
    gi = types.ModuleType("gi")
    gi.repository = repository
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    started_loop, thread = voxatest_main._start_glib_main_loop()
    thread.join(timeout=1.0)
    started_loop.quit()

    assert started_loop is loop
    assert loop.ran.is_set()
    assert loop.quit_called is True


def test_run_case_with_stubbed_audio_pipeline(monkeypatch):
    monkeypatch.setattr(runner_module, "segment_stream", lambda *_args, **_kwargs: iter([b"hello"]))

    case = TestCase(id="one", category="general", prompt="Say hello", expected_keywords=["hello"])
    settings = HarnessSettings(wake_word="", settle_seconds=0.0, speak_timeout_seconds=1.0, reply_wait_seconds=1.0)
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


def test_run_case_transcribes_every_captured_segment(monkeypatch):
    monkeypatch.setattr(runner_module, "segment_stream", lambda *_args, **_kwargs: iter([b"one", b"two"]))

    class TwoPartWhisper:
        def transcribe(self, pcm, language):
            return {b"one": "First part.", b"two": "Second part."}[pcm]

    case = TestCase(id="two", category="general", prompt="Tell me a story", expected_keywords=["story"])
    settings = HarnessSettings(wake_word="", settle_seconds=0.0, speak_timeout_seconds=1.0, reply_wait_seconds=1.0)
    client = FakeGradingClient("{\"score\": 88, \"passed\": true, \"notes\": \"Both parts heard.\"}")
    capture = FakeCapture()

    result = run_case(
        case,
        settings,
        client,
        TwoPartWhisper(),
        capture,
        FakeSpeech(),
        log=lambda _message: None,
    )

    assert result.transcript == "First part. Second part."
    assert result.grade.score == 88
    assert result.duration_seconds >= 0.0
    assert result.to_dict()["duration_seconds"] == round(result.duration_seconds, 2)
    assert capture.started == 1
    assert capture.stopped == 1


def test_run_case_returns_immediately_when_speech_fails():
    case = TestCase(id="one", category="general", prompt="Say hello")
    settings = HarnessSettings(speak_timeout_seconds=30.0)
    capture = FakeCapture()

    result = run_case(
        case,
        settings,
        FakeGradingClient("{}"),
        FakeWhisper(),
        capture,
        FailingSpeech(),
        log=lambda _message: None,
    )

    assert result.error == "speaker unavailable"
    assert result.grade.notes == "TTS failed."
    assert capture.started == 0


def _stub_harness(monkeypatch, capture, speech):
    """Replace the audio, speech and GLib dependencies of the run loop with fakes."""
    audio_module = types.ModuleType("voxa.audio")
    audio_module.AudioCapture = lambda: capture
    speech_module = types.ModuleType("voxa.speech")
    speech_module.SpeechService = lambda: speech
    monkeypatch.setitem(sys.modules, "voxa.audio", audio_module)
    monkeypatch.setitem(sys.modules, "voxa.speech", speech_module)

    loop = FakeMainLoop()
    repository = types.ModuleType("gi.repository")
    repository.GLib = types.SimpleNamespace(MainLoop=lambda: loop)
    gi_module = types.ModuleType("gi")
    gi_module.repository = repository
    monkeypatch.setitem(sys.modules, "gi", gi_module)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    monkeypatch.setattr(voxatest_main, "_load_whisper", lambda _settings, _log: (FakeWhisper(), ""))
    return loop


def test_run_records_harness_errors_and_continues(tmp_path, monkeypatch, capsys):
    cases = [
        TestCase(id="boom", category="general", prompt="Say hello"),
        TestCase(id="good", category="general", prompt="Say hello", expected_keywords=["hello"]),
    ]
    settings = HarnessSettings(retries=0, settle_seconds=0.0)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())

    def fake_run_case(case, _settings, _client, _whisper, _capture, _speech, log=None):
        if case.id == "boom":
            raise RuntimeError("capture exploded")
        return RunResult(case.id, case.category, case.prompt, "Hello there.", GradeResult(85, True, "Good", "model"), duration_seconds=14.2)

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)

    code = voxatest_main._run(settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    out = capsys.readouterr().out

    assert code == 1
    assert "[1/2] ERROR boom score 0 (0.0s)" in out
    assert "[2/2] PASS good score 85 (14.2s)" in out

    json_paths = list((tmp_path / "reports").glob("run_*.json"))
    assert len(json_paths) == 1
    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert len(payload["results"]) == 2
    assert payload["results"][0]["error"] == "capture exploded"
    assert payload["results"][0]["notes"] == "Harness error."
    assert payload["results"][0]["method"] == "error"
    assert payload["results"][1]["passed"] is True
    assert "boom" in (tmp_path / "failures.md").read_text(encoding="utf-8")


def test_run_preflight_returns_two_when_backend_is_unreachable(tmp_path, monkeypatch, capsys):
    class UnreachableClient:
        def list_models(self):
            raise OSError("connection refused")

    settings = HarnessSettings()
    monkeypatch.setattr(HarnessSettings, "make_client", lambda _self: UnreachableClient())

    code = voxatest_main._run(settings, [], None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    err = capsys.readouterr().err

    assert code == 2
    assert "Grading model backend is not reachable at http://127.0.0.1:8080: connection refused" in err
    assert not list((tmp_path / "reports").glob("run_*.json"))


def test_run_retries_cases_with_empty_transcript(tmp_path, monkeypatch, capsys):
    settings = HarnessSettings(retries=2, settle_seconds=0.0)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    attempts = []

    def fake_run_case(case, _settings, _client, _whisper, _capture, _speech, log=None):
        attempts.append(case.id)
        if len(attempts) < 3:
            return RunResult(case.id, case.category, case.prompt, "", GradeResult(0, False, "No spoken reply was captured.", "empty"))
        return RunResult(case.id, case.category, case.prompt, "Hello.", GradeResult(80, True, "Good", "model"), duration_seconds=2.0)

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)

    cases = [TestCase(id="retry", category="general", prompt="Say hello")]
    code = voxatest_main._run(settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    out = capsys.readouterr().out

    assert code == 0
    assert len(attempts) == 3
    assert "Empty transcript for retry" in out
    assert "[1/1] PASS retry score 80 (2.0s)" in out


def test_run_warns_when_model_is_not_listed(tmp_path, monkeypatch, capsys):
    class OtherModelsClient(FakeGradingClient):
        def list_models(self):
            return ["tiny-model"]

    settings = HarnessSettings(retries=0, settle_seconds=0.0)
    monkeypatch.setattr(HarnessSettings, "make_client", lambda _self: OtherModelsClient("{}"))
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    monkeypatch.setattr(
        voxatest_main,
        "run_case",
        lambda case, _settings, _client, _whisper, _capture, _speech, log=None: RunResult(
            case.id, case.category, case.prompt, "Hello.", GradeResult(90, True, "Good", "model")
        ),
    )

    code = voxatest_main._run(settings, [TestCase(id="one", category="general", prompt="Say hello")], None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    out = capsys.readouterr().out

    assert code == 0
    assert "Warning: grading model qwen38-flash-next is not listed" in out
    assert "tiny-model" in out


def test_run_returns_130_and_writes_partial_report(tmp_path, monkeypatch):
    settings = HarnessSettings(retries=0, settle_seconds=0.0)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    monkeypatch.setattr(voxatest_main, "run_case", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    code = voxatest_main._run(
        settings,
        [TestCase(id="one", category="general", prompt="Say hello")],
        None,
        None,
        str(tmp_path / "reports"),
        str(tmp_path / "failures.md"),
    )
    payload = json.loads(next(iter((tmp_path / "reports").glob("run_*.json"))).read_text(encoding="utf-8"))

    assert code == 130
    assert payload["complete"] is False
    assert payload["results"] == []


def test_run_restores_the_previous_sigterm_handler(monkeypatch, tmp_path):
    settings = HarnessSettings(retries=0, settle_seconds=0.0)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    monkeypatch.setattr(
        voxatest_main,
        "run_case",
        lambda *_args, **_kwargs: RunResult(
            "one", "general", "Say hello", "Hello.", GradeResult(90, True, "Good", "model")
        ),
    )
    original = signal.getsignal(signal.SIGTERM)
    assert voxatest_main._run(
        settings,
        [TestCase(id="one", category="general", prompt="Say hello")],
        None,
        None,
        tmp_path,
        tmp_path / "failures.json",
    ) == 0
    assert signal.getsignal(signal.SIGTERM) is original


def test_run_stop_at_deadline_leaves_partial_report(tmp_path, monkeypatch):
    settings = HarnessSettings(retries=0, settle_seconds=0.0, run_deadline_seconds=0.05)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())

    def fake_run_case(case, _settings, _client, _whisper, _capture, _speech, log=None):
        time.sleep(0.03)
        return RunResult(case.id, case.category, case.prompt, "Hello.", GradeResult(90, True, "Good", "model"))

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)
    cases = [TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(3)]

    code = voxatest_main._run(settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    payload = json.loads(next(iter((tmp_path / "reports").glob("run_*.json"))).read_text(encoding="utf-8"))

    assert code == 0
    assert 0 < len(payload["results"]) < len(cases)
    assert payload["complete"] is False


def test_run_progress_numbers_are_absolute_when_starting_mid_file(tmp_path, monkeypatch, capsys):
    settings = HarnessSettings(retries=0, settle_seconds=0.0)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    monkeypatch.setattr(
        voxatest_main,
        "run_case",
        lambda case, *_args, **_kwargs: RunResult(
            case.id, case.category, case.prompt, "Hello.", GradeResult(90, True, "Good", "model"), duration_seconds=0.5
        ),
    )
    cases = [TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(1, 3)]

    code = voxatest_main._run(
        settings,
        cases,
        None,
        None,
        str(tmp_path / "reports"),
        str(tmp_path / "failures.md"),
        first_index=150,
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "[150/151] PASS c1 score 90 (0.5s)" in out
    assert "[151/151] PASS c2 score 90 (0.5s)" in out
    assert "[1/2]" not in out


def test_run_stops_when_voxa_stops_answering(tmp_path, monkeypatch, capsys):
    settings = HarnessSettings(retries=0, settle_seconds=0.0, max_consecutive_no_reply=5)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())
    seen: list[str] = []

    def fake_run_case(case, _settings, _client, _whisper, _capture, _speech, log=None):
        seen.append(case.id)
        return RunResult(
            case.id, case.category, case.prompt, "", GradeResult(0, False, "No spoken reply was captured.", "empty")
        )

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)
    cases = [TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(1, 9)]

    code = voxatest_main._run(settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    out = capsys.readouterr().out
    payload = json.loads(next(iter((tmp_path / "reports").glob("run_*.json"))).read_text(encoding="utf-8"))

    assert code == 3
    assert seen == ["c1", "c2", "c3", "c4", "c5"]
    assert 'No reply from Voxa for 5 cases in a row.' in out
    assert 'listening for the wake word "voxa"? Stopping.' in out
    assert payload["complete"] is False
    assert len(payload["results"]) == 5


def test_run_no_reply_counter_resets_after_a_reply(tmp_path, monkeypatch):
    settings = HarnessSettings(retries=0, settle_seconds=0.0, max_consecutive_no_reply=5)
    _stub_harness(monkeypatch, FakeCapture(), FakeSpeech())

    def fake_run_case(case, _settings, _client, _whisper, _capture, _speech, log=None):
        if case.id == "reply":
            return RunResult(case.id, case.category, case.prompt, "Hello.", GradeResult(80, True, "Good", "model"))
        return RunResult(
            case.id, case.category, case.prompt, "", GradeResult(0, False, "No spoken reply was captured.", "empty")
        )

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)
    cases = [TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(1, 5)]
    cases.append(TestCase(id="reply", category="general", prompt="Say hello"))
    cases.extend(TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(5, 9))

    code = voxatest_main._run(settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"))
    payload = json.loads(next(iter((tmp_path / "reports").glob("run_*.json"))).read_text(encoding="utf-8"))

    assert code == 1
    assert len(payload["results"]) == len(cases)
    assert payload["complete"] is True


def test_cli_start_at_slices_cases_before_limit(tmp_path, monkeypatch):
    fake_config = types.ModuleType("voxa.config")
    fake_config.ConfigStore = type(
        "FakeConfigStore",
        (),
        {"__init__": lambda _self, _path=None: None, "load": lambda _self, **_kwargs: None},
    )
    monkeypatch.setitem(sys.modules, "voxa.config", fake_config)

    cases_path = tmp_path / "cases.json"
    save_cases(cases_path, [TestCase(id=f"c{i}", category="general", prompt=f"Prompt {i}") for i in range(1, 6)])

    seen = []
    monkeypatch.setattr(voxatest_main, "_run", lambda _settings, cases, *_rest, **_kwargs: seen.append(cases) or 0)

    code = voxatest_main.main(["run", "--cases", str(cases_path), "--start-at", "3", "--limit", "2"])

    assert code == 0
    assert [case.id for case in seen[0]] == ["c3", "c4"]


def test_cli_run_deadline_retries_and_end_silence_overrides_settings():
    args = voxatest_main._parse_args(
        ["run", "--deadline-seconds", "900", "--retries", "2", "--end-silence", "1.5", "--max-silent", "0"]
    )
    settings = HarnessSettings()
    voxatest_main._apply_cli_overrides(settings, args)

    assert settings.run_deadline_seconds == 900.0
    assert settings.retries == 2
    assert settings.end_silence_seconds == 1.5
    assert settings.max_consecutive_no_reply == 0

    defaults = voxatest_main._parse_args(["run"])
    assert defaults.deadline_seconds is None
    assert defaults.retries is None
    assert defaults.end_silence is None
    assert defaults.max_silent is None
    assert defaults.start_at == 1
