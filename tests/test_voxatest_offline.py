"""V2: ``voxatest run --offline-audio`` - comprehension through Voxa's real hearing pipeline.

Every fixture here is a file and a stubbed recogniser: no microphone, no speakers, no network and no
language model. The stubs are the point of the task's test brief - the hearing pipeline
(:class:`voxa.bench.runner.ClipRunner`) runs for real on real speech (the repo's own
``voxatest/data/samples/hello-sample.wav``), and only the ASR call is replaced, so what is tested is
the render -> degrade -> pipeline -> router -> V1 grading chain, not a mock of it.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from voxatest import __main__ as voxatest_main
from voxatest import comprehension, offline
from voxatest.cases import TestCase
from voxatest.config import HarnessSettings
from voxatest.offline import SR, VARIANTS, Variant
from voxatest.report import format_results, write_reports

SAMPLE = Path(__file__).resolve().parent.parent / "voxatest" / "data" / "samples" / "hello-sample.wav"
SAMPLE_RATE = 22050
PROMPT = "open gmail"
WAKE = "voxa"


# --------------------------------------------------------------------------- stubs


def speech_samples(seconds: float | None = None) -> np.ndarray:
    """The repo's own spoken sample, resampled to the bench's 16 kHz: real speech, not a tone."""
    with wave.open(str(SAMPLE), "rb") as handle:
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    samples = np.interp(np.linspace(0, len(raw) - 1, int(len(raw) * SR / SAMPLE_RATE)),
                        np.arange(len(raw)), raw).astype(np.float32)
    if seconds is not None:
        samples = samples[: int(seconds * SR)]
    return samples


class StubASR:
    """A recogniser that answers from the script: short segment = the wake word, long = the prompt.

    It keeps the attributes and the call shape :class:`voxa.bench.runner.PinnedASR` has, because
    that is what the pipeline calls; nothing here loads a model.
    """

    def __init__(self, prompt_text: str = PROMPT, wake_text: str = "Voxa.", split_bytes: int = 60000):
        self.prompt_text = prompt_text
        self.wake_text = wake_text
        self.model_name = "stub-whisper"
        self.split_bytes = split_bytes
        self.calls = 0
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0
        self.prompts_seen: list[str] = []

    def text_for(self, pcm: bytes) -> str:
        return self.wake_text if len(pcm) < self.split_bytes else self.prompt_text

    def transcribe_words(self, pcm: bytes, hint: str = "") -> dict:
        text = self.text_for(pcm)
        self.calls += 1
        self.audio_seconds += len(pcm) / 2 / SR
        self.prompts_seen.append(hint)
        return {
            "text": text,
            "words": [{"word": word, "onset_s": 0.0, "offset_s": 0.2} for word in text.split()],
            "no_speech_prob": 0.01,
            "audio_seconds": round(len(pcm) / 2 / SR, 3),
            "asr_seconds": 0.0,
            "segments": 1 if text else 0,
        }

    def transcribe(self, pcm: bytes, language: str = "en", hint: str = "") -> str:
        return self.text_for(pcm)


@pytest.fixture
def voice(monkeypatch):
    """Voxa's natural voice chain, replaced by the repo's spoken sample (short text = short clip)."""
    calls: list[tuple] = []

    def fake(text: str, voice_name: str, rate: int, tmp_dir: Path):
        calls.append((text, voice_name, rate))
        return speech_samples(0.4 if len(text) <= 8 else 1.6), [], "stub-tts"

    monkeypatch.setattr(offline, "natural_voice", fake)
    return calls


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path


def settings_for(**changes) -> HarnessSettings:
    settings = HarnessSettings()
    settings.wake_word = WAKE
    for key, value in changes.items():
        setattr(settings, key, value)
    return settings


def case(case_id: str = "c1", prompt: str = PROMPT, *, tool: str = "open_site",
         category: str = "web", args: dict | None = None, **changes) -> TestCase:
    return TestCase(id=case_id, category=category, prompt=prompt, expected_tool=tool,
                    expected_args=dict(args or {"name": "gmail"}), **changes)


def run_one(monkeypatch, root: Path, *, asr: StubASR | None = None,
            variant: Variant = VARIANTS["clean"], the_case: TestCase | None = None,
            wake_mode: str = "separate", settings: HarnessSettings | None = None) -> tuple:
    """One case through the whole offline chain, with the voice stubbed and the real router."""
    from voxa.agent.tools import default_registry

    the_case = the_case or case()
    settings = settings or settings_for()
    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": asr or StubASR())
    result = offline.run_offline_case(
        the_case, settings, asr or StubASR(), default_registry(), variant, 0, Path(root),
        wake_word=WAKE, log=lambda *_a, **_k: None, wake_mode=wake_mode,
    )
    return result, the_case, settings


# --------------------------------------------------------------------------- the variants


def test_the_variants_are_the_ones_the_task_names():
    assert set(VARIANTS) == {"clean", "quiet", "noisy", "music", "fast", "slow", "voice2"}


def test_the_variant_numbers_are_the_ones_the_task_fixes():
    assert VARIANTS["quiet"].level_db == -15.0
    assert VARIANTS["noisy"].noise == "babble" and VARIANTS["noisy"].snr_db == 10.0
    assert VARIANTS["music"].noise == "music" and VARIANTS["music"].snr_db == 10.0
    assert (VARIANTS["fast"].rate_scale, VARIANTS["slow"].rate_scale) == (1.2, 0.85)
    assert VARIANTS["voice2"].voice and VARIANTS["voice2"].voice != "en-US-GuyNeural"
    assert VARIANTS["clean"] == Variant("clean")


def test_parse_variants_keeps_the_command_line_order_and_drops_duplicates():
    names = [variant.name for variant in offline.parse_variants("noisy, clean ,clean,fast")]
    assert names == ["noisy", "clean", "fast"]


def test_parse_variants_defaults_to_clean_alone():
    assert [variant.name for variant in offline.parse_variants("")] == ["clean"]


def test_parse_variants_names_an_unknown_variant_loudly():
    with pytest.raises(ValueError, match="telepathy"):
        offline.parse_variants("clean,telepathy")


# --------------------------------------------------------------------------- the render


def test_render_prompt_writes_and_reuses_the_cache_under_the_bench_data_root(voice, data):
    first = offline.render_prompt("open gmail", "en-US-GuyNeural", 250, data)
    assert first["cached"] is False
    assert (data / "voxatest" / f"{offline.cache_key('open gmail', 'en-US-GuyNeural', 250)}.wav").exists()
    second = offline.render_prompt("open gmail", "en-US-GuyNeural", 250, data)
    assert second["cached"] is True and second["engine"] == "stub-tts"
    assert len(voice) == 1, "the cached prompt must not be spoken twice"
    assert np.array_equal(first["samples"], second["samples"])


def test_the_cache_separates_voice_and_speaking_rate(voice, data):
    keys = {offline.render_prompt(text, voice_name, rate, data)["key"]
            for text, voice_name, rate in (("open gmail", "en-US-GuyNeural", 250),
                                           ("open gmail", "en-US-JennyNeural", 250),
                                           ("open gmail", "en-US-GuyNeural", 212))}
    assert len(keys) == 3


def test_the_render_falls_back_to_espeak_and_says_so(data, monkeypatch):
    def fail(text, voice_name, rate, tmp_dir):
        raise RuntimeError("no natural voice here")

    monkeypatch.setattr(offline, "natural_voice", fail)
    rendered = offline.render_prompt("open the file", "en-US-GuyNeural", 250, data)
    assert rendered["engine"] == "espeak-ng"
    assert rendered["samples"].size > 0


def test_the_stimulus_is_the_wake_word_in_its_own_breath_then_the_prompt(voice, data):
    samples, info = offline.build_stimulus(case(), settings_for(), VARIANTS["clean"], data, WAKE, "separate")
    assert info["wake_mode"] == "separate"
    assert info["spoken"] == "voxa. | open gmail"
    spoken = [item[0] for item in voice]
    assert spoken == ["voxa.", "open gmail"]
    assert samples.size > 1.6 * SR


def test_the_single_wake_mode_speaks_the_prompt_as_one_breath(voice, data):
    _samples, info = offline.build_stimulus(case(), settings_for(), VARIANTS["clean"], data, WAKE, "single")
    assert info["wake_mode"] == "single"
    assert info["spoken"] == "voxa, open gmail"
    assert [item[0] for item in voice] == ["voxa, open gmail"]


def test_the_fast_and_slow_variants_render_at_the_scaled_rate(voice, data):
    offline.build_stimulus(case(), settings_for(tts_rate=250), VARIANTS["fast"], data, WAKE, "separate")
    offline.build_stimulus(case(), settings_for(tts_rate=250), VARIANTS["slow"], data, WAKE, "separate")
    rates = sorted({item[2] for item in voice})
    assert rates == [212, 300], "0.85 and 1.2 of the harness rate 250"


def test_the_voice2_variant_uses_the_second_voice(voice, data):
    _samples, info = offline.build_stimulus(case(), settings_for(), VARIANTS["voice2"], data, WAKE, "separate")
    assert info["voice"] == offline.SECOND_VOICE


# --------------------------------------------------------------------------- degradation


def test_the_quiet_variant_lands_at_minus_fifteen_decibels(data):
    clean = speech_samples(1.0)
    mixed, info = offline.apply_variant(clean, VARIANTS["quiet"], np.random.default_rng(0), data)
    ratio_db = 20.0 * np.log10(float(np.sqrt(np.mean(mixed**2))) / float(np.sqrt(np.mean(clean**2))))
    assert ratio_db == pytest.approx(-15.0, abs=0.05)
    assert info["level_db"] == -15.0 and info["noise"] == "none"


def test_the_noisy_variant_mixes_real_babble_at_the_stated_snr(data):
    sources = data / "sources" / "user"
    sources.mkdir(parents=True)
    for index in range(3):
        clip = speech_samples(0.8) * (0.4 + 0.2 * index)
        offline.write_wav(sources / f"speaker{index}.wav", clip)
    clean = speech_samples(1.2)
    mixed, info = offline.apply_variant(clean, VARIANTS["noisy"], np.random.default_rng(7), data)
    assert info["noise_kind"] == "babble"
    assert info["snr_db"] == pytest.approx(10.0, abs=0.2)
    assert np.array_equal(mixed.shape, clean.shape)


def test_babble_without_a_speech_cache_says_it_used_pink_noise(data):
    clean = speech_samples(1.0)
    mixed, info = offline.apply_variant(clean, VARIANTS["noisy"], np.random.default_rng(1), data)
    assert "pink" in info["noise_kind"]
    assert info["snr_db"] == pytest.approx(10.0, abs=0.2)


def test_the_music_variant_labels_its_music_as_synthesised(data):
    clean = speech_samples(1.0)
    mixed, info = offline.apply_variant(clean, VARIANTS["music"], np.random.default_rng(3), data)
    assert info["noise_kind"] == "music (synthesised)"
    assert info["snr_db"] == pytest.approx(10.0, abs=0.2)


def test_the_mix_is_deterministic_for_one_seed_and_moves_with_another(data):
    clean = speech_samples(1.0)
    sources = data / "sources" / "user"
    sources.mkdir(parents=True)
    for index in range(2):
        offline.write_wav(sources / f"clip{index}.wav", speech_samples(1.5) * 0.5)
    first, _ = offline.apply_variant(clean, VARIANTS["noisy"], np.random.default_rng(11), data)
    again, _ = offline.apply_variant(clean, VARIANTS["noisy"], np.random.default_rng(11), data)
    other, _ = offline.apply_variant(clean, VARIANTS["noisy"], np.random.default_rng(12), data)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_mix_never_clips_and_records_the_guard(data):
    loud = speech_samples(0.6) * 3.0
    mixed, info = offline.apply_variant(loud, VARIANTS["clean"], np.random.default_rng(0), data)
    assert info["peak"] <= 0.98
    assert info["peak_guard"] is not None and info["peak_guard"] < 1.0


def test_build_clip_adds_silence_and_a_silent_reference():
    samples = speech_samples(0.5)
    mic, ref = offline.build_clip(samples)
    assert mic.dtype == np.dtype("<i2") and ref.dtype == np.float32
    assert mic.size == samples.size + int((offline.LEAD_SILENCE_SECONDS + offline.TAIL_SILENCE_SECONDS) * SR)
    assert np.all(ref == 0.0)
    assert np.all(mic[: int(0.2 * SR)] == 0) and np.all(mic[-int(1.0 * SR):] == 0)


# --------------------------------------------------------------------------- the pipeline


def test_hear_runs_the_bench_pipeline_and_delivers_the_request():
    out = offline.hear(speech_samples(1.2), StubASR(prompt_text="Voxa. open gmail"), settings_for(), "c1")
    assert out["requests"] == ["open gmail"]
    assert out["counts"]["turns"] == 1
    assert out["audio_seconds"] > 1.0


def test_hear_reports_the_asr_cost_of_the_case():
    asr = StubASR(prompt_text="Voxa. open gmail")
    out = offline.hear(speech_samples(1.2), asr, settings_for(), "c1")
    assert out["asr_calls"] == asr.calls >= 1
    assert out["asr_processing_seconds"] >= 0.0


def test_the_pipeline_settings_come_from_the_harness():
    built = offline.pipeline_settings(settings_for(silence_ms=1200, voice_threshold=900))
    assert (built.silence_ms, built.voice_threshold) == (1200, 900)
    assert built.wake_word == WAKE


# --------------------------------------------------------------------------- one case


def test_the_case_fills_the_heard_text_and_the_routed_tool(monkeypatch, voice, tmp_path):
    result, the_case, settings = run_one(monkeypatch, tmp_path)
    assert result.heard == PROMPT
    assert result.observed_tools == ["open_site"]
    assert result.action_record["tool"] == "open_site"
    assert result.action_record["args"] == {"name": "gmail"}
    assert result.expected_tool == "open_site"
    assert result.grade.method == "offline" and result.grade.passed


def test_reply_grading_is_skipped_and_marked_not_applicable(monkeypatch, voice, tmp_path):
    result, _case, _settings = run_one(monkeypatch, tmp_path)
    assert "n/a" in result.grade.notes
    assert result.transcript == ""
    assert comprehension.classify_layer(result.to_dict(), 1.0, WAKE) != "misanswered"


def test_the_case_is_ok_when_every_word_arrived_and_the_right_tool_ran(monkeypatch, voice, tmp_path):
    result, _case, settings = run_one(monkeypatch, tmp_path)
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert scored["layer"] == "ok" and scored["hearing_word_accuracy"] == 1.0
    assert scored["routed"] is True


def test_one_wrong_word_is_the_misheard_layer(monkeypatch, voice, tmp_path):
    result, _case, settings = run_one(monkeypatch, tmp_path, asr=StubASR(prompt_text="open google"))
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert scored["layer"] == "misheard"
    assert scored["hearing_word_accuracy"] == pytest.approx(0.5, abs=1e-4)
    assert scored["confusions"] == [["gmail", "google"]]


def test_the_right_words_in_the_wrong_tool_is_the_misrouted_layer(monkeypatch, voice, tmp_path):
    the_case = case(tool="open_app", args={"name": "gmail"})
    result, _case, settings = run_one(monkeypatch, tmp_path, the_case=the_case)
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert scored["layer"] == "misrouted" and scored["hearing_word_accuracy"] == 1.0


def test_a_prompt_the_pipeline_never_delivered_is_not_heard(monkeypatch, voice, tmp_path):
    result, _case, settings = run_one(monkeypatch, tmp_path, asr=StubASR(prompt_text=""))
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert result.heard == "" and scored["layer"] == "not_heard"
    assert result.grade.passed is False


def test_a_case_that_names_no_tool_is_never_a_routing_failure(monkeypatch, voice, tmp_path):
    the_case = case(prompt="remember that my cat is named mittens", tool="", args={})
    result, _case, settings = run_one(monkeypatch, tmp_path, the_case=the_case,
                                      asr=StubASR(prompt_text="remember that my cat is named mittens"))
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert scored["routed"] is None, "routing is n/a when the case names no tool"
    assert scored["layer"] == "ok" and result.grade.passed is True
    assert "remember_fact" in result.grade.notes, "what the router did is still reported"


def test_the_case_records_the_wall_time_and_how_its_audio_was_made(monkeypatch, voice, tmp_path):
    result, _case, _settings = run_one(monkeypatch, tmp_path, variant=VARIANTS["noisy"])
    block = result.offline
    assert block["wall_seconds"] > 0.0 and block["audio_seconds"] > 0.0
    assert block["variant"] == "noisy" and block["engine"] == "stub-tts"
    assert block["noise_kind"] == "pink (no babble sources)"
    assert block["wake_mode"] == "separate"


def test_arguments_are_answered_the_way_the_live_harness_answers_them(monkeypatch, voice, tmp_path):
    """Letter case is free (``voxatest/runner.py:_same_argument``): a capital is not a mishearing."""
    the_case = case(prompt="open youtube", tool="open_site", args={"name": "youtube"})
    ok, problem = offline.routing_matches(the_case, "open_site", {"name": "YouTube"})
    assert ok and problem == ""
    result, _case, settings = run_one(monkeypatch, tmp_path, the_case=the_case,
                                      asr=StubASR(prompt_text="Open YouTube"))
    scored = comprehension.score_case(result.to_dict(), settings.wake_word)
    assert scored["layer"] == "ok" and result.grade.passed is True


def test_a_number_spelled_by_the_ear_is_reported_as_an_argument_problem(monkeypatch, voice, tmp_path):
    """The finding stays visible: the router's own argument text can differ from the case's."""
    the_case = case(prompt="set a timer for 2 hours", tool="set_timer", args={"duration": "2 hours"})
    ok, problem = offline.routing_matches(the_case, "set_timer", {"duration": "two hours"})
    assert ok is False and "duration" in problem


def test_the_separate_wake_mode_is_the_default_and_single_is_measurable(monkeypatch, voice, tmp_path):
    separate, _case, settings = run_one(monkeypatch, tmp_path)
    single, _case, _settings = run_one(monkeypatch, tmp_path, wake_mode="single",
                                       asr=StubASR(prompt_text="Voxa. open gmail"))
    assert separate.heard == PROMPT
    assert single.heard == PROMPT
    assert separate.offline["wake_mode"] == "separate"
    assert single.offline["wake_mode"] == "single"


# --------------------------------------------------------------------------- a whole run


def test_run_offline_covers_every_case_in_every_variant(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    cases = [case("a"), case("b", prompt="close the window", tool="close_window", args={})]
    results, block = offline.run_offline(cases, settings_for(), offline.parse_variants("clean,fast"),
                                         0, Path(tmp_path),
                                         registry=default_registry(), log=lambda *_a, **_k: None)
    assert len(results) == 4
    assert [item.offline["variant"] for item in results] == ["clean", "clean", "fast", "fast"]
    assert block["variants"] == ["clean", "fast"] and block["seed"] == 0


def test_run_offline_reuses_one_recogniser_for_the_whole_run(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    shared = StubASR()
    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": pytest.fail("no model in tests"))
    cases = [case("a"), case("b")]
    results, _block = offline.run_offline(cases, settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          asr=shared, log=lambda *_a, **_k: None)
    assert len(results) == 2 and shared.calls > 0


def test_run_offline_skips_cases_that_need_a_real_desktop(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    cases = [case("a"), case("vm", vm_only=True)]
    results, _block = offline.run_offline(cases, settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path),
                                          registry=default_registry(), log=lambda *_a, **_k: None)
    assert [item.case_id for item in results] == ["a"]


def test_the_offline_run_never_builds_the_language_model_client(monkeypatch, voice, tmp_path):
    def no_client(self):
        raise AssertionError("offline audio must not reach the language model")

    monkeypatch.setattr(HarnessSettings, "make_client", no_client)
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    results, _block = offline.run_offline([case()], settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    assert results and results[0].grade.passed


def test_the_offline_run_never_opens_the_microphone(monkeypatch, voice, tmp_path):
    import voxa.audio as audio

    def no_mic(self, *_args, **_kwargs):
        raise AssertionError("offline audio must not open a microphone")

    monkeypatch.setattr(audio.AudioCapture, "start", no_mic)
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    results, _block = offline.run_offline([case()], settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    assert results and results[0].heard == PROMPT


def test_the_offline_block_reports_the_engines_and_wall_times(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    _results, block = offline.run_offline([case("a"), case("b")], settings_for(),
                                          offline.parse_variants("clean"), 0,
                                          Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    assert block["mode"] == "offline-audio" and block["engines"] == ["stub-tts"]
    assert block["cases"] == 2
    assert block["wall_seconds_per_case"]["median"] > 0.0
    assert block["wall_seconds_per_case"]["p95"] >= block["wall_seconds_per_case"]["median"]
    assert block["reply_grading"] == "n/a"
    assert any("microphone" in item for item in block["not_measured"])


# --------------------------------------------------------------------------- case files


def test_load_offline_cases_reads_the_command_bench_file(tmp_path):
    path = tmp_path / "command_cases.json"
    path.write_text(json.dumps({"cases": [{"id": "g1", "category": "web", "say": "open gmail",
                                           "tool": "open_site", "args": {"name": "gmail"}}]}))
    cases = offline.load_offline_cases(path)
    assert cases[0].prompt == "open gmail" and cases[0].expected_tool == "open_site"
    assert cases[0].expected_args == {"name": "gmail"}


def test_load_offline_cases_reads_spoken_test_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "1", "category": "math", "prompt": "Add 15 and 20."}]}))
    cases = offline.load_offline_cases(path)
    assert cases[0].prompt == "Add 15 and 20." and cases[0].expected_tool == ""


# --------------------------------------------------------------------------- reports


def test_the_json_report_carries_the_offline_and_comprehension_blocks(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    results, block = offline.run_offline([case("a")], settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    reports = tmp_path / "reports"
    paths = write_reports(results, reports, wake_word=WAKE, extra={"offline": block})
    payload = json.loads(Path(paths["json_path"]).read_text())
    assert payload["offline"]["mode"] == "offline-audio"
    assert payload["comprehension"]["hearing_word_accuracy"] == 1.0
    assert payload["results"][0]["offline"]["variant"] == "clean"


def test_the_text_report_shows_the_offline_section_and_the_wall_time(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    results, block = offline.run_offline([case("a")], settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    text = format_results(results, wake_word=WAKE, offline=block)
    assert "Offline audio (no speakers, no microphone)" in text
    assert "wall time per case" in text and "reply grading: n/a" in text
    assert "offline: variant clean engine stub-tts" in text


def test_the_offline_report_is_checked_like_a_bench_report(monkeypatch, voice, tmp_path):
    from voxa.agent.tools import default_registry

    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    results, block = offline.run_offline([case("a")], settings_for(), offline.parse_variants("clean"),
                                          0, Path(tmp_path), registry=default_registry(),
                                          log=lambda *_a, **_k: None)
    paths = write_reports(results, tmp_path / "reports", wake_word=WAKE, extra={"offline": block})
    report = str(paths["json_path"])
    assert comprehension.check([report, "hearing_word_accuracy>=0.9"]) == 0
    assert comprehension.check([report, "hearing_word_accuracy>=1.01"]) == 1


# --------------------------------------------------------------------------- the command line


def test_the_cli_runs_offline_and_writes_the_report(monkeypatch, voice, tmp_path):
    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR())
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [
        {"id": "1", "category": "web", "prompt": "open gmail", "expected_tool": "open_site",
         "expected_args": {"name": "gmail"}}]}))
    json_report = tmp_path / "run.json"
    code = voxatest_main.main([
        "run", "--offline-audio", "--cases", str(cases_path), "--json-report", str(json_report),
        "--reports", str(tmp_path / "reports"), "--failure-report", str(tmp_path / "failures.md"),
        "--bench-data", str(tmp_path / "bench"),
    ])
    assert code == 0
    payload = json.loads(json_report.read_text())
    assert payload["offline"]["mode"] == "offline-audio"
    assert payload["comprehension"]["layer_counts"]["ok"] == 1
    assert payload["results"][0]["heard"] == PROMPT


def test_the_cli_offline_run_exits_zero_when_the_measurement_ran(monkeypatch, voice, tmp_path):
    monkeypatch.setattr(offline, "PinnedASR", lambda model_name="": StubASR(prompt_text="open google"))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [{"id": "1", "prompt": "open gmail",
                                                 "expected_tool": "open_site"}]}))
    code = voxatest_main.main([
        "run", "--offline-audio", "--cases", str(cases_path), "--json-report", str(tmp_path / "run.json"),
        "--reports", str(tmp_path / "reports"), "--bench-data", str(tmp_path / "bench"),
    ])
    assert code == 0, "the exit code says the measurement ran; `voxatest check` is the verdict"


def test_the_cli_names_an_unknown_variant_and_stops(monkeypatch, voice, tmp_path, capsys):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [{"id": "1", "prompt": "open gmail"}]}))
    code = voxatest_main.main(["run", "--offline-audio", "--cases", str(cases_path),
                              "--variants", "clean,telepathy", "--bench-data", str(tmp_path / "bench")])
    assert code == 2
    assert "telepathy" in capsys.readouterr().err


def test_the_cli_offline_run_fails_when_the_case_file_is_empty(monkeypatch, voice, tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": []}))
    code = voxatest_main.main(["run", "--offline-audio", "--cases", str(cases_path),
                              "--bench-data", str(tmp_path / "bench")])
    assert code == 1


def test_the_spoken_run_still_uses_the_microphone_path():
    """The offline flag is opt-in: without it ``run`` keeps the live harness path."""
    cases_path = Path("voxatest/data/test_cases_smoke.json")
    parsed = voxatest_main._parse_args(["run", "--cases", str(cases_path)])
    assert getattr(parsed, "offline_audio", False) is False
    parsed_offline = voxatest_main._parse_args(["run", "--offline-audio", "--cases", str(cases_path)])
    assert parsed_offline.offline_audio is True
    assert parsed_offline.wake_mode == "separate" and parsed_offline.variants == "clean"
