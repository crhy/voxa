import json

from voxa.config import ConfigStore, Settings
from voxa.speechfit import BENCHMARK_CLIP, choose, clip_pcm, measure_rtf, predicted_rtf, summary


def test_clip_exists_and_is_16k_mono_16bit_5_to_12_seconds():
    assert BENCHMARK_CLIP.exists()
    pcm, seconds = clip_pcm()
    assert 5.0 <= seconds <= 12.0
    assert len(pcm) == int(seconds * 16000) * 2


def test_measure_rtf_uses_fastest_run_with_fake_clock():
    ticks = iter([0.0, 4.0, 5.0, 6.0])  # first run 4 s, second run 1 s
    calls = []

    def transcribe(pcm):
        calls.append(pcm)
        return ""

    rtf = measure_rtf(transcribe, b"x", 2.0, runs=2, clock=lambda: next(ticks))
    assert calls == [b"x", b"x"]
    assert rtf == 0.5  # fastest run (1 s) / 2 s


def test_predicted_rtf_prefers_measured_values():
    predictions = predicted_rtf(0.1, {"base": 0.9})
    assert predictions["base"] == 0.9
    assert predictions["tiny"] == 0.1
    assert predictions["small"] == 0.1 * 6.3


def test_choose_picks_most_accurate_under_budget():
    assert choose(0.02, {}, gpu_usable=False) == "turbo"
    assert choose(0.05, {}, gpu_usable=False) == "small"
    assert choose(0.2, {}, gpu_usable=False) == "base"
    assert choose(0.4, {}, gpu_usable=False) == "tiny"
    assert choose(0.1, {"base": 0.9}, gpu_usable=False) == "tiny"


def test_choose_on_gpu_returns_recommendation():
    assert choose(9.0, {}, gpu_usable=True) == "turbo"


def test_summary_wording():
    assert summary("base", 0.2) == "Base transcribes 10 seconds of speech in about 2 seconds on this computer."


def test_settings_round_trip_with_new_fields(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    settings = Settings()
    settings.whisper_rtf = {"tiny": 0.12}
    settings.whisper_benchmarked = True
    store.save(settings)
    loaded = store.load()
    assert loaded.whisper_rtf == {"tiny": 0.12}
    assert loaded.whisper_benchmarked is True


def test_settings_load_old_file_without_new_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"whisper_model": "small", "language": "en"}) + "\n", encoding="utf-8")
    store = ConfigStore(path)
    loaded = store.load()
    assert loaded.whisper_model == "small"
    assert loaded.whisper_rtf == {}
    assert loaded.whisper_benchmarked is False
