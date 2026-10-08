from __future__ import annotations

import pytest

from voxa.speechfit import HEAVY_ON_CPU, detect, explain, is_too_heavy, recommend_whisper, spoken_note

MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")


def display_name(model: str) -> str:
    return model[:1].upper() + model[1:]


def test_recommend_gpu_usable():
    assert recommend_whisper(True, 2, None) == "turbo"
    assert recommend_whisper(True, 16, 64.0) == "turbo"


def test_recommend_no_gpu_few_threads():
    assert recommend_whisper(False, 2, 16.0) == "tiny"


def test_recommend_no_gpu_low_ram():
    assert recommend_whisper(False, 8, 3.0) == "tiny"


def test_recommend_no_gpu_big_cpu():
    assert recommend_whisper(False, 8, 16.0) == "base"


def test_recommend_ram_unknown():
    assert recommend_whisper(False, 8, None) == "base"
    assert recommend_whisper(False, 2, None) == "tiny"


@pytest.mark.parametrize("model", MODELS)
def test_is_too_heavy(model: str):
    assert is_too_heavy(model, False) is (model in HEAVY_ON_CPU)
    assert is_too_heavy(model, True) is False


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("gpu", (True, False))
@pytest.mark.parametrize("threads", (2, 8))
def test_explain_plain_words(model: str, gpu: bool, threads: int):
    text = explain(model, gpu, threads)
    assert display_name(model) in text
    assert "_" not in text
    assert model not in text
    assert "CPU" not in text and "GPU" not in text


def test_spoken_note_exact():
    assert spoken_note("turbo", "base") == (
        "I switched my hearing to the Base model, which this computer runs comfortably."
    )


@pytest.mark.parametrize("model", MODELS)
def test_spoken_note_plain_words(model: str):
    text = spoken_note("turbo", model)
    assert display_name(model) in text
    assert "_" not in text
    assert model not in text


def test_detect_types_when_cuda_probe_raises(monkeypatch):
    def boom() -> bool:
        raise RuntimeError("no usable compute")

    monkeypatch.setattr("voxa.transcription._cuda_compute_usable", boom)
    gpu, threads, ram = detect()
    assert gpu is False
    assert isinstance(threads, int) and threads >= 1
    assert ram is None or isinstance(ram, float)
