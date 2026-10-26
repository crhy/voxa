from voxa.hardware import ModelSuggestion, no_gpu_suggestions
from voxa.smallbrain import SMALL_BRAIN, ModelFile, path_for, row_state


def test_no_gpu_suggestions_returns_only_the_small_brain_present():
    catalog = (
        ModelSuggestion("qwen2.5:0.5b", 0.4, "small"),
        ModelSuggestion("gemma3:12b", 7.0, "big"),
        ModelSuggestion("llama3.1:8b", 4.9, "big"),
    )
    assert [m.name for m in no_gpu_suggestions(catalog)] == ["qwen2.5:0.5b"]


def test_no_gpu_suggestions_falls_back_to_first_when_neither_name_present():
    catalog = (
        ModelSuggestion("gemma3:12b", 7.0, "big"),
        ModelSuggestion("llama3.1:8b", 4.9, "big"),
    )
    assert [m.name for m in no_gpu_suggestions(catalog)] == ["gemma3:12b"]


def test_no_gpu_suggestions_keeps_both_small_brains_in_declared_order():
    catalog = (
        ModelSuggestion("llama3.1:8b", 4.9, "big"),
        ModelSuggestion("gemma3:1b", 0.8, "small"),
        ModelSuggestion("qwen2.5:1.5b", 1.0, "small"),
        ModelSuggestion("qwen2.5:0.5b", 0.4, "small"),
    )
    assert [m.name for m in no_gpu_suggestions(catalog)] == ["qwen2.5:0.5b", "gemma3:1b", "qwen2.5:1.5b"]


def test_row_state_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("voxa.engine.models_dir", lambda: tmp_path)
    assert row_state(SMALL_BRAIN, "") == "missing"


def test_row_state_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr("voxa.engine.models_dir", lambda: tmp_path)
    tiny = ModelFile("Tiny", "r", "rev", "tiny.gguf", 4, "0" * 64)
    path_for(tiny).write_bytes(b"abcd")
    assert row_state(tiny, "") == "downloaded"


def test_row_state_in_use(tmp_path, monkeypatch):
    monkeypatch.setattr("voxa.engine.models_dir", lambda: tmp_path)
    assert row_state(SMALL_BRAIN, str(path_for(SMALL_BRAIN))) == "in use"
