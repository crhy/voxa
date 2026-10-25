"""The chosen AI backend must be asked for the right model name.

llama.cpp serves exactly one model and accepts any name, but its client rejects an empty
one, so a llama.cpp install that stores its brain in ``llamacpp_model`` (with an empty
``ollama_model``) must send the GGUF file's stem, not the empty string.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

window = pytest.importorskip("voxa.window")
MainWindow = window.MainWindow


def _settings(ai_backend: str, ollama_model: str, llamacpp_model: str) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(
            ai_backend=ai_backend,
            ollama_model=ollama_model,
            llamacpp_model=llamacpp_model,
        )
    )


def test_llamacpp_uses_gguf_stem_when_ollama_model_empty() -> None:
    host = _settings("llamacpp", "", "/home/rhy/models/Qwen3.5-0.8B-Q4_K_M.gguf")
    assert MainWindow._chat_model(host) == "Qwen3.5-0.8B-Q4_K_M"


def test_llamacpp_nothing_set_falls_back_to_local() -> None:
    host = _settings("llamacpp", "", "")
    assert MainWindow._chat_model(host) == "local"


def test_llamacpp_prefers_ollama_model_when_set() -> None:
    host = _settings("llamacpp", "qwen2.5:0.5b", "/home/rhy/models/Qwen3.5-0.8B-Q4_K_M.gguf")
    assert MainWindow._chat_model(host) == "qwen2.5:0.5b"


def test_ollama_uses_ollama_model() -> None:
    host = _settings("ollama", "qwen2.5:0.5b", "")
    assert MainWindow._chat_model(host) == "qwen2.5:0.5b"


def test_strata_uses_ollama_model() -> None:
    host = _settings("strata", "coder-iq1_m", "")
    assert MainWindow._chat_model(host) == "coder-iq1_m"


def test_window_no_longer_sends_ollama_model_directly() -> None:
    source = Path(window.__file__).read_text(encoding="utf-8")
    assert "model=self.settings.ollama_model" not in source
