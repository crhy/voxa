from __future__ import annotations

from voxa.errors import friendly_error

GPU = "The graphics card is out of memory. Another model is probably using it."
NOT_RUNNING = "The AI server isn't running."
TIMED_OUT = "The AI server took too long to answer."
NOT_INSTALLED = "That model isn't installed."
HTTP_ERROR = "The AI server hit an error."


def test_friendly_error_table() -> None:
    cases = [
        (
            'Ollama returned HTTP 500: {"error":"llama-server process has terminated: '
            'exit status 1: ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer ..."}',
            GPU,
        ),
        ("ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer", GPU),
        ("CUDA out of memory while loading the model", GPU),
        ("Vulkan0 buffer allocation failed", GPU),
        ("out of memory", GPU),
        ("Connection refused by 127.0.0.1:11434", NOT_RUNNING),
        ("Could not connect to the AI server", NOT_RUNNING),
        ("The request timed out after 120 seconds", TIMED_OUT),
        ("model llama-7b not found", NOT_INSTALLED),
        ("Model not found in catalog", NOT_INSTALLED),
        ("Ollama returned HTTP 502: bad gateway", HTTP_ERROR),
        ("HTTP 503 service unavailable", HTTP_ERROR),
        ("Something odd happened. Ignore the rest.", "Something odd happened."),
        ("", ""),
    ]
    for raw, expected in cases:
        assert friendly_error(raw) == expected


def test_never_returns_json_braces_or_over_160_chars() -> None:
    raw = "{" + "x" * 300 + "}"
    out = friendly_error(raw)
    assert "{" not in out and "}" not in out
    assert len(out) <= 160


def test_long_detail_is_cut_to_one_short_sentence() -> None:
    raw = "A" * 200 + ". trailing"
    assert friendly_error(raw) == "A" * 120
