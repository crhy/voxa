from __future__ import annotations

import io
import json
import threading
import urllib.error
from unittest.mock import patch

import pytest

from voxa.llamacpp import LlamaCppClient, LlamaCppError, StrataClient, detect_backend
from voxa.ollama import OllamaError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_list_models_sorts_ids() -> None:
    response = FakeResponse(b'{"data":[{"id":"zeta"},{"id":"alpha"}]}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert LlamaCppClient().list_models() == ["alpha", "zeta"]


def test_streamed_chunks_are_delivered() -> None:
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        b"data: [DONE]\n"
    )
    chunks: list[str] = []
    with patch("voxa.llamacpp.open_url", return_value=response) as mocked:
        answer = LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
        )
    assert answer == "Hello world"
    assert chunks == ["Hello", " world"]
    request = mocked.call_args[0][0]
    assert request.get_method() == "POST"
    assert request.full_url.endswith("/v1/chat/completions")
    body = json.loads(request.data)
    assert body["model"] == "test"
    assert body["stream"] is True
    assert body["max_tokens"] == 768
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_stream_uses_given_messages_and_ignores_reasoning() -> None:
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking hard"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"GNOME."}}]}\n'
        b"data: [DONE]\n"
    )
    history = [{"role": "user", "content": "which DEs?"}]
    chunks: list[str] = []
    with patch("voxa.llamacpp.open_url", return_value=response) as mocked:
        answer = LlamaCppClient().generate_stream(
            model="test",
            prompt="which DEs?",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
            messages=history,
        )
    assert answer == "GNOME."
    assert chunks == ["GNOME."]
    body = json.loads(mocked.call_args[0][0].data)
    assert body["messages"] == history


def test_stream_stops_at_done() -> None:
    response = FakeResponse(
        b"data: [DONE]\n"
        b'data: {"choices":[{"delta":{"content":"should not be read"}}]}\n'
    )
    chunks: list[str] = []
    with patch("voxa.llamacpp.open_url", return_value=response):
        answer = LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
        )
    assert answer == ""
    assert chunks == []


def test_generate_stream_stops_on_cancel() -> None:
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        b"data: [DONE]\n"
    )
    cancel_event = threading.Event()
    cancel_event.set()
    chunks: list[str] = []
    with patch("voxa.llamacpp.open_url", return_value=response):
        answer = LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=cancel_event,
            on_chunk=chunks.append,
        )
    assert answer == ""
    assert chunks == []


def test_generate_stream_raises_on_http_error() -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8080/v1/chat/completions",
        500,
        "Internal Server Error",
        None,
        io.BytesIO(b"model exploded"),
    )
    with patch("voxa.llamacpp.open_url", side_effect=error), pytest.raises(LlamaCppError):
        LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=lambda *_args: None,
        )


def test_generate_stream_timeout_mentions_busy_server() -> None:
    with patch("voxa.llamacpp.open_url", side_effect=TimeoutError("timed out")):
        with pytest.raises(LlamaCppError) as exc_info:
            LlamaCppClient().generate_stream(
                model="test",
                prompt="hello",
                cancel_event=threading.Event(),
                on_chunk=lambda *_args: None,
            )
    assert "busy" in str(exc_info.value)


def test_generate_stream_timeout_wrapped_in_urlerror() -> None:
    error = urllib.error.URLError(TimeoutError("timed out"))
    with patch("voxa.llamacpp.open_url", side_effect=error):
        with pytest.raises(LlamaCppError) as exc_info:
            LlamaCppClient().generate_stream(
                model="test",
                prompt="hello",
                cancel_event=threading.Event(),
                on_chunk=lambda *_args: None,
            )
    assert "busy" in str(exc_info.value)


def test_generate_stream_raises_on_stream_error_object() -> None:
    response = FakeResponse(b'data: {"error":{"message":"bad request"}}\n')
    with patch("voxa.llamacpp.open_url", return_value=response), pytest.raises(LlamaCppError):
        LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=lambda *_args: None,
        )


def test_generate_stream_rejects_blank_model_and_prompt() -> None:
    with pytest.raises(LlamaCppError):
        LlamaCppClient().generate_stream(
            model="",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=lambda *_args: None,
        )
    with pytest.raises(LlamaCppError):
        LlamaCppClient().generate_stream(
            model="test",
            prompt="   ",
            cancel_event=threading.Event(),
            on_chunk=lambda *_args: None,
        )


def test_llamacpp_error_is_an_ollama_error() -> None:
    assert issubclass(LlamaCppError, OllamaError)


def test_is_model_loaded_true_when_loaded_key_is_true() -> None:
    response = FakeResponse(b'{"status": "ok", "loaded": true}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert LlamaCppClient().is_model_loaded() is True


def test_is_model_loaded_false_when_loaded_key_is_false() -> None:
    response = FakeResponse(b'{"status": "ok", "loaded": false}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert LlamaCppClient().is_model_loaded() is False


def test_is_model_loaded_true_when_health_has_no_loaded_key() -> None:
    response = FakeResponse(b'{"status": "ok"}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert LlamaCppClient().is_model_loaded() is True


def test_is_model_loaded_false_on_http_503() -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8080/health",
        503,
        "Service Unavailable",
        None,
        None,
    )
    with patch("voxa.llamacpp.open_url", side_effect=error):
        assert LlamaCppClient().is_model_loaded() is False


def test_is_model_loaded_none_when_unreachable() -> None:
    with patch("voxa.llamacpp.open_url", side_effect=urllib.error.URLError("down")):
        assert LlamaCppClient().is_model_loaded() is None


def test_server_info_returns_health_payload() -> None:
    response = FakeResponse(
        b'{"status": "ok", "service": "strata", "model": "qwen3", "max_context": 65536, "loaded": true}'
    )
    with patch("voxa.llamacpp.open_url", return_value=response):
        info = StrataClient().server_info()
    assert info == {
        "status": "ok",
        "service": "strata",
        "model": "qwen3",
        "max_context": 65536,
        "loaded": True,
    }


def test_server_info_none_when_unreachable() -> None:
    with patch("voxa.llamacpp.open_url", side_effect=urllib.error.URLError("down")):
        assert StrataClient().server_info() is None


def test_detect_backend_strata_when_health_says_strata() -> None:
    response = FakeResponse(b'{"status": "ok", "service": "strata"}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert detect_backend("http://127.0.0.1:8080") == "strata"


def test_detect_backend_llamacpp_otherwise() -> None:
    response = FakeResponse(b'{"status": "ok", "service": "llama-server"}')
    with patch("voxa.llamacpp.open_url", return_value=response):
        assert detect_backend("http://127.0.0.1:8080") == "llamacpp"


def test_detect_backend_llamacpp_when_unreachable() -> None:
    with patch("voxa.llamacpp.open_url", side_effect=urllib.error.URLError("down")):
        assert detect_backend("http://127.0.0.1:8080") == "llamacpp"


def test_strata_request_carries_effort_and_budget() -> None:
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
        b"data: [DONE]\n"
    )
    chunks: list[str] = []
    with patch("voxa.llamacpp.open_url", return_value=response) as mocked:
        StrataClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
        )
    body = json.loads(mocked.call_args[0][0].data)
    assert body["reasoning_effort"] == "none"
    assert body["reasoning_budget_tokens"] == 600


def test_server_info_reads_model_from_v1_models() -> None:
    health = FakeResponse(b'{"status": "ok", "service": "strata"}')
    models = FakeResponse(b'{"data":[{"id":"qwen3.8-flash-next-coder-iq1_m","status":{"value":"loaded"}}]}')
    with patch("voxa.llamacpp.open_url", side_effect=[health, models]):
        info = StrataClient().server_info()
    assert info["model"] == "qwen3.8-flash-next-coder-iq1_m"


def test_list_models_scans_gguf_folder_without_server() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "b.gguf").touch()
        (Path(folder) / "a.gguf").touch()
        (Path(folder) / "notes.txt").touch()
        with patch("voxa.llamacpp.open_url", side_effect=urllib.error.URLError("down")):
            found = LlamaCppClient().list_models(folder)
        assert found == [str(Path(folder) / "a.gguf"), str(Path(folder) / "b.gguf")]
