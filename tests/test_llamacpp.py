from __future__ import annotations

import io
import json
import threading
import urllib.error
from unittest.mock import patch

import pytest

from voxa.llamacpp import LlamaCppClient, LlamaCppError
from voxa.ollama import OllamaError


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_list_models_sorts_ids() -> None:
    response = FakeResponse(b'{"data":[{"id":"zeta"},{"id":"alpha"}]}')
    with patch("urllib.request.urlopen", return_value=response):
        assert LlamaCppClient().list_models() == ["alpha", "zeta"]


def test_streamed_chunks_are_delivered() -> None:
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        b"data: [DONE]\n"
    )
    chunks: list[str] = []
    with patch("urllib.request.urlopen", return_value=response) as mocked:
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
    with patch("urllib.request.urlopen", return_value=response) as mocked:
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
    with patch("urllib.request.urlopen", return_value=response):
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
    with patch("urllib.request.urlopen", return_value=response):
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
    with patch("urllib.request.urlopen", side_effect=error), pytest.raises(LlamaCppError):
        LlamaCppClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=lambda *_args: None,
        )


def test_generate_stream_raises_on_stream_error_object() -> None:
    response = FakeResponse(b'data: {"error":{"message":"bad request"}}\n')
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(LlamaCppError):
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
