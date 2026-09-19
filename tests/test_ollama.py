from __future__ import annotations

import io
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from voxa.ollama import OllamaClient, OllamaError, open_url, strip_reasoning


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_list_models_sorts_names() -> None:
    response = FakeResponse(b'{"models":[{"name":"zeta"},{"name":"alpha"}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().list_models() == ["alpha", "zeta"]


def test_list_models_detailed_returns_sizes_sorted_by_name() -> None:
    response = FakeResponse(
        b'{"models":[{"name":"zeta","size":2000},{"name":"alpha","size":1000}]}'
    )
    with patch("voxa.ollama.open_url", return_value=response):
        infos = OllamaClient().list_models_detailed()
    assert [(info.name, info.size_bytes) for info in infos] == [("alpha", 1000), ("zeta", 2000)]


def test_pull_model_reports_progress_and_stops_on_success() -> None:
    response = FakeResponse(
        b'{"status":"pulling manifest"}\n'
        b'{"status":"downloading","completed":50,"total":100}\n'
        b'{"status":"success"}\n'
    )
    events: list[tuple[str, int, int]] = []
    with patch("voxa.ollama.open_url", return_value=response):
        OllamaClient().pull_model(
            "qwen2.5:0.5b",
            cancel_event=threading.Event(),
            on_progress=lambda status, completed, total: events.append((status, completed, total)),
        )
    assert events == [
        ("pulling manifest", 0, 0),
        ("downloading", 50, 100),
        ("success", 0, 0),
    ]


def test_pull_model_raises_on_stream_error() -> None:
    response = FakeResponse(b'{"error":"model not found"}\n')
    with patch("voxa.ollama.open_url", return_value=response), pytest.raises(OllamaError):
        OllamaClient().pull_model(
            "does-not-exist",
            cancel_event=threading.Event(),
            on_progress=lambda *_args: None,
        )


def test_pull_model_rejects_blank_name() -> None:
    with pytest.raises(OllamaError):
        OllamaClient().pull_model("  ", cancel_event=threading.Event(), on_progress=lambda *_args: None)


def test_delete_model_sends_request() -> None:
    response = FakeResponse(b"")
    with patch("voxa.ollama.open_url", return_value=response) as mocked:
        OllamaClient().delete_model("qwen2.5:0.5b")
    request = mocked.call_args[0][0]
    assert request.get_method() == "DELETE"
    assert request.full_url.endswith("/api/delete")


def test_streaming_response_calls_chunk_callback() -> None:
    response = FakeResponse(
        b'{"response":"Hello","done":false}\n'
        b'{"response":" world","done":false}\n'
        b'{"done":true}\n'
    )
    chunks: list[str] = []
    with patch("voxa.ollama.open_url", return_value=response):
        answer = OllamaClient().generate_stream(
            model="test",
            prompt="hello",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
        )
    assert answer == "Hello world"
    assert chunks == ["Hello", " world"]


def test_generate_stream_uses_chat_endpoint_when_messages_are_given() -> None:
    response = FakeResponse(
        b'{"message":{"role":"assistant","content":"Hi"},"done":false}\n'
        b'{"message":{"role":"assistant","content":" there"},"done":true,"done_reason":"stop"}\n'
    )
    chunks: list[str] = []
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "and you?"},
    ]
    with patch("voxa.ollama.open_url", return_value=response) as mocked:
        answer = OllamaClient().generate_stream(
            model="test",
            prompt="and you?",
            cancel_event=threading.Event(),
            on_chunk=chunks.append,
            messages=history,
        )
    assert answer == "Hi there"
    assert chunks == ["Hi", " there"]
    request = mocked.call_args[0][0]
    assert request.get_method() == "POST"
    assert request.full_url.endswith("/api/chat")
    body = json.loads(request.data)
    assert body["model"] == "test"
    assert body["stream"] is True
    assert body["messages"] == history
    assert "prompt" not in body


def test_strip_reasoning_removes_a_matched_think_block():
    assert strip_reasoning("<think>weighing it up</think>\n\nGNOME.") == "GNOME."


def test_strip_reasoning_handles_an_unopened_closing_tag():
    # What Ollama actually served for a Qwen3 GGUF: the opening tag was already
    # consumed by the chat template, leaving the scratchpad bare.
    reply = 'We need answer user. Simple. Need final concise.\n</think>\n\nGNOME, KDE Plasma, and XFCE.'
    assert strip_reasoning(reply) == "GNOME, KDE Plasma, and XFCE."


def test_strip_reasoning_leaves_an_ordinary_reply_alone():
    assert strip_reasoning("GNOME, KDE Plasma, and XFCE.") == "GNOME, KDE Plasma, and XFCE."


def test_strip_reasoning_keeps_the_last_answer_when_several_blocks_appear():
    assert strip_reasoning("<think>a</think>mid</think>final") == "final"


def test_open_url_bypasses_proxy_for_loopback_urls(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"models":[]}')

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("http_proxy", "http://proxy-that-does-not-exist.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy-that-does-not-exist.invalid:8080")
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/tags")
        with open_url(request, timeout=5) as response:
            assert json.load(response) == {"models": []}
    finally:
        server.shutdown()
        server.server_close()


def test_open_url_uses_plain_urlopen_for_remote_urls():
    response = FakeResponse(b"{}")
    with patch("urllib.request.urlopen", return_value=response) as mocked:
        open_url(urllib.request.Request("http://example.invalid/api/tags"), timeout=1)
    assert mocked.called
