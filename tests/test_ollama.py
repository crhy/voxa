from __future__ import annotations

import io
import json
import threading
import urllib.error
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


def _http_error(detail: dict | str, code: int = 400) -> urllib.error.HTTPError:
    raw = json.dumps(detail) if isinstance(detail, dict) else detail
    return urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/chat",
        code,
        "request failed",
        {},
        io.BytesIO(raw.encode()),
    )


def _context_error(prompt_tokens: int = 4358, context: int = 4096) -> urllib.error.HTTPError:
    nested = {
        "error": {
            "code": 400,
            "message": f"request ({prompt_tokens} tokens) exceeds the available context size ({context} tokens)",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": prompt_tokens,
            "n_ctx": context,
        }
    }
    # Some Ollama-compatible runners return the useful JSON as a string in
    # an outer error object, which is the form seen in the live failure.
    return _http_error({"error": json.dumps(nested)})


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


def test_is_model_loaded_true_when_running() -> None:
    response = FakeResponse(b'{"models":[{"name":"qwen2.5:7b","size":1}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("qwen2.5:7b") is True


def test_is_model_loaded_matches_model_key_too() -> None:
    response = FakeResponse(b'{"models":[{"model":"mistral:latest"}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("mistral:latest") is True


def test_is_model_loaded_treats_latest_as_equal() -> None:
    response = FakeResponse(b'{"models":[{"name":"llama3:latest"}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("llama3") is True
    response = FakeResponse(b'{"models":[{"name":"llama3"}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("llama3:latest") is True


def test_is_model_loaded_false_when_absent() -> None:
    response = FakeResponse(b'{"models":[{"name":"llama3"},{"name":"mistral"}]}')
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("qwen2.5:7b") is False


def test_is_model_loaded_none_when_unreachable() -> None:
    error = urllib.error.URLError("connection refused")
    with patch("voxa.ollama.open_url", side_effect=error):
        assert OllamaClient().is_model_loaded("qwen2.5:7b") is None


def test_is_model_loaded_none_when_reply_is_not_json() -> None:
    response = FakeResponse(b"not json at all")
    with patch("voxa.ollama.open_url", return_value=response):
        assert OllamaClient().is_model_loaded("qwen2.5:7b") is None


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
    assert body["think"] is False
    assert body["messages"] == history
    assert "prompt" not in body


def test_generate_stream_disables_thinking_for_single_shot_requests() -> None:
    response = FakeResponse(b'{"response":"Four.","done":true}\n')
    with patch("voxa.ollama.open_url", return_value=response) as mocked:
        answer = OllamaClient().generate_stream(
            model="qwen3.5:8b",
            prompt="What is two plus two?",
            cancel_event=threading.Event(),
            on_chunk=lambda _chunk: None,
        )

    assert answer == "Four."
    body = json.loads(mocked.call_args[0][0].data)
    assert body["think"] is False
    assert body["prompt"] == "What is two plus two?"


def test_generate_stream_grows_context_when_model_supports_it() -> None:
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "an earlier question"},
        {"role": "assistant", "content": "an earlier answer"},
        {"role": "user", "content": "read me a passage"},
    ]
    model_info = FakeResponse(b'{"model_info":{"qwen35.context_length":262144}}')
    answer = FakeResponse(b'{"message":{"content":"Once upon a midnight dreary."},"done":true}\n')
    with patch(
        "voxa.ollama.open_url", side_effect=[_context_error(), model_info, answer]
    ) as mocked:
        result = OllamaClient().generate_stream(
            model="qwen3.5:0.8b",
            prompt="read me a passage",
            cancel_event=threading.Event(),
            on_chunk=lambda _chunk: None,
            messages=history,
        )

    assert result == "Once upon a midnight dreary."
    chat_requests = [
        call.args[0] for call in mocked.call_args_list if call.args[0].full_url.endswith("/api/chat")
    ]
    assert len(chat_requests) == 2
    assert "num_ctx" not in json.loads(chat_requests[0].data)["options"]
    assert json.loads(chat_requests[1].data)["options"]["num_ctx"] == 8192


def test_generate_stream_discards_old_history_when_model_cannot_grow() -> None:
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "old question one"},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old question two"},
        {"role": "assistant", "content": "old answer two"},
        {"role": "user", "content": "new question"},
    ]
    model_info = FakeResponse(b'{"model_info":{"small.context_length":4096}}')
    answer = FakeResponse(b'{"message":{"content":"A recent answer."},"done":true}\n')
    with patch(
        "voxa.ollama.open_url", side_effect=[_context_error(), model_info, answer]
    ) as mocked:
        result = OllamaClient().generate_stream(
            model="small",
            prompt="new question",
            cancel_event=threading.Event(),
            on_chunk=lambda _chunk: None,
            messages=history,
        )

    assert result == "A recent answer."
    chat_requests = [
        call.args[0] for call in mocked.call_args_list if call.args[0].full_url.endswith("/api/chat")
    ]
    retried = json.loads(chat_requests[1].data)
    assert retried["messages"][0] == history[0]
    assert retried["messages"][-1] == history[-1]
    assert len(retried["messages"]) < len(history)
    assert "num_ctx" not in retried["options"]


def test_generate_stream_falls_back_when_larger_context_runs_out_of_memory() -> None:
    history = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    model_info = FakeResponse(b'{"model_info":{"large.context_length":32768}}')
    oom = _http_error({"error": "CUDA out of memory while allocating KV cache"}, code=500)
    answer = FakeResponse(b'{"message":{"content":"Recovered."},"done":true}\n')
    with patch(
        "voxa.ollama.open_url", side_effect=[_context_error(), model_info, oom, answer]
    ) as mocked:
        result = OllamaClient().generate_stream(
            model="large",
            prompt="new question",
            cancel_event=threading.Event(),
            on_chunk=lambda _chunk: None,
            messages=history,
        )

    assert result == "Recovered."
    chat_requests = [
        call.args[0] for call in mocked.call_args_list if call.args[0].full_url.endswith("/api/chat")
    ]
    assert json.loads(chat_requests[1].data)["options"]["num_ctx"] == 8192
    final = json.loads(chat_requests[2].data)
    assert "num_ctx" not in final["options"]
    assert len(final["messages"]) < len(history)


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
