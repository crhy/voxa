"""Offline tests for the OpenAI-compatible connector (crhy/voxa#137, first slice).

A local http.server on 127.0.0.1 stands in for ChatGPT / a corporate gateway so
nothing here reaches the internet or a real service. The API key is read from
the environment, and the module must pull in only the standard library.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from voxa.openai_compat import OpenAICompatibleClient, OpenAICompatibleError

CANNED_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gpt-mock",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello from the gateway."}, "finish_reason": "stop"}
    ],
}


class _StubHandler(BaseHTTPRequestHandler):
    """Returns a canned chat.completions body and records what it received."""

    received: dict = {}

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        _StubHandler.received = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "body": json.loads(raw.decode("utf-8")) if raw else None,
        }
        payload = json.dumps(CANNED_REPLY).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):  # silence the default stderr logging
        return


@pytest.fixture()
def stub_server():
    _StubHandler.received = {}
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_chat_returns_text_and_sends_bearer_and_shape(stub_server, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_OPENAI_API_KEY", "secret-token-123")
    client = OpenAICompatibleClient(stub_server, "gpt-mock", "VOXA_OPENAI_API_KEY")
    answer = client.chat([{"role": "user", "content": "hi there"}])

    assert answer == "Hello from the gateway."
    received = _StubHandler.received
    assert received["path"] == "/chat/completions"
    assert received["authorization"] == "Bearer secret-token-123"
    assert received["content_type"] == "application/json"
    body = received["body"]
    assert body["model"] == "gpt-mock"
    assert body["messages"] == [{"role": "user", "content": "hi there"}]


def test_generate_stream_delivers_one_chunk(stub_server, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_OPENAI_API_KEY", "sk-corporate")
    client = OpenAICompatibleClient(stub_server, "gpt-mock", "VOXA_OPENAI_API_KEY")
    chunks: list[str] = []
    answer = client.generate_stream(
        prompt="hi there",
        cancel_event=threading.Event(),
        on_chunk=chunks.append,
    )
    assert answer == "Hello from the gateway."
    assert chunks == ["Hello from the gateway."]
    body = _StubHandler.received["body"]
    assert body["messages"] == [{"role": "user", "content": "hi there"}]


def test_missing_base_url_raises_clearly(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_OPENAI_API_KEY", "secret-token")
    client = OpenAICompatibleClient("", "gpt-mock", "VOXA_OPENAI_API_KEY")
    with pytest.raises(OpenAICompatibleError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "URL" in str(exc.value)


def test_missing_model_raises_clearly(stub_server, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_OPENAI_API_KEY", "secret-token")
    client = OpenAICompatibleClient(stub_server, "", "VOXA_OPENAI_API_KEY")
    with pytest.raises(OpenAICompatibleError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "model" in str(exc.value)


def test_missing_key_env_raises_clearly(stub_server, monkeypatch) -> None:
    client = OpenAICompatibleClient(stub_server, "gpt-mock", "")
    with pytest.raises(OpenAICompatibleError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "environment variable" in str(exc.value)
    # The message never carries a secret.
    assert "secret" not in str(exc.value).lower()


def test_unset_key_never_leaks_a_value(stub_server, monkeypatch) -> None:
    monkeypatch.delenv("VOXA_OPENAI_API_KEY", raising=False)
    client = OpenAICompatibleClient(stub_server, "gpt-mock", "VOXA_OPENAI_API_KEY")
    with pytest.raises(OpenAICompatibleError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    message = str(exc.value)
    assert "VOXA_OPENAI_API_KEY" in message  # names the variable
    assert "Bearer" not in message  # never builds an auth header here


def test_module_imports_only_stdlib() -> None:
    """The connector adds no dependency: it imports only the standard library and its own package."""
    import ast

    from voxa import openai_compat as module

    source = open(module.__file__, encoding="utf-8").read()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import within voxa is not a dependency
                continue
            if node.module and node.module != "__future__":
                imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])

    stdlib = {"json", "os", "threading", "urllib", "collections"}
    assert imported <= stdlib, f"non-stdlib imports: {imported - stdlib}"

