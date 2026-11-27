"""#138 part 3: one AI backend contract with normalized errors.

Every client (Ollama, llama.cpp, Strata) is pushed through the same failure
scenarios by one fake HTTP server. Each must raise the SAME BackendError kind
with the SAME spoken sentence — the conversation code never branches on the
backend to explain an error — and the same healthy query through all three
clients gets the same reply.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import threading

import pytest

from voxa.backend import (
    SPOKEN_MESSAGES,
    AIBackend,
    BackendError,
    BackendStatus,
    classify_error_text,
)
from voxa.llamacpp import LlamaCppClient, LlamaCppError, StrataClient
from voxa.ollama import OllamaClient, OllamaError

ANSWER = "The sky is blue."


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    scenario: dict = {}

    def log_message(self, *_args) -> None:  # silent
        return None

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        scenario = type(self).scenario
        if self.path in ("/", "/api/version"):
            self._send(200, b"Ollama is running")
        elif self.path == "/health":
            code, body = scenario.get("health", (200, b'{"loaded": true}'))
            self._send(code, body)
        elif self.path == "/api/tags":
            if "tags" in scenario:
                code, body = scenario["tags"]
                self._send(code, body)
            else:
                self._send(200, b'{"models": [{"name": "m"}]}')
        elif self.path == "/v1/models":
            if "models" in scenario:
                code, body = scenario["models"]
                self._send(code, body)
            else:
                self._send(200, b'{"data": [{"id": "m"}]}')
        elif self.path == "/slots":
            self._send(200, b"[]")
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        scenario = type(self).scenario
        if self.path == "/api/show":
            # no advertised context limit: the client falls back to its error
            self._send(404, b'{"error": "model not found"}')
            return
        if "http" in scenario:
            code, body = scenario["http"]
            self._send(code, body)
            return
        if "stream" in scenario:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            for line in scenario["stream"]:
                self.wfile.write(line)
                self.wfile.flush()
            return
        # healthy: the same answer through whichever endpoint the client uses
        if self.path.startswith("/api/"):
            self._send(200, b'{"response": "' + ANSWER.encode() + b'", "done": true}\n')
        else:
            delta = {"choices": [{"delta": {"content": ANSWER}}]}
            self._send(200, b"data: " + _json(delta) + b"\ndata: [DONE]\n")


def _json(obj) -> bytes:
    return json.dumps(obj).encode()


@contextlib.contextmanager
def fake_server(scenario: dict):
    handler = type("ScenarioHandler", (_FakeHandler,), {"scenario": scenario})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def make_client(name: str, url: str):
    if name == "ollama":
        return OllamaClient(url)
    if name == "strata":
        return StrataClient(url)
    return LlamaCppClient(url)


def stream_lines(name: str, payload: bytes) -> bytes:
    """The same server line in whichever framing the client's API uses."""
    return payload if name == "ollama" else b"data: " + payload


CLIENT_NAMES = ("ollama", "llamacpp", "strata")

CONTEXT_BODY = (
    b'{"error": {"type": "exceed_context_size_error", '
    b'"message": "the request too long because you have not enough available '
    b'context slots for the request", "n_prompt_tokens": 9000, "n_ctx": 8192}}'
)

KIND_SCENARIOS = {
    "model_missing": lambda name: {"http": (404, b'{"error": "model \'m\' not found"}')},
    "out_of_memory": lambda name: {"http": (500, b'{"error": "unable to allocate memory"}')},
    "context_limit": lambda name: {"http": (400, CONTEXT_BODY)},
    "loading": lambda name: {"stream": [stream_lines(name, b'{"error": "the model is loading, please wait"}\n')]},
    "cancelled": lambda name: {
        "stream": [
            stream_lines(name, b'{"response": "partial"}\n')
            if name == "ollama"
            else stream_lines(name, _json({"choices": [{"delta": {"content": "partial"}}]})) + b"\n",
            stream_lines(name, b'{"error": "request cancelled by the user"}\n'),
        ]
    },
}


def ask(client, chunks: list[str]) -> str:
    return client.generate_stream(
        model="m",
        prompt="what color is the sky?",
        cancel_event=threading.Event(),
        on_chunk=chunks.append,
    )


@pytest.mark.parametrize("kind", sorted(KIND_SCENARIOS))
@pytest.mark.parametrize("name", CLIENT_NAMES)
def test_each_failure_kind_is_the_same_error_and_sentence_on_every_backend(name, kind) -> None:
    scenario = KIND_SCENARIOS[kind](name)
    with fake_server(scenario) as url:
        client = make_client(name, url)
        chunks: list[str] = []
        with pytest.raises(BackendError) as raised:
            ask(client, chunks)
        error = raised.value
        assert error.kind == kind
        # the same kind means the same spoken sentence on every backend (#138)
        assert error.message == SPOKEN_MESSAGES[kind]
        # the pre-contract handlers keep working: every client error is an OllamaError
        assert isinstance(error, OllamaError)
        if kind == "cancelled":
            assert chunks == ["partial"]


@pytest.mark.parametrize("name", CLIENT_NAMES)
def test_connection_refused_is_down_on_every_backend(name) -> None:
    client = make_client(name, "http://127.0.0.1:1")
    with pytest.raises(BackendError) as raised:
        ask(client, [])
    assert raised.value.kind == "down"
    assert raised.value.message == SPOKEN_MESSAGES["down"]


@pytest.mark.parametrize("name", CLIENT_NAMES)
def test_a_backend_that_answers_with_garbage_is_malformed(name) -> None:
    scenario = {"tags": (200, b"this is not json"), "models": (200, b"this is not json")}
    with fake_server(scenario) as url:
        client = make_client(name, url)
        with pytest.raises(BackendError) as raised:
            client.list_models()
        assert raised.value.kind == "malformed"
        assert raised.value.message == SPOKEN_MESSAGES["malformed"]


@pytest.mark.parametrize("name", CLIENT_NAMES)
def test_cancel_stops_the_query_this_client_last_started(name) -> None:
    scenario = {
        "stream": [
            stream_lines(name, b'{"response": "only this"}\n')
            if name == "ollama"
            else stream_lines(name, _json({"choices": [{"delta": {"content": "only this"}}]})) + b"\n",
            stream_lines(name, b'{"response": "must not arrive"}\n')
            if name == "ollama"
            else stream_lines(name, _json({"choices": [{"delta": {"content": "must not arrive"}}]})) + b"\n",
        ]
    }
    with fake_server(scenario) as url:
        client = make_client(name, url)
        chunks: list[str] = []
        client.generate_stream(
            model="m",
            prompt="hi",
            cancel_event=threading.Event(),
            on_chunk=lambda chunk: (chunks.append(chunk), client.cancel()),
        )
        assert "".join(chunks) == "only this"


@pytest.mark.parametrize("client,expectation", [
    (OllamaClient(), {"managed": True, "serves_one_at_a_time": False, "installs_by_pull": True}),
    (LlamaCppClient(), {"managed": True, "serves_one_at_a_time": True, "installs_by_pull": False}),
    (StrataClient(), {"managed": False, "serves_one_at_a_time": True, "installs_by_pull": False}),
])
def test_clients_follow_the_contract_and_say_how_they_behave(client, expectation) -> None:
    assert isinstance(client, AIBackend)
    caps = client.capabilities()
    for field, value in expectation.items():
        assert getattr(caps, field) is value


def test_health_is_normalized_up_and_down() -> None:
    with fake_server({}) as url:
        assert OllamaClient(url).health() == BackendStatus(up=True)
        assert LlamaCppClient(url).health().up is True
        assert LlamaCppClient(url).health().model_loaded is True
    with fake_server({"health": (503, b'{"status": "loading model"}')}) as loading_url:
        status = LlamaCppClient(loading_url).health()
        assert status.up is True and status.model_loaded is False
    down = OllamaClient("http://127.0.0.1:1").health()
    assert down.up is False


def test_the_same_query_answers_the_same_through_every_backend() -> None:
    answers = []
    chunks_by_backend = {}
    with fake_server({}) as url:
        for name in CLIENT_NAMES:
            chunks: list[str] = []
            answers.append(ask(make_client(name, url), chunks))
            chunks_by_backend[name] = chunks
    assert answers == [ANSWER, ANSWER, ANSWER]
    assert chunks_by_backend["ollama"] == chunks_by_backend["llamacpp"] == chunks_by_backend["strata"]


def test_old_callers_and_tests_keep_working_on_the_alias() -> None:
    assert issubclass(OllamaError, BackendError)
    assert issubclass(LlamaCppError, OllamaError)
    plain = OllamaError("Ollama returned HTTP 500: boom")
    assert plain.kind == "down"  # an unwrapped error still gets a normal sentence
    assert plain.message == SPOKEN_MESSAGES["down"]


@pytest.mark.parametrize("text,kind", [
    ("out of memory", "out_of_memory"),
    ("ggml could not allocate buffer", "out_of_memory"),
    ("the input exceeds the available context size limit", "context_limit"),
    ("exceed_context_size_error", "context_limit"),
    ("model 'qwen2.5:0.5b' not found", "model_missing"),
    ("request cancelled by the user", "cancelled"),
    ("the model is loading", "loading"),
    ("connection refused", "down"),
])
def test_raw_backend_text_classifies_into_one_kind(text, kind) -> None:
    assert classify_error_text(text) == kind
