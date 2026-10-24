from __future__ import annotations

import json
import threading
import urllib.error

import pytest

from voxa import engine, llamacpp
from voxa.config import Settings
from voxa.engine import ENGINE_PORT, ENGINE_URL, server_command
from voxa.llamacpp import LlamaCppClient, StrataClient
from voxa.server import AiServerManager


def test_engine_constants() -> None:
    assert ENGINE_PORT == 8781
    assert ENGINE_URL == "http://127.0.0.1:8781"


def test_server_command_shape() -> None:
    command = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server")
    assert command == [
        "/app/bin/llama-server",
        "--model",
        "m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8781",
        "-ngl",
        "99",
        "-c",
        "8192",
        "--jinja",
    ]


def _manager(settings: Settings, *, flatpak: bool = False) -> AiServerManager:
    return AiServerManager(
        settings,
        popen=lambda *args, **kwargs: None,
        is_reachable=lambda _url: False,
        flatpak=lambda: flatpak,
    )


def test_command_uses_bundled_server_without_flatpak_spawn(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: "/app/bin/llama-server")
    manager = _manager(Settings(ai_backend="llamacpp", llamacpp_model="m.gguf"), flatpak=True)

    command, env = manager._command("127.0.0.1", 8781)

    assert command[0] == "/app/bin/llama-server"
    assert "flatpak-spawn" not in command
    assert "-ngl" in command and "--jinja" in command
    assert env is None


def test_command_falls_back_to_host_llama_server(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: None)
    manager = _manager(Settings(ai_backend="llamacpp", llamacpp_model="m.gguf"), flatpak=False)

    command, env = manager._command("127.0.0.1", 8781)

    assert command == ["llama-server", "--model", "m.gguf", "--host", "127.0.0.1", "--port", "8781"]
    assert env is None


def test_settings_defaults_use_bundled_engine() -> None:
    settings = Settings()
    assert settings.ai_backend == "llamacpp"
    assert settings.llamacpp_url == "http://127.0.0.1:8781"


def _capture_body(monkeypatch, client) -> dict:
    captured: dict = {}

    def fake_open_url(request, timeout=0):
        captured["data"] = request.data
        raise urllib.error.URLError("stop")

    monkeypatch.setattr(llamacpp, "open_url", fake_open_url)
    with pytest.raises(llamacpp.LlamaCppError):
        client.generate_stream(
            model="m.gguf",
            prompt="hi",
            cancel_event=threading.Event(),
            on_chunk=lambda _text: None,
        )
    return json.loads(captured["data"])


def test_llamacpp_body_disables_thinking(monkeypatch) -> None:
    body = _capture_body(monkeypatch, LlamaCppClient(base_url="http://127.0.0.1:8781"))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_strata_body_keeps_thinking(monkeypatch) -> None:
    body = _capture_body(monkeypatch, StrataClient(base_url="http://127.0.0.1:8781"))
    assert "chat_template_kwargs" not in body
