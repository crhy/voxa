from __future__ import annotations

import io
import subprocess
import urllib.error
from unittest.mock import patch

from voxa.config import Settings
from voxa.server import AiServerManager


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeProcess:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.terminate_called = False
        self.kill_called = False
        self.wait_timeout = None

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminate_called = True
        self.running = False

    def kill(self) -> None:
        self.kill_called = True
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeout = timeout
        return 0


class FakePopen:
    def __init__(self, process: FakeProcess | None = None) -> None:
        self.process = process or FakeProcess()
        self.calls: list[tuple[list[str], dict[str, str] | None, object, object]] = []

    def __call__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        stdout: object = subprocess.DEVNULL,
        stderr: object = subprocess.DEVNULL,
    ) -> FakeProcess:
        self.calls.append((command, env, stdout, stderr))
        return self.process


def _manager(settings: Settings, *, reachable: bool = False, flatpak: bool = False):
    fake_popen = FakePopen()
    manager = AiServerManager(
        settings,
        popen=fake_popen,
        is_reachable=lambda _url: reachable,
        flatpak=lambda: flatpak,
    )
    return manager, fake_popen


def test_start_ignores_backend_url_without_host() -> None:
    manager, fake = _manager(Settings(ai_backend="ollama", ollama_url="no-host"))

    assert manager.start() is False
    assert fake.calls == []


def test_start_does_not_spawn_when_server_is_already_reachable() -> None:
    manager, fake = _manager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        reachable=True,
    )

    assert manager.start() is False
    assert fake.calls == []


def test_ollama_start_uses_command_env_and_null_stdio() -> None:
    manager, fake = _manager(Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"))

    assert manager.start() is True
    command, env, stdout, stderr = fake.calls[0]

    assert command == ["ollama", "serve"]
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434"}
    assert stdout is subprocess.DEVNULL
    assert stderr is subprocess.DEVNULL


def test_ollama_start_uses_default_port_when_url_omits_it() -> None:
    manager, fake = _manager(Settings(ai_backend="ollama", ollama_url="http://localhost"))

    assert manager.start() is True
    command, env, _stdout, _stderr = fake.calls[0]

    assert command == ["ollama", "serve"]
    assert env == {"OLLAMA_HOST": "localhost:11434"}


def test_ollama_start_prefixes_with_flatpak_spawn_in_flatpak() -> None:
    manager, fake = _manager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        flatpak=True,
    )

    assert manager.start() is True
    command, env, _stdout, _stderr = fake.calls[0]

    assert command == [
        "flatpak-spawn",
        "--host",
        "env",
        "OLLAMA_HOST=127.0.0.1:11434",
        "ollama",
        "serve",
    ]
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434"}


def test_llamacpp_start_requires_model() -> None:
    manager, fake = _manager(Settings(ai_backend="llamacpp", llamacpp_model=""))

    assert manager.start() is False
    assert fake.calls == []


def test_llamacpp_start_uses_model_url_and_port() -> None:
    manager, fake = _manager(
        Settings(
            ai_backend="llamacpp",
            llamacpp_url="http://localhost:9000",
            llamacpp_model="model.gguf",
        )
    )

    assert manager.start() is True
    command, env, _stdout, _stderr = fake.calls[0]

    assert command == [
        "llama-server",
        "--model",
        "model.gguf",
        "--host",
        "localhost",
        "--port",
        "9000",
    ]
    assert env is None


def test_llamacpp_start_prefixes_with_flatpak_spawn_in_flatpak() -> None:
    manager, fake = _manager(
        Settings(
            ai_backend="llamacpp",
            llamacpp_url="http://localhost:8080",
            llamacpp_model="model.gguf",
        ),
        flatpak=True,
    )

    assert manager.start() is True
    command, env, _stdout, _stderr = fake.calls[0]

    assert command == [
        "flatpak-spawn",
        "--host",
        "llama-server",
        "--model",
        "model.gguf",
        "--host",
        "localhost",
        "--port",
        "8080",
    ]
    assert env is None


def test_start_does_not_spawn_twice_while_own_process_is_running() -> None:
    process = FakeProcess(running=True)
    fake = FakePopen(process)
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=fake,
        is_reachable=lambda _url: False,
        flatpak=lambda: False,
    )
    manager._process = process

    assert manager.start() is True
    assert fake.calls == []
    assert manager._process is process


def test_start_replaces_dead_process() -> None:
    old_process = FakeProcess(running=False)
    fake = FakePopen()
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=fake,
        is_reachable=lambda _url: False,
        flatpak=lambda: False,
    )
    manager._process = old_process

    assert manager.start() is True
    assert fake.calls == [
        (
            ["ollama", "serve"],
            {"OLLAMA_HOST": "127.0.0.1:11434"},
            subprocess.DEVNULL,
            subprocess.DEVNULL,
        )
    ]
    assert manager._process is fake.process


def test_start_treats_reachability_probe_errors_as_unreachable() -> None:
    fake = FakePopen()

    def bad_probe(_url: str) -> bool:
        raise RuntimeError("probe failed")

    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=fake,
        is_reachable=bad_probe,
        flatpak=lambda: False,
    )

    assert manager.start() is True
    assert len(fake.calls) == 1


def test_stop_without_process_returns_false() -> None:
    manager, _fake = _manager(Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"))

    assert manager.stop() is False


def test_stop_terminates_running_process_and_clears_handle() -> None:
    process = FakeProcess(running=True)
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=FakePopen(process),
        is_reachable=lambda _url: True,
        flatpak=lambda: False,
    )
    manager._process = process

    assert manager.stop() is True
    assert process.terminate_called is True
    assert process.wait_timeout == 5
    assert manager._process is None


def test_stop_does_not_terminate_already_exited_process() -> None:
    process = FakeProcess(running=False)
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=FakePopen(process),
        is_reachable=lambda _url: True,
        flatpak=lambda: False,
    )
    manager._process = process

    assert manager.stop() is False
    assert process.terminate_called is False
    assert manager._process is None


def test_stop_kills_process_when_terminate_fails() -> None:
    class TerminateRaisesProcess(FakeProcess):
        def terminate(self) -> None:
            raise OSError("terminate failed")

    process = TerminateRaisesProcess()
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=FakePopen(process),
        is_reachable=lambda _url: True,
        flatpak=lambda: False,
    )
    manager._process = process

    assert manager.stop() is True
    assert process.kill_called is True
    assert manager._process is None


def test_restart_stops_old_process_and_starts_new_one() -> None:
    old_process = FakeProcess(running=True)
    fake = FakePopen()
    manager = AiServerManager(
        Settings(ai_backend="ollama", ollama_url="http://127.0.0.1:11434"),
        popen=fake,
        is_reachable=lambda _url: False,
        flatpak=lambda: False,
    )
    manager._process = old_process

    assert manager.restart() is True
    assert old_process.terminate_called is True
    assert manager._process is fake.process


def test_default_reachability_returns_true_for_successful_response() -> None:
    response = FakeResponse(b"ok")
    with patch("voxa.server.open_url", return_value=response):
        assert AiServerManager._default_is_reachable("http://127.0.0.1:11434") is True


def test_default_reachability_treats_http_errors_as_reachable() -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:11434",
        404,
        "not found",
        None,
        None,
    )
    with patch("voxa.server.open_url", side_effect=error):
        assert AiServerManager._default_is_reachable("http://127.0.0.1:11434") is True


def test_default_reachability_returns_false_for_connection_errors() -> None:
    with patch("voxa.server.open_url", side_effect=urllib.error.URLError("connection refused")):
        assert AiServerManager._default_is_reachable("http://127.0.0.1:11434") is False

    with patch("voxa.server.open_url", side_effect=TimeoutError()):
        assert AiServerManager._default_is_reachable("http://127.0.0.1:11434") is False

    with patch("voxa.server.open_url", side_effect=OSError("network down")):
        assert AiServerManager._default_is_reachable("http://127.0.0.1:11434") is False
