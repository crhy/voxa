from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .config import Settings
from .installer import is_flatpak
from .ollama import open_url

log = logging.getLogger(__name__)

SERVER_IDLE = "IDLE"
SERVER_STARTING = "STARTING"
SERVER_READY = "READY"
SERVER_UNAVAILABLE = "UNAVAILABLE"
SERVER_FAILED = "FAILED"
MAX_LOG_BYTES = 64 * 1024


@dataclass(slots=True)
class BackendHealth:
    status: str = SERVER_IDLE
    kind: str = ""
    owned: bool = False
    ready: bool = False
    log_path: Path | None = None
    last_error: str = ""


class AiServerManager:
    """Starts a local AI server only when Voxa needs one, and stops only its own process."""

    def __init__(
        self,
        settings: Settings,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        is_reachable: Callable[[str], bool] | None = None,
        flatpak: Callable[[], bool] | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self._popen = popen
        self._is_reachable = is_reachable or self._default_is_reachable
        self._flatpak = flatpak() if flatpak is not None else is_flatpak()
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._log_path: Path | None = None
        self._log_buffer = bytearray()
        self._log_thread: threading.Thread | None = None
        self._health = BackendHealth()

    def start(self) -> bool:
        target = self._target()
        if target is None:
            with self._lock:
                self._set_health_locked(SERVER_FAILED, ready=False, error="AI server address is not configured.")
            return False

        host, port, url = target
        old_thread: threading.Thread | None = None
        with self._lock:
            if not self._is_loopback_host(host):
                self._set_health_locked(
                    SERVER_UNAVAILABLE,
                    ready=False,
                    error="Voxa can only start managed AI servers on loopback addresses.",
                )
                return False

            if self._process is not None:
                if self._process.poll() is None:
                    self._set_health_locked(SERVER_STARTING, ready=False)
                    return True
                self._process = None
                old_thread, self._log_thread = self._log_thread, None

        if old_thread is not None:
            old_thread.join(timeout=1.0)

        with self._lock:
            try:
                reachable = bool(self._is_reachable(url))
            except Exception:  # noqa: BLE001 - a broken reachability probe must not start a server blindly
                log.exception("could not check whether the AI server is already running")
                reachable = False

            if reachable:
                self._set_health_locked(SERVER_READY, ready=True)
                return False

            command, env_overrides = self._command(host, port)
            if command is None:
                log.warning("cannot start the selected AI server")
                self._set_health_locked(
                    SERVER_FAILED,
                    ready=False,
                    error="The selected AI server needs a model path.",
                )
                return False

            try:
                sink = self._open_log_sink_locked()
                process_env = {**os.environ, **env_overrides} if env_overrides is not None else None
                self._process = self._popen(
                    command,
                    env=process_env,
                    stdout=sink,
                    stderr=subprocess.STDOUT if sink is subprocess.PIPE else sink,
                )
                if sink is subprocess.PIPE:
                    self._start_log_drain_locked(self._process)
            except Exception:  # noqa: BLE001 - startup is best-effort; assistant activation must continue
                log.exception("could not start the AI server")
                self._process = None
                self._set_health_locked(
                    SERVER_FAILED,
                    ready=False,
                    error="Could not start the AI server. Check the backend settings and server log.",
                )
                return False

            self._set_health_locked(SERVER_STARTING, ready=False)
            return True

    def stop(self) -> bool:
        with self._lock:
            process = self._process
            log_thread = self._log_thread
            self._process = None
            self._log_thread = None
            if process is None:
                self._set_health_locked(SERVER_IDLE, ready=False)
                return False

        was_running = process.poll() is None
        if was_running:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - OFFLINE must still clear its owned process handle
                with contextlib.suppress(Exception):
                    process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=1)
        if log_thread is not None:
            log_thread.join(timeout=1.0)
        with self._lock:
            self._set_health_locked(SERVER_IDLE, ready=False)
        return was_running

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def health(self) -> BackendHealth:
        with self._lock:
            return replace(self._health)

    def server_log_tail(self, max_bytes: int = 4096, max_lines: int = 20) -> str:
        if max_bytes <= 0 or max_lines <= 0:
            return ""
        with self._lock:
            data = bytes(self._log_buffer[-max_bytes:])
        return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-max_lines:])

    def wait_until_ready(
        self,
        timeout: float = 5.0,
        interval: float = 0.25,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        target = self._target()
        if target is None:
            return False

        _, _, url = target
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            if cancelled is not None and cancelled():
                return False
            with self._lock:
                process = self._process

            running = process is not None and process.poll() is None
            try:
                reachable = bool(self._is_reachable(url))
            except Exception:  # noqa: BLE001 - readiness is best-effort; OFFLINE can still recover later
                log.exception("could not check whether the AI server is ready")
                reachable = False

            if cancelled is not None and cancelled():
                return False
            if reachable:
                with self._lock:
                    self._set_health_locked(SERVER_READY, ready=True)
                return True

            if process is not None and not running:
                tail = self.server_log_tail(max_bytes=2048, max_lines=8)
                detail = f"The AI server exited with code {process.poll()} before becoming ready."
                if tail:
                    detail = f"{detail}\n{tail}"
                with self._lock:
                    self._set_health_locked(SERVER_FAILED, ready=False, error=detail)
                return False

            if timeout <= 0 or time.monotonic() >= deadline:
                break
            time.sleep(max(interval, 0.0))

        with self._lock:
            if cancelled is not None and cancelled():
                return False
            self._set_health_locked(
                SERVER_UNAVAILABLE,
                ready=False,
                error="The AI server did not become ready. Check the backend settings and server log.",
            )
        return False

    def _target(self) -> tuple[str, int, str] | None:
        if self.settings.ai_backend == "ollama":
            url = self.settings.ollama_url
            default_port = 11434
        else:
            url = self.settings.llamacpp_url
            default_port = 8080

        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError:
            return None

        host = parsed.hostname or ""
        if not host:
            return None

        port = parsed.port or default_port
        scheme = parsed.scheme or "http"
        return host, port, f"{scheme}://{self._host_port(host, port)}"

    def _command(self, host: str, port: int) -> tuple[list[str], dict[str, str] | None]:
        if self.settings.ai_backend == "ollama":
            command = ["ollama", "serve"]
            listen_address = self._host_port(host, port)
            if self._flatpak:
                command = ["flatpak-spawn", "--host", "env", f"OLLAMA_HOST={listen_address}", *command]
            return command, {"OLLAMA_HOST": listen_address}

        model = self.settings.llamacpp_model
        if not model:
            return None, None

        command = ["llama-server", "--model", model, "--host", host, "--port", str(port)]
        if self._flatpak:
            command = ["flatpak-spawn", "--host", *command]
        return command, None

    @staticmethod
    def _default_is_reachable(url: str) -> bool:
        request = urllib.request.Request(url, method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                response.read(1)
            return True
        except urllib.error.HTTPError:
            # Even a non-2xx response means a server is listening on that port.
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        normalized = host.strip().casefold()
        return normalized in {
            "127.0.0.1",
            "localhost",
            "::1",
            "0:0:0:0:0:0:0:1",
        } or normalized.startswith("127.")

    @staticmethod
    def _host_port(host: str, port: int) -> str:
        """Format a host and port without producing an invalid IPv6 URL."""
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    def _open_log_sink_locked(self) -> object:
        if self._log_dir is None:
            self._log_path = None
            self._log_buffer.clear()
            return subprocess.DEVNULL

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self._log_dir / f"{self.settings.ai_backend}.log"
            self._log_buffer.clear()
            self._log_path.write_bytes(b"")
            return subprocess.PIPE
        except OSError:
            log.exception("could not open the bounded AI server log")
            self._log_path = None
            return subprocess.DEVNULL

    def _start_log_drain_locked(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is None:
            return
        thread = threading.Thread(
            target=self._drain_log,
            args=(stream,),
            name="voxa-ai-server-log",
            daemon=True,
        )
        self._log_thread = thread
        thread.start()

    def _drain_log(self, stream: object) -> None:
        try:
            while chunk := stream.read(4096):
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                with self._lock:
                    self._log_buffer.extend(chunk)
                    if len(self._log_buffer) > MAX_LOG_BYTES:
                        del self._log_buffer[:-MAX_LOG_BYTES]
                    path = self._log_path
                    snapshot = bytes(self._log_buffer)
                if path is not None:
                    try:
                        path.write_bytes(snapshot)
                    except OSError:
                        log.exception("could not update the bounded AI server log")
                        with self._lock:
                            self._log_path = None
        except (OSError, ValueError):
            log.debug("AI server log stream closed", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def _set_health_locked(self, status: str, *, ready: bool, error: str = "") -> None:
        self._health = BackendHealth(
            status=status,
            kind=self.settings.ai_backend,
            owned=self._process is not None and self._process.poll() is None,
            ready=ready,
            log_path=self._log_path,
            last_error=error,
        )
