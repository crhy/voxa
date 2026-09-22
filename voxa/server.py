from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from .config import Settings
from .installer import is_flatpak
from .ollama import open_url

log = logging.getLogger(__name__)


class AiServerManager:
    """Starts a local AI server only when Voxa needs one, and stops only its own process."""

    def __init__(
        self,
        settings: Settings,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        is_reachable: Callable[[str], bool] | None = None,
        flatpak: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings
        self._popen = popen
        self._is_reachable = is_reachable or self._default_is_reachable
        self._flatpak = flatpak() if flatpak is not None else is_flatpak()
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        target = self._target()
        if target is None:
            log.warning("AI server address is not configured")
            return False

        host, port, url = target
        with self._lock:
            if self._process is not None:
                if self._process.poll() is None:
                    return True
                self._process = None

            try:
                reachable = bool(self._is_reachable(url))
            except Exception:  # noqa: BLE001 - a broken reachability probe must not start a server blindly
                log.exception("could not check whether the AI server is already running")
                reachable = False

            if reachable:
                return False

            command, env = self._command(host, port)
            if command is None:
                log.warning("cannot start the selected AI server")
                return False

            try:
                self._process = self._popen(
                    command,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:  # noqa: BLE001 - startup is best-effort; assistant activation must continue
                log.exception("could not start the AI server")
                self._process = None
                return False

            return True

    def stop(self) -> bool:
        with self._lock:
            process = self._process
            if process is None:
                return False

            self._process = None
            if process.poll() is not None:
                return False

            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - OFFLINE must still clear its owned process handle
                with contextlib.suppress(Exception):
                    process.kill()
            return True

    def restart(self) -> bool:
        self.stop()
        return self.start()

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
        return host, port, f"{scheme}://{host}:{port}"

    def _command(self, host: str, port: int) -> tuple[list[str], dict[str, str] | None]:
        if self.settings.ai_backend == "ollama":
            command = ["ollama", "serve"]
            if self._flatpak:
                command = ["flatpak-spawn", "--host", "env", f"OLLAMA_HOST={host}:{port}", *command]
            return command, {"OLLAMA_HOST": f"{host}:{port}"}

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
