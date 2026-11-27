from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .backend import (
    DOWN,
    MALFORMED,
    MODEL_MISSING,
    BackendCapabilities,
    BackendStatus,
    classify_error_text,
)
from .ollama import OllamaError, open_url


class LlamaCppError(OllamaError):
    """Subclasses OllamaError so the window's handlers keep working."""


# llama-server serves one request at a time, so a question can queue behind
# another client (e.g. a coding agent using the same server). The urlopen
# timeout applies until the first byte arrives, not just connection setup, so
# generate_stream needs a generous value instead of the short API timeout.
GENERATE_TIMEOUT_SECONDS = 600


class LlamaCppClient:
    """Talks to a llama.cpp server exposing the OpenAI-compatible API."""

    disable_thinking: bool = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 5.0,
        reasoning_effort: str | None = None,
        reasoning_budget_tokens: int | None = None,
        models_dir: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.models_dir = models_dir
        self._active_cancel: threading.Event | None = None

    def health(self) -> BackendStatus:
        """llama-server's /health: 200 loaded, 503 while the model is loading."""
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return BackendStatus(up=True, model_loaded=False)
            return BackendStatus(up=True, detail=f"HTTP {exc.code}")
        except (OSError, http.client.HTTPException, ValueError) as exc:
            return BackendStatus(up=False, detail=str(exc))
        loaded = payload.get("loaded") if isinstance(payload, dict) else None
        return BackendStatus(up=True, model_loaded=bool(loaded) if loaded is not None else True)

    def capabilities(self) -> BackendCapabilities:
        """Voxa starts llama-server itself; it serves one request at a time."""
        return BackendCapabilities(managed=True, serves_one_at_a_time=True, installs_by_pull=False)

    def cancel(self) -> None:
        """Stop the query this client last started, like the Active button does."""
        if self._active_cancel is not None:
            self._active_cancel.set()

    def list_models(self, models_dir: str | None = None) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/v1/models", method="GET")
        if models_dir is None:
            models_dir = self.models_dir
        try:
            with open_url(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # No llama-server yet: the GGUF files on disk are what the user can pick.
            # OSError/HTTPException cover dropped-handshake errors (RemoteDisconnected)
            # that urllib leaves bare.
            if models_dir is not None:
                return sorted(str(path) for path in Path(models_dir).glob("*.gguf"))
            kind = MALFORMED if isinstance(exc, ValueError) else DOWN
            raise LlamaCppError(f"Could not connect to the llama.cpp server: {exc}", kind=kind) from exc
        data = payload.get("data", []) if isinstance(payload, dict) else []
        names = [item.get("id", "") for item in data if isinstance(item, dict)]
        return sorted(name for name in names if name)

    def is_busy(self) -> bool:
        """True when every server slot is already working on another request."""
        request = urllib.request.Request(f"{self.base_url}/slots", method="GET")
        try:
            with open_url(request, timeout=self.timeout) as response:
                slots = json.load(response)
        except (OSError, http.client.HTTPException, ValueError):
            return False
        if not isinstance(slots, list) or not slots:
            return False
        return all(isinstance(slot, dict) and slot.get("is_processing") for slot in slots)

    def is_model_loaded(self, model: str = "") -> bool | None:
        """True when the server has a model resident, False while it is loading, None when unknown."""
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return False
            return None
        except (OSError, http.client.HTTPException, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if "loaded" in payload:
            return bool(payload["loaded"])
        return True

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        cancel_event: threading.Event,
        on_chunk: Callable[[str], None],
        num_predict: int = 768,
        messages: list[dict] | None = None,
    ) -> str:
        if not model:
            raise LlamaCppError("No model is selected.", kind=MODEL_MISSING)
        if not prompt.strip():
            raise LlamaCppError("There is no text to send.", kind=MALFORMED)
        self._active_cancel = cancel_event

        body: dict = {
            "model": model,
            "messages": messages or [{"role": "user", "content": prompt.strip()}],
            "stream": True,
            "max_tokens": num_predict,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_budget_tokens:
            body["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        chunks: list[str] = []
        stream_error: str | None = None
        try:
            with open_url(request, timeout=GENERATE_TIMEOUT_SECONDS) as response:
                for raw_line in response:
                    if cancel_event.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    line = line.removeprefix("data:").strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        # Record but do not raise mid-iteration: let already
                        # delivered chunks stay in the UI, raise after closing.
                        stream_error = str(event["error"])
                        break
                    choices = event.get("choices") or []
                    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                    delta = choice.get("delta")
                    text = delta.get("content", "") if isinstance(delta, dict) else ""
                    if text:
                        chunks.append(text)
                        on_chunk(text)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            # One HTTP failure, one normalized kind: a 404 is the model, and
            # the server's own text says whether it ran out of memory or memory
            # of context (#138).
            kind = MODEL_MISSING if exc.code == 404 else classify_error_text(detail)
            raise LlamaCppError(f"llama.cpp server returned HTTP {exc.code}: {detail}", kind=kind) from exc
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # OSError covers URLError/TimeoutError/connection resets; the
            # http.client errors surface when the server drops the response
            # status line (RemoteDisconnected) mid-request.
            timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
            if timed_out:
                raise LlamaCppError(
                    f"llama.cpp request timed out after {GENERATE_TIMEOUT_SECONDS} seconds; "
                    "the llama.cpp server may be busy with another request.",
                    kind=DOWN,
                ) from exc
            raise LlamaCppError(f"llama.cpp request failed: {exc}", kind=DOWN) from exc

        if stream_error is not None:
            raise LlamaCppError(stream_error, kind=classify_error_text(stream_error))

        return "".join(chunks).strip()


class StrataClient(LlamaCppClient):
    """Talks to a Strata server exposing the OpenAI-compatible API."""

    disable_thinking = False

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 5.0) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            reasoning_effort="none",
            reasoning_budget_tokens=600,
        )

    def capabilities(self) -> BackendCapabilities:
        """Voxa never starts or stops Strata; it only talks to it."""
        return BackendCapabilities(managed=False, serves_one_at_a_time=True, installs_by_pull=False)

    def server_info(self) -> dict | None:
        """The parsed /health payload, with the model id filled from /v1/models when /health omits it."""
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                payload = json.load(response)
        except (OSError, http.client.HTTPException, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        info = dict(payload)
        if not info.get("model"):
            model = self._model_from_v1_models()
            if model:
                info["model"] = model
        return info

    def _model_from_v1_models(self) -> str | None:
        """The first model id from /v1/models, or None when the server is unreachable."""
        request = urllib.request.Request(f"{self.base_url}/v1/models", method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                payload = json.load(response)
        except (OSError, http.client.HTTPException, ValueError):
            return None
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
        return None


def detect_backend(url: str) -> str:
    """"strata" when /health reports service "strata", else "llamacpp"."""
    request = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
    try:
        with open_url(request, timeout=2.0) as response:
            payload = json.load(response)
    except (OSError, http.client.HTTPException, ValueError):
        return "llamacpp"
    if isinstance(payload, dict) and payload.get("service") == "strata":
        return "strata"
    return "llamacpp"
