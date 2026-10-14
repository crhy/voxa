from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


class OllamaError(RuntimeError):
    pass


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Local servers must never be reachable through an HTTP proxy: urlopen would
# happily route a request for 127.0.0.1 to whatever http_proxy is set.
_local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def open_url(request: urllib.request.Request, timeout: float):
    """Open ``request``, bypassing any proxy for loopback URLs."""
    url = getattr(request, "full_url", str(request))
    host = urllib.parse.urlparse(url).hostname or ""
    if host in _LOOPBACK_HOSTS or host.startswith("127."):
        return _local_opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Context is grown only after Ollama reports that the current allocation is
# too small.  Keeping a ceiling prevents one unusually large document from
# asking Ollama for an unreasonably large KV cache; older conversation turns
# are discarded beyond this point instead.
MAX_AUTO_CONTEXT_TOKENS = 32768


def _context_error_limits(detail: str) -> tuple[int, int] | None:
    """Extract ``(prompt tokens, allocated context)`` from nested Ollama errors."""
    value: object = detail
    for _ in range(5):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                break
            continue
        if isinstance(value, dict) and "error" in value:
            value = value["error"]
            continue
        break

    error = value if isinstance(value, dict) else {}
    searchable = detail.replace('\\"', '"').casefold()
    error_type = str(error.get("type", "")).casefold()
    message = str(error.get("message", "")).casefold()
    if not (
        "exceed_context_size" in error_type
        or "exceeds the available context" in message
        or "exceeds the available context" in searchable
    ):
        return None
    try:
        prompt_tokens = int(error["n_prompt_tokens"])
        context = int(error["n_ctx"])
    except (KeyError, TypeError, ValueError):
        prompt_match = re.search(r'"?n_prompt_tokens"?\s*:\s*(\d+)', searchable)
        context_match = re.search(r'"?n_ctx"?\s*:\s*(\d+)', searchable)
        if prompt_match is None or context_match is None:
            return None
        prompt_tokens = int(prompt_match.group(1))
        context = int(context_match.group(1))
    return (prompt_tokens, context) if prompt_tokens > 0 and context > 0 else None


def _discard_oldest_history(body: dict) -> bool:
    """Reduce prompt size while preserving the system message and newest turn."""
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        system = messages[:1] if messages[0].get("role") == "system" else []
        middle = messages[len(system) : -1]
        newest = messages[-1:]
        if middle:
            # Remove the oldest half, then avoid leaving an assistant answer
            # without the user turn that prompted it.
            remaining = middle[(len(middle) + 1) // 2 :]
            while remaining and remaining[0].get("role") == "assistant":
                remaining.pop(0)
            body["messages"] = [*system, *remaining, *newest]
            return True

        candidates = [item for item in [*system, *newest] if isinstance(item, dict)]
        if candidates:
            largest = max(candidates, key=lambda item: len(str(item.get("content", ""))))
            content = str(largest.get("content", ""))
            if len(content) > 512:
                edge = len(content) // 4
                largest["content"] = content[:edge] + "\n[… earlier text omitted …]\n" + content[-edge:]
                return True

    prompt = body.get("prompt")
    if isinstance(prompt, str) and len(prompt) > 512:
        edge = len(prompt) // 4
        body["prompt"] = prompt[:edge] + "\n[… earlier text omitted …]\n" + prompt[-edge:]
        return True
    return False


def _is_memory_error(detail: str) -> bool:
    lowered = detail.casefold()
    return any(
        marker in lowered
        for marker in ("out of memory", "unable to allocate", "failed to allocate", "cuda error")
    )


def strip_reasoning(text: str) -> str:
    """Drop a reasoning model's scratchpad from its reply.

    Qwen3, DeepSeek-R1 and other thinking models wrap their working in
    <think>...</think>. Ollama serves some GGUF builds with the opening tag
    already consumed by the chat template, leaving a bare closing tag, so
    anything ahead of the last </think> is treated as reasoning rather than
    insisting on a matched pair. Text with no closing tag is left alone.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


@dataclass(slots=True, frozen=True)
class ModelInfo:
    name: str
    size_bytes: int


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _list_models_payload(self) -> list[dict]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with open_url(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Could not connect to Ollama: {exc}") from exc
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [item for item in models if isinstance(item, dict)]

    def _model_context_limit(self, model: str) -> int | None:
        """Return the context window advertised by the model, if available."""
        payload = json.dumps({"model": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_url(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return None
        model_info = reply.get("model_info", {}) if isinstance(reply, dict) else {}
        if not isinstance(model_info, dict):
            return None
        limits: list[int] = []
        for key, value in model_info.items():
            if not str(key).endswith(".context_length"):
                continue
            try:
                limits.append(int(value))
            except (TypeError, ValueError):
                continue
        return max(limits) if limits else None

    def _larger_context(
        self, model: str, prompt_tokens: int, current_context: int, num_predict: int
    ) -> int | None:
        """Choose the smallest useful power-of-two context the model supports."""
        model_limit = self._model_context_limit(model)
        if model_limit is None or model_limit <= current_context:
            return None
        needed = prompt_tokens + min(max(num_predict, 0), 1024)
        if needed <= current_context:
            return None
        target = 1 << (needed - 1).bit_length()
        target = min(target, model_limit, MAX_AUTO_CONTEXT_TOKENS)
        return target if target > current_context else None

    def list_models(self) -> list[str]:
        names = [item.get("name", "") for item in self._list_models_payload()]
        return sorted(name for name in names if name)

    def list_models_detailed(self) -> list[ModelInfo]:
        infos = [
            ModelInfo(name=item["name"], size_bytes=int(item.get("size", 0) or 0))
            for item in self._list_models_payload()
            if item.get("name")
        ]
        return sorted(infos, key=lambda info: info.name)

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
            raise OllamaError("No Ollama model is selected.")
        if not prompt.strip():
            raise OllamaError("There is no text to send.")

        # With a message history we use the chat endpoint so the model keeps
        # the earlier turns of a conversation; without one the plain
        # generate endpoint stays the cheap single-shot path.
        chat = messages is not None
        body: dict = {
            "model": model,
            "stream": True,
            # Reasoning-capable models such as Qwen 3.5 can spend the entire
            # prediction budget in Ollama's separate ``thinking`` field and
            # return an empty spoken answer.  Voxa needs the user-facing
            # response, not a hidden scratchpad, so disable thinking for both
            # chat and single-shot voice requests.
            "think": False,
            "options": {"num_predict": num_predict},
        }
        if chat:
            body["messages"] = [dict(message) for message in messages]
        else:
            body["prompt"] = prompt.strip()
        endpoint = "/api/chat" if chat else "/api/generate"

        chunks: list[str] = []
        stream_error: str | None = None
        grew_context = False
        while True:
            payload = json.dumps(body).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}{endpoint}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with open_url(request, timeout=180) as response:
                    for raw_line in response:
                        if cancel_event.is_set():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        # Newer Ollama servers (0.1.33+) may serve streaming responses
                        # as Server-Sent Events: skip "event:" lines and strip the
                        # "data:" prefix before parsing the JSON payload.
                        if line.startswith("event:"):
                            continue
                        if line.startswith("data:"):
                            line = line.removeprefix("data:").strip()
                            if not line:
                                continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("error"):
                            # Record but do not raise mid-iteration: let the loop
                            # finish so any already-delivered chunks stay in the
                            # UI, and raise after the connection is closed.
                            stream_error = str(event["error"])
                            break
                        # Chat events carry the stream in a message object; the
                        # generate endpoint uses a plain "response" string.
                        message = event.get("message")
                        if isinstance(message, dict):
                            text = message.get("content", "")
                        else:
                            text = event.get("response", "")
                        if text:
                            chunks.append(text)
                            on_chunk(text)
                        if event.get("done"):
                            break
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                context_error = _context_error_limits(detail)
                if context_error is not None:
                    prompt_tokens, current_context = context_error
                    requested_context = int(body["options"].get("num_ctx", 0) or 0)
                    larger = self._larger_context(
                        model,
                        prompt_tokens,
                        max(current_context, requested_context),
                        num_predict,
                    )
                    if larger is not None:
                        body["options"]["num_ctx"] = larger
                        grew_context = True
                        continue
                    if _discard_oldest_history(body):
                        body["options"].pop("num_ctx", None)
                        grew_context = False
                        continue
                # A larger KV cache can fail when the model fits but the
                # machine has no remaining VRAM/RAM. Retry at the default
                # context with only recent turns before surfacing the error.
                if grew_context and _is_memory_error(detail) and _discard_oldest_history(body):
                    body["options"].pop("num_ctx", None)
                    grew_context = False
                    continue
                raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail[:300]}") from exc
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                raise OllamaError(f"Ollama request failed: {exc}") from exc

        if stream_error is not None:
            raise OllamaError(stream_error)

        return "".join(chunks).strip()

    def pull_model(
        self,
        model: str,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[str, int, int], None],
    ) -> None:
        """Pull ``model``, reporting ``(status, completed_bytes, total_bytes)`` as it downloads."""
        name = model.strip()
        if not name:
            raise OllamaError("No model name was given.")

        payload = json.dumps({"name": name, "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_url(request, timeout=1800) as response:
                for raw_line in response:
                    if cancel_event.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise OllamaError(str(event["error"]))
                    status = str(event.get("status", ""))
                    total = int(event.get("total", 0) or 0)
                    completed = int(event.get("completed", 0) or 0)
                    on_progress(status, completed, total)
                    if status == "success":
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OllamaError(f"Pulling {name} failed: {exc}") from exc

    def delete_model(self, model: str) -> None:
        name = model.strip()
        if not name:
            raise OllamaError("No model name was given.")

        payload = json.dumps({"name": name}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/delete",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        try:
            with open_url(request, timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(f"Could not delete {name}: {exc}") from exc

    def is_model_loaded(self, model: str) -> bool | None:
        """True when ``model`` is resident in memory, False when it is not, None when unknown."""
        request = urllib.request.Request(f"{self.base_url}/api/ps", method="GET")
        try:
            with open_url(request, timeout=2.0) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        models = payload.get("models", [])
        if not isinstance(models, list):
            return None
        wanted = model.strip()
        wanted_base = wanted.removesuffix(":latest")
        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model"):
                value = str(item.get(key, ""))
                if value == wanted or value.removesuffix(":latest") == wanted_base:
                    return True
        return False
