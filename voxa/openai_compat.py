"""A provider-agnostic connector for any OpenAI-compatible chat-completions gateway.

ChatGPT-compatible gateways and most "corporate AI" endpoints already speak the
OpenAI ``/chat/completions`` shape, so one connector lets Voxa point at any of
them. The API key is never stored in config or the repo: only the *name* of an
environment variable is kept, and the secret is read from ``os.environ`` at
request time. This module is exercised offline against a local stub and imports
only the standard library.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable

from .ollama import open_url


class OpenAICompatibleError(RuntimeError):
    """A problem talking to an OpenAI-compatible endpoint. Never carries the key."""


# Keeping the path as its own constant lets the connector and its checks agree
# on the exact endpoint without the base URL bleeding into a literal.
CHAT_COMPLETIONS_PATH = "/chat/completions"

# A chat answer can take a while to arrive on a distant gateway; the short API
# timeout is only used for the reachability probes.
GENERATE_TIMEOUT_SECONDS = 600


def _content_from_payload(payload: object) -> str:
    """The assistant text from a chat.completions body (choices[0].message.content)."""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


class OpenAICompatibleClient:
    """Talks to an OpenAI-compatible endpoint exposing ``/chat/completions``."""

    def __init__(self, base_url: str, model: str, api_key_env: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def _missing_setting(self) -> str | None:
        """A clear message for a missing base_url/model/key_env, or None when complete."""
        if not self.base_url:
            return "No OpenAI-compatible endpoint URL is set."
        if not self.model:
            return "No OpenAI-compatible model is selected."
        if not self.api_key_env:
            return "No environment variable is named for the OpenAI-compatible key."
        return None

    def _api_key(self) -> str:
        """Read the secret from the named environment variable at request time."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            # Names the variable only; the value never reaches the message.
            raise OpenAICompatibleError(
                f"The environment variable {self.api_key_env!r} is not set; "
                "an OpenAI-compatible endpoint needs an API key."
            )
        return key

    def _request(self, body: dict) -> urllib.request.Request:
        payload = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}{CHAT_COMPLETIONS_PATH}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
            },
            method="POST",
        )

    def _post(self, body: dict, timeout: float) -> object:
        request = self._request(body)
        try:
            with open_url(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # The server's body is the only detail shown; the Authorization
            # header we sent is never part of the response.
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise OpenAICompatibleError(
                f"OpenAI-compatible endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise OpenAICompatibleError(f"OpenAI-compatible request failed: {exc}") from exc

    def chat(self, messages: list[dict], *, num_predict: int = 768) -> str:
        """Send ``messages`` and return the assistant's text answer."""
        missing = self._missing_setting()
        if missing:
            raise OpenAICompatibleError(missing)
        if not messages:
            raise OpenAICompatibleError("There is no text to send.")
        body = {"model": self.model, "messages": messages, "max_tokens": num_predict}
        return _content_from_payload(self._post(body, timeout=self.timeout))

    def generate_stream(
        self,
        *,
        model: str = "",
        prompt: str = "",
        cancel_event: threading.Event | None = None,
        on_chunk: Callable[[str], None] | None = None,
        num_predict: int = 768,
        messages: list[dict] | None = None,
    ) -> str:
        """Ask the endpoint and deliver the answer, mirroring the other brain clients.

        Uses ``self.model`` (the configured model) and the OpenAI message shape;
        with no history the single ``prompt`` becomes the one user turn.
        """
        missing = self._missing_setting()
        if missing:
            raise OpenAICompatibleError(missing)
        chat_messages = messages if messages is not None else [{"role": "user", "content": (prompt or "").strip()}]
        if not chat_messages:
            raise OpenAICompatibleError("There is no text to send.")
        body = {"model": self.model, "messages": chat_messages, "max_tokens": num_predict}
        answer = _content_from_payload(self._post(body, timeout=GENERATE_TIMEOUT_SECONDS))
        if cancel_event is not None and cancel_event.is_set():
            return ""
        if answer and on_chunk is not None:
            on_chunk(answer)
        return answer
