"""One AI backend contract (#138 part 3).

Conversation code talks to every backend through ``AIBackend`` and every
failure through ``BackendError``: it never asks which backend answered, and
never branches on the backend to explain an error. The clients (Ollama,
llama.cpp, Strata) translate their own connection/stream failures into one
of the normalized kinds below; each kind carries one plain-worded sentence
the assistant can say out loud.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import engine

# The normalized failure kinds. Anything a backend can say goes through here.
DOWN = "down"
MODEL_MISSING = "model_missing"
LOADING = "loading"
OUT_OF_MEMORY = "out_of_memory"
CONTEXT_LIMIT = "context_limit"
CANCELLED = "cancelled"
MALFORMED = "malformed"

ERROR_KINDS = frozenset(
    {DOWN, MODEL_MISSING, LOADING, OUT_OF_MEMORY, CONTEXT_LIMIT, CANCELLED, MALFORMED}
)

# One sentence per kind, plain words, no jargon: what the user hears.
SPOKEN_MESSAGES = {
    DOWN: "The AI server is not answering, so Voxa cannot ask it.",
    MODEL_MISSING: "That model is not installed on the AI server.",
    LOADING: "The model is still loading; please give it a moment.",
    OUT_OF_MEMORY: "The AI server ran out of memory while it was answering.",
    CONTEXT_LIMIT: "The question is too long for what the model can remember.",
    CANCELLED: "The request was stopped.",
    MALFORMED: "The AI server answered with something Voxa cannot read.",
}


class BackendError(Exception):
    """Any backend failure, tagged with one normalized ``kind``.

    ``detail`` keeps the raw text for the toast/action log; ``message`` is the
    short spoken sentence for the kind.
    """

    def __init__(self, detail: str = "", kind: str = DOWN) -> None:
        super().__init__(detail)
        self.kind = kind if kind in ERROR_KINDS else DOWN
        self.detail = detail

    @property
    def message(self) -> str:
        return SPOKEN_MESSAGES[self.kind]


_OOM_MARKERS = (
    "out of memory",
    "unable to allocate",
    "failed to allocate",
    "could not allocate",
    "cuda error",
    "not enough memory",
    "no available memory",
)
_CONTEXT_MARKERS = (
    "exceed_context_size",
    "exceeds the available context",
    "exceeds the context",
    "not enough available context",
    "request too long",
    "context_length_exceeded",
    "context length exceeded",
)
_MODEL_MISSING_MARKERS = ("model not found", "no such model", "model is not installed")
_CANCELLED_MARKERS = ("cancel",)
_LOADING_MARKERS = ("loading",)


def is_memory_error(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in _OOM_MARKERS)


def is_context_error(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


def classify_error_text(text: str) -> str:
    """The BackendError kind for a backend's raw error text."""
    lowered = (text or "").casefold()
    if is_memory_error(lowered):
        return OUT_OF_MEMORY
    if is_context_error(lowered):
        return CONTEXT_LIMIT
    if "model" in lowered and "not found" in lowered:
        return MODEL_MISSING
    if any(marker in lowered for marker in _MODEL_MISSING_MARKERS):
        return MODEL_MISSING
    if any(marker in lowered for marker in _CANCELLED_MARKERS):
        return CANCELLED
    if any(marker in lowered for marker in _LOADING_MARKERS):
        return LOADING
    return DOWN


@dataclass(slots=True, frozen=True)
class BackendStatus:
    """What a backend said about itself when asked ``health()``."""

    up: bool
    model_loaded: bool | None = None
    detail: str = ""


@dataclass(slots=True, frozen=True)
class BackendCapabilities:
    """How this backend behaves; UI/server decisions read these, not names."""

    managed: bool  # Voxa starts/stops this server itself (never Strata)
    serves_one_at_a_time: bool  # a second request queues behind the first
    installs_by_pull: bool  # models are pulled/removed by API, not files


@runtime_checkable
class AIBackend(Protocol):
    """The one protocol conversation, skills and UI code use for all backends."""

    def health(self) -> BackendStatus: ...

    def list_models(self) -> list[str]: ...

    def is_model_loaded(self, model: str) -> bool | None: ...

    def generate_stream(
        self,
        *,
        model: str,
        prompt: str,
        cancel_event: threading.Event,
        on_chunk: Callable[[str], None],
        num_predict: int = 768,
        messages: list[dict] | None = None,
    ) -> str: ...

    def cancel(self) -> None: ...

    def capabilities(self) -> BackendCapabilities: ...


BACKEND_LABELS = {"ollama": "Ollama", "strata": "Strata"}


def backend_label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, "llama.cpp")


def llamacpp_models_dir(settings) -> Path:
    """The folder whose GGUF files the llama.cpp picker should offer."""
    if settings.llamacpp_model:
        return Path(settings.llamacpp_model).parent
    return engine.models_dir()


def client_for(settings) -> AIBackend:
    """The Ask AI client for whichever backend the user picked.

    The one place Voxa names backends to build one; conversation code gets an
    ``AIBackend`` and never asks which it is (#138).
    """
    backend = settings.ai_backend
    if backend == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(settings.ollama_url)
    if backend == "strata":
        from .llamacpp import StrataClient

        return StrataClient(settings.strata_url)
    from .llamacpp import LlamaCppClient

    return LlamaCppClient(settings.llamacpp_url, models_dir=str(llamacpp_models_dir(settings)))


def chat_model(settings) -> str:
    """The model name to send with a request for the chosen backend."""
    if settings.ai_backend == "llamacpp":
        if settings.llamacpp_model:
            return Path(settings.llamacpp_model).stem
        return "local"
    return settings.ollama_model
