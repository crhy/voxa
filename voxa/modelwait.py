"""Poll a model lister until the AI server answers, for the window's refresh."""

from __future__ import annotations

import time
from collections.abc import Callable

from .ollama import OllamaError


def wait_for_models(
    list_models: Callable[[], list[str]],
    attempts: int = 30,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    on_waiting: Callable[[int], None] | None = None,
) -> list[str] | None:
    """Call ``list_models()`` until it returns without raising.

    Returns the list (possibly empty: the server is up and truly has no
    models) or ``None`` when every attempt failed (``OllamaError`` or
    ``OSError``) or ``should_stop()`` became true. ``on_waiting(attempt_number)``
    is called before each sleep.
    """
    for attempt in range(1, attempts + 1):
        if should_stop():
            return None
        try:
            return list_models()
        except (OllamaError, OSError):
            if attempt >= attempts or should_stop():
                return None
            if on_waiting is not None:
                on_waiting(attempt)
            sleep(delay)
    return None


def models_to_show(disk: list[str], live: list[str] | None) -> list[str] | None:
    """Decide what the model picker shows once the server had its chance to answer.

    A list that came from the running server always wins.  When the server never
    answered, the models read from disk are shown instead of "No models available"
    (crhy/voxa#27: installed models must be visible with Ollama stopped); ``None``
    is returned only when there is nothing on disk either, which is the case that
    still has to say so.  Pure: no waiting, no I/O.
    """
    if live is not None:
        return live
    return disk or None
