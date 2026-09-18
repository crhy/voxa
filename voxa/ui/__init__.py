"""New-interface presentation widgets (issue #5).

MainWindow orchestrates these; backend services (audio, conversation,
dictation, ollama, speech, transcription, hardware) stay independent.

``AssistantView`` is imported lazily: it needs GTK, while ``state`` is
plain Python so tests stay headless-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voxa.ui.state import STATUS_TEXT, AssistantState

if TYPE_CHECKING:
    from voxa.ui.assistant_view import AssistantView

__all__ = ["STATUS_TEXT", "AssistantState", "AssistantView"]


def __getattr__(name: str) -> Any:
    if name == "AssistantView":
        from voxa.ui.assistant_view import AssistantView

        return AssistantView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
