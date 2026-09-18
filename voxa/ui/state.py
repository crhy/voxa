"""Assistant state model (issue #5).

The UI renders itself from :class:`AssistantState` via a single
``set_assistant_state(state, detail=None)`` entry point instead of many
callbacks poking unrelated labels and buttons.
"""

from __future__ import annotations

from enum import StrEnum


class AssistantState(StrEnum):
    OFFLINE = "offline"
    READY = "ready"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WORKING = "working"
    ERROR = "error"


#: Tiny caption shown under the avatar for each state.
STATUS_TEXT: dict[AssistantState, str] = {
    AssistantState.OFFLINE: "Offline",
    AssistantState.READY: "Ready",
    AssistantState.LISTENING: "Listening…",
    AssistantState.THINKING: "Thinking…",
    AssistantState.SPEAKING: "Speaking…",
    AssistantState.WORKING: "Working…",
    AssistantState.ERROR: "Something went wrong",
}
