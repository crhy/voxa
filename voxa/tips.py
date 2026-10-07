"""Short contextual tips shown in the top-left of the window.

``tips_for`` is pure: it maps a state name plus a context dict to at most
three short lines. Unknown states yield no tips. Context flags:
``media_playing`` adds a music tip, ``no_models`` points at the model
installer, and ``face_mode == "live"`` adds nothing extra.
"""

from __future__ import annotations

MAX_TIPS = 3

_STATE_TIPS: dict[str, tuple[str, ...]] = {
    "OFFLINE": (
        "Press ACTIVE to start",
        "Voxa sleeps until you activate it",
    ),
    "READY": (
        "Say “Voxa”, then your request",
        "Try: “Voxa, play some jazz”",
        "Try: “Voxa, get me the cheapest ticket to Hawaii”",
    ),
    "LISTENING": (
        "I am listening — just say it",
        "Say “never mind” to cancel",
    ),
    "THINKING": (
        "One moment — thinking it through",
        "Say more to add to my thinking",
    ),
    "SPEAKING": (
        "Talk over me to interrupt",
        "Say “Voxa, pause” to make me wait",
    ),
    "PAUSED": ("Say “Voxa” to continue",),
    "DICTATING": (
        "Say “stop dictation” to exit dictation mode",
        "Say “new line”, “comma”, “period” for punctuation",
    ),
}

_MEDIA_TIP = "Say “stop music” or “pause the music”"
_NO_MODELS_TIP = "Open Preferences → Local AI to install a model"


def tips_for(state: str, context: dict) -> list[str]:
    """At most three short tip lines for ``state`` given ``context`` flags."""
    base = list(_STATE_TIPS.get(state, ()))
    if state == "READY" and base:
        tick = int(context.get("tick", 0)) % len(base)
        base = base[tick:] + base[:tick]
    extras = []
    if context.get("media_playing"):
        extras.append(_MEDIA_TIP)
    if context.get("no_models"):
        extras.append(_NO_MODELS_TIP)
    if not base:
        return extras[:MAX_TIPS]
    return (base[:1] + extras + base[1:])[:MAX_TIPS]
