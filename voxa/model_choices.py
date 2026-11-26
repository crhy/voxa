"""Pure model-picker helpers, free of GTK so they can be unit-tested with a fake backend.

The model dropdown must never read "(None)" while Voxa is really using a model: these
functions decide what the picker shows and which entry is selected."""

from __future__ import annotations

NOT_INSTALLED_SUFFIX = " (not installed)"


def model_choices(configured: str, models: list[str]) -> tuple[list[str], str]:
    """The strings the model picker shows, and which one is selected.

    ``models`` is whatever the reachable backend reported (possibly empty). The
    configured model is *always* shown so the picker never reads "(None)" while
    Voxa is really using a model; when the backend does not list it, it is shown
    marked ``"(not installed)"`` rather than hidden. Returns ``(choices, selected)``.
    """
    listed = [m for m in models if m]
    configured = (configured or "").strip()
    if configured and configured not in listed:
        marked = f"{configured}{NOT_INSTALLED_SUFFIX}"
        return [marked, *listed], marked
    return listed, configured


def backend_offline_message(label: str) -> str:
    """What the picker says when a backend is down, instead of a blank "(None)"."""
    return f"{label} is not running"
