"""Gate real external actions so unattended test runs stay harmless."""

from __future__ import annotations

import os


def actions_simulated() -> bool:
    """True when VOXA_SIMULATE_ACTIONS is set to a truthy value."""
    return os.environ.get("VOXA_SIMULATE_ACTIONS", "").strip().lower() in {"1", "true", "yes", "on"}
