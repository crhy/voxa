from __future__ import annotations

import enum


class RiskLevel(enum.IntEnum):
    """Plain metadata about what a tool can change; Voxa never asks before doing it."""

    READ_ONLY = 0
    REVERSIBLE = 1
    EXTERNAL_WRITE = 2
    SENSITIVE = 3
