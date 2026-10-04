from __future__ import annotations

import enum


class RiskLevel(enum.IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    EXTERNAL_WRITE = 2
    SENSITIVE = 3


def needs_confirmation(
    level: RiskLevel,
    granted: frozenset[str] = frozenset(),
    tool_name: str = "",
) -> bool:
    """Level 0-1: never. Level 2: unless tool_name is in `granted`. Level 3: always."""
    if level >= RiskLevel.SENSITIVE:
        return True
    if level == RiskLevel.EXTERNAL_WRITE:
        return tool_name not in granted
    return False
