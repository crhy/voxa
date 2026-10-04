from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    speech: str
    detail: str = ""

    @classmethod
    def success(cls, speech: str, detail: str = "") -> ToolResult:
        return cls(ok=True, speech=speech, detail=detail)

    @classmethod
    def failure(cls, speech: str, detail: str = "") -> ToolResult:
        return cls(ok=False, speech=speech, detail=detail)
