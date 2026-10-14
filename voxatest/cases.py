from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TestCase:
    __test__ = False

    id: str
    category: str
    prompt: str
    expected_contains: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    rubric: str = ""
    expected_tool: str = ""
    expected_args: dict[str, Any] = field(default_factory=dict)
    # "stop_media": the harness stops playback itself after the case. Voxa cancels what the
    # speakers play, and the prompts come from those same speakers, so a spoken "stop the
    # radio" cannot reach her while the radio she started is on.
    after: str = ""
    # Checks of the real outcome, made after the reply passed:
    #   "file" with "file_contains" / "file_lacks": text that must (not) be in that file afterwards;
    #   "says_within" (seconds) with "says": words Voxa must speak on her own within that time,
    #   for a timer or reminder that has to go off.
    verify: dict[str, Any] = field(default_factory=dict)
    # Which of the 100 assistant capabilities (GitHub issue 102) this case is evidence for; 0 for none.
    capability: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_contains": list(self.expected_contains),
            "expected_keywords": list(self.expected_keywords),
            "rubric": self.rubric,
            "expected_tool": self.expected_tool,
            "expected_args": dict(self.expected_args),
            "after": self.after,
            "verify": dict(self.verify),
            "capability": self.capability,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> TestCase:
        if not isinstance(raw, dict):
            raise ValueError("Test case must be an object.")

        prompt = str(raw.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Test case prompt is empty.")

        expected_contains = _string_list(raw.get("expected_contains"))
        expected_keywords = _string_list(raw.get("expected_keywords"))

        case_id = str(raw.get("id", "")).strip() or f"case-{re.sub(r'[^a-zA-Z0-9]+', '-', prompt.lower())[:24]}"
        category = str(raw.get("category", "general")).strip() or "general"
        rubric = str(raw.get("rubric", "")).strip()
        expected_tool = str(raw.get("expected_tool", "")).strip()
        raw_expected_args = raw.get("expected_args", {})
        expected_args = dict(raw_expected_args) if isinstance(raw_expected_args, dict) else {}

        return cls(
            id=case_id,
            category=category,
            prompt=prompt,
            expected_contains=expected_contains,
            expected_keywords=expected_keywords,
            rubric=rubric,
            expected_tool=expected_tool,
            expected_args=expected_args,
            after=str(raw.get("after", "")).strip(),
            verify=dict(raw["verify"]) if isinstance(raw.get("verify"), dict) else {},
            capability=int(raw.get("capability") or 0) if str(raw.get("capability") or "0").isdigit() else 0,
        )


def load_cases(path: Path) -> list[TestCase]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_cases = payload.get("cases", payload.get("test_cases", []))
    else:
        raw_cases = payload

    if not isinstance(raw_cases, list):
        raise ValueError("Test case file must contain a JSON array or an object with a cases array.")

    return [TestCase.from_dict(item) for item in raw_cases]


def save_cases(path: Path, cases: list[TestCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cases": [case.to_dict() for case in cases]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
