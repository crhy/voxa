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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_contains": list(self.expected_contains),
            "expected_keywords": list(self.expected_keywords),
            "rubric": self.rubric,
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

        return cls(
            id=case_id,
            category=category,
            prompt=prompt,
            expected_contains=expected_contains,
            expected_keywords=expected_keywords,
            rubric=rubric,
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
