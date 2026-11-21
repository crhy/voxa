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
    # Steps run on the host before speaking: copy a sample file, open it, press keys, remove it, wait.
    setup: list[dict[str, Any]] = field(default_factory=list)
    # True when the case can only run in the test VM (needs a real desktop, xdotool, a GUI editor).
    vm_only: bool = False
    # --- V3 hearing suite (voxatest/hearing.py). Empty/None on every ordinary case.
    # "reply": the words Voxa is reading out while the user speaks. The hearing cells are defined
    # by what she is saying, so a case that expects a turn decision names it here.
    reply: str = ""
    # "barge_at": where the user starts. A number below 1 is a FRACTION of Voxa's reply, a number
    # of 1 or more is ABSOLUTE SECONDS from the start of her reply, and a name is one of the
    # handoff 8.2 overlap positions: onset, first_syllable, middle, last_word, after_cancel,
    # after_reply. Empty means "the user speaks on her own, with nothing playing".
    barge_at: Any = None
    # "level_db": the user's level AT THE MICROPHONE relative to Voxa's echo (a ratio in dB, not a
    # volume slider). None is the suite default (+5 dB).
    level_db: float | None = None
    # "voice": the test user's TTS voice for this case (empty = the harness test voice).
    voice: str = ""
    # "pause_at" / "pause_s": a deliberate mid-sentence pause - the fraction of the user's
    # utterance it falls at, and how long it lasts (handoff 8.2 "User pacing").
    pause_at: float | None = None
    pause_s: float | None = None
    # "expect": the turn decision the case asks for - "yield" (she must stop for the user),
    # "keep" (she must keep speaking: a back-channel, her own reply with nobody talking),
    # "answer" (the user's words must reach her whole). Empty = not a hearing case.
    expect: str = ""
    # "cell": the declared test cell. "normal" is the declared normal-use range; the difficult
    # cells ("quiet", "far", "loud") are reported apart, never averaged into the headline.
    cell: str = ""

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
            "setup": [dict(step) for step in self.setup],
            "vm_only": self.vm_only,
            "reply": self.reply,
            "barge_at": self.barge_at,
            "level_db": self.level_db,
            "voice": self.voice,
            "pause_at": self.pause_at,
            "pause_s": self.pause_s,
            "expect": self.expect,
            "cell": self.cell,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> TestCase:
        if not isinstance(raw, dict):
            raise ValueError("Test case must be an object.")

        prompt = str(raw.get("prompt", "")).strip()
        # A Voxa-only hearing cell (``expect: "keep"``, nobody talking) has no prompt by design:
        # the whole case is the reply she must get through without stopping.
        if not prompt and not str(raw.get("expect", "")).strip():
            raise ValueError("Test case prompt is empty.")

        expected_contains = _string_list(raw.get("expected_contains"))
        expected_keywords = _string_list(raw.get("expected_keywords"))

        case_id = str(raw.get("id", "")).strip() or f"case-{re.sub(r'[^a-zA-Z0-9]+', '-', prompt.lower())[:24]}"
        category = str(raw.get("category", "general")).strip() or "general"
        rubric = str(raw.get("rubric", "")).strip()
        expected_tool = str(raw.get("expected_tool", "")).strip()
        raw_expected_args = raw.get("expected_args", {})
        expected_args = dict(raw_expected_args) if isinstance(raw_expected_args, dict) else {}

        raw_setup = raw.get("setup") or []
        setup = [dict(step) for step in raw_setup if isinstance(step, dict)] if isinstance(raw_setup, list) else []

        def _number(key: str) -> float | None:
            value = raw.get(key)
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                # Kept as it came: voxatest/hearing.py is the one that tells the author why it is bad.
                return value

        raw_barge_at = raw.get("barge_at")
        if raw_barge_at is None or (isinstance(raw_barge_at, str) and not raw_barge_at.strip()):
            raw_barge_at = None
        elif isinstance(raw_barge_at, str) and not raw_barge_at.strip().isdigit():
            pass  # a named overlap position, validated by voxatest/hearing.py
        else:
            try:
                raw_barge_at = float(raw_barge_at)
            except (TypeError, ValueError):
                raw_barge_at = str(raw_barge_at).strip()

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
            setup=setup,
            vm_only=bool(raw.get("vm_only", False)),
            reply=str(raw.get("reply", "")).strip(),
            barge_at=raw_barge_at,
            level_db=_number("level_db"),
            voice=str(raw.get("voice", "")).strip(),
            pause_at=_number("pause_at"),
            pause_s=_number("pause_s"),
            expect=str(raw.get("expect", "")).strip(),
            cell=str(raw.get("cell", "")).strip(),
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
