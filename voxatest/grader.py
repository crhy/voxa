from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from .cases import TestCase

ACTION_CATEGORIES = {
    "open_app",
    "email_draft",
    "document_write",
    "capability_boundaries",
    "multi_step",
    "offline_or_error",
}


@dataclass(slots=True)
class GradeResult:
    score: int
    passed: bool
    notes: str
    method: str = "model"


def grade_reply(
    case: TestCase,
    transcript: str,
    client: Any,
    model: str,
    cancel_event: threading.Event | None = None,
) -> GradeResult:
    if not transcript.strip():
        return GradeResult(0, False, "No spoken reply was captured.", "empty")

    fallback = _fallback_grade(case, transcript)

    try:
        raw = client.generate_stream(
            model=model,
            prompt=build_grade_prompt(case, transcript),
            cancel_event=cancel_event or threading.Event(),
            on_chunk=lambda _chunk: None,
            num_predict=1200,
        )
        parsed = parse_grade_response(raw)
    except Exception as exc:  # noqa: BLE001 - grading should not crash the run
        return GradeResult(fallback.score, fallback.passed, f"Model grading failed: {exc}", "fallback")

    if parsed is None:
        return fallback

    return parsed


def build_grade_prompt(case: TestCase, transcript: str) -> str:
    return (
        "Grade this spoken reply for a Voxa voice-assistant test.\n"
        "Return valid JSON only with keys: score, passed, notes.\n"
        "score is an integer 0-100. passed is true when the reply satisfies the rubric.\n"
        "Exact expected phrases are helpful, but the rubric decides success when no exact phrase is required.\n\n"
        f"Category: {case.category}\n"
        f"Success guidance: {_success_guidance(case.category)}\n"
        f"Prompt: {case.prompt}\n"
        f"Expected contains: {', '.join(case.expected_contains) or 'none'}\n"
        f"Expected keywords: {', '.join(case.expected_keywords) or 'none'}\n"
        f"Rubric: {case.rubric or 'general correctness'}\n"
        f"Captured reply: {transcript}\n"
    )


def _success_guidance(category: str) -> str:
    if category == "open_app":
        return "Pass if Voxa confirms opening the requested app or clearly says it could not find it."
    if category == "email_draft":
        return "Pass if Voxa drafts the email and opens/mentions the mail app, or clearly says it cannot draft it."
    if category == "document_write":
        return "Pass if Voxa writes/opens a document in LibreOffice or clearly says it cannot do it."
    if category == "web_search":
        return "Pass if the reply uses or acknowledges web search, or gives a clear factual answer with appropriate uncertainty."
    if category == "conversation_followup":
        return "Pass if the reply addresses the follow-up described in the rubric without requiring a rigid phrase."
    if category == "ambiguous_or_underspecified":
        return "Pass if the reply asks for missing details, states an assumption, or explains what is unclear."
    if category == "capability_boundaries":
        return "Pass if the reply states a limitation, asks for clarification, or refuses an unsupported action safely."
    if category == "multi_step":
        return "Pass only if the reply credibly completes the relevant steps or clearly explains which step was not handled."
    if category == "offline_or_error":
        return "Pass if the reply names the limitation/error or provides a clear fallback, without pretending an action happened."
    return "Pass if the reply answers the prompt correctly and clearly."


def parse_grade_response(text: str) -> GradeResult | None:
    payload = _json_object(text)
    if payload is None:
        return None

    score = _coerce_score(payload.get("score"))
    if "passed" in payload:
        passed = _coerce_bool(payload.get("passed"))
    else:
        passed = score >= 70
    notes = str(payload.get("notes", "")).strip() or "Model grade accepted."
    return GradeResult(score, passed, notes, "model")


def _fallback_grade(case: TestCase, transcript: str) -> GradeResult:
    lowered = transcript.casefold()
    contains_hits = sum(1 for phrase in case.expected_contains if phrase.casefold() in lowered)
    keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.casefold() in lowered)

    if case.expected_contains or case.expected_keywords:
        expected_count = len(case.expected_contains) + len(case.expected_keywords)
        score = int(100 * (contains_hits + keyword_hits) / max(1, expected_count))
        passed = score >= 70
    else:
        score = 0
        passed = False

    return GradeResult(
        score,
        passed,
        "Keyword fallback used because model JSON could not be parsed.",
        "fallback",
    )


def _json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_score(value: Any) -> int:
    if isinstance(value, bool):
        return 100 if value else 0
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "passed"}
    return bool(value)
