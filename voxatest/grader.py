from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .cases import TestCase

ACTION_CATEGORIES = {
    "desktop_apps",
    "web_and_maps",
    "media_controls",
    "system_status",
    "files_and_productivity",
    "software_help_and_knowledge",
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

    if case.expected_contains or case.expected_keywords:
        # A stated expectation is checked exactly. The small local model is only asked
        # when a case gives nothing but a rubric: its verdicts on short spoken replies
        # were not repeatable, so it must not overrule a plain word check.
        return keyword_grade(case, transcript)

    fallback = _fallback_grade(case, transcript)

    try:
        raw = client.generate_stream(
            model=model,
            prompt=build_grade_prompt(case, transcript),
            cancel_event=cancel_event or threading.Event(),
            on_chunk=lambda _chunk: None,
            num_predict=300,
        )
        parsed = parse_grade_response(raw)
    except Exception as exc:  # noqa: BLE001 - grading should not crash the run
        return GradeResult(fallback.score, fallback.passed, f"Model grading failed: {exc}", "fallback")

    if parsed is None:
        return fallback

    return parsed


def build_grade_prompt(case: TestCase, transcript: str) -> str:
    return (
        "You are a strict but semantic grader for one Voxa voice-assistant test.\n"
        "Evaluate only the captured reply; do not invent hidden actions, context, or environment state.\n"
        "Treat the rubric as primary. Expected phrases and keywords are supporting evidence, not a checklist.\n"
        "Allow synonymous wording, minor transcription errors, and concise spoken confirmations.\n"
        "A claimed action passes only when the reply credibly confirms the requested supported action.\n"
        "Return exactly one single-line JSON object and nothing else: "
        "{\"score\": <integer 0-100>, \"passed\": <boolean>, \"notes\": \"<brief reason>\"}.\n"
        "Set passed=true exactly when score is at least 70; otherwise set passed=false.\n\n"
        f"Category: {case.category}\n"
        f"Success guidance: {_success_guidance(case.category)}\n"
        f"Prompt: {case.prompt}\n"
        f"Expected contains: {', '.join(case.expected_contains) or 'none'}\n"
        f"Expected keywords: {', '.join(case.expected_keywords) or 'none'}\n"
        f"Rubric: {case.rubric or 'general correctness'}\n"
        f"Captured reply: {transcript}\n"
    )


def _success_guidance(category: str) -> str:
    if category == "desktop_apps":
        return "Pass if Voxa confirms the requested app, desktop, or window operation, or accurately reports its result."
    if category == "web_and_maps":
        return "Pass if Voxa confirms opening the requested site, images, nearby search, or directions."
    if category == "media_controls":
        return "Pass if Voxa reports stations or credibly confirms the requested playback or volume operation."
    if category == "system_status":
        return "Pass if Voxa gives the requested status or credibly confirms taking the screenshot."
    if category == "files_and_productivity":
        return "Pass if Voxa credibly confirms the requested folder, hidden-file, note, or email-compose operation."
    if category == "software_help_and_knowledge":
        return "Pass if Voxa handles the supported software flow or answers the help, arithmetic, or stable factual question correctly."
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


_NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split()
    )
}
WORD_MATCH = 0.74
REPLY_MATCH = 0.6


def spoken_words(text: str) -> list[str]:
    """Lower-case words of a spoken sentence, written the way Whisper tends to write them."""
    lowered = text.casefold().replace("%", " percent ").replace("\u2019", "'")
    lowered = re.sub(r"(?<=\d)[:.,](?=\d)", "", lowered)
    words = [word.replace("'", "") for word in re.findall(r"[a-z0-9']+", lowered) if word.strip("'")]
    return [_NUMBER_WORDS.get(word, word) for word in words]


def _same_word(wanted: str, heard: str) -> bool:
    if wanted == heard:
        return True
    if wanted.isdigit() or heard.isdigit():
        return False
    return SequenceMatcher(None, wanted, heard).ratio() >= WORD_MATCH


def reply_match(spoken: str, transcript: str) -> float:
    """Share of the words Voxa said that the microphone transcript also contains."""
    wanted = spoken_words(spoken)
    if not wanted:
        return 1.0
    heard = spoken_words(transcript)
    found = sum(1 for word in wanted if any(_same_word(word, other) for other in heard))
    share = found / len(wanted)
    if share < 1.0 and len(heard) <= 2 * len(wanted) + 2:
        # A name Whisper wrote as two other words ("Caja" as "car jam") still sounds the same
        # letter by letter; only tried when the transcript is about as long as the reply.
        share = max(share, SequenceMatcher(None, "".join(wanted), "".join(heard)).ratio())
    return share


def _phrase_heard(phrase: str, transcript: str) -> bool:
    """True when any ``a|b`` alternative of ``phrase`` is in the transcript, word for word."""
    heard = spoken_words(transcript)
    for alternative in phrase.split("|"):
        wanted = spoken_words(alternative)
        if not wanted:
            continue
        for start in range(len(heard) - len(wanted) + 1):
            if all(_same_word(word, heard[start + offset]) for offset, word in enumerate(wanted)):
                return True
    return False


def keyword_grade(case: TestCase, transcript: str) -> GradeResult:
    """Grade by the case's own expected phrases and keywords; every one must be heard."""
    expected = [*case.expected_contains, *case.expected_keywords]
    missing = [phrase for phrase in expected if not _phrase_heard(phrase, transcript)]
    score = int(100 * (len(expected) - len(missing)) / max(1, len(expected)))
    if missing:
        return GradeResult(score, False, f"The reply did not contain: {', '.join(missing)}.", "keywords")
    return GradeResult(100, True, "The reply contained every expected word.", "keywords")


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
