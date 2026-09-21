from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from .cases import TestCase

CATEGORIES = (
    "open_app",
    "email_draft",
    "document_write",
    "web_search",
    "conversation_followup",
    "ambiguous_or_underspecified",
    "capability_boundaries",
    "multi_step",
    "offline_or_error",
    "general_knowledge",
)

CATEGORY_GUIDANCE = {
    "open_app": (
        "Use realistic desktop app-launch commands such as opening a browser, "
        "calculator, file manager, terminal, mail client, or document editor. "
        "Include both common installed apps and one clearly missing app. "
        "Expected behavior is either 'Opening <app>.' or 'I couldn't find an app called <app>.'"
    ),
    "email_draft": (
        "Use clear requests like 'draft an email to Bob about the meeting'. "
        "Voxa does not send email; it drafts text and opens the mail client. "
        "Expected confirmation should mention drafting an email, recipient, subject, "
        "or reviewing/sending it."
    ),
    "document_write": (
        "Use clear requests like 'write a memo about project updates' or 'draft a letter to my landlord'. "
        "Voxa should write a new document and open it in LibreOffice. "
        "Expected confirmation should mention written, opened, LibreOffice, or saved."
    ),
    "web_search": (
        "Use questions that benefit from current or online information, such as weather, "
        "latest score, current price, recent news, or 'search for'. "
        "Expected behavior is that Voxa uses web context or gives a clear factual answer."
    ),
    "conversation_followup": (
        "Make prompts that are natural follow-ups such as 'what about that one?', "
        "'why?', 'explain that', or 'change it to something shorter'. "
        "The rubric should state what the prior context or assumed context must make the reply sensible."
    ),
    "ambiguous_or_underspecified": (
        "Make prompts that are vague or incomplete, such as 'fix it', 'open the thing', "
        "or 'email him'. Success is asking for missing details, stating an assumption, "
        "or explaining that more information is needed."
    ),
    "capability_boundaries": (
        "Use requests outside Voxa's real abilities, such as sending mail, deleting files, "
        "making calls, installing software, or changing settings without a supported command. "
        "Success is a safe limitation, refusal, or request for clarification."
    ),
    "multi_step": (
        "Use requests with two actions, such as open Firefox and search for news, or draft "
        "a memo and open the calculator. Because the current Voxa parser usually handles "
        "one intent, the rubric should check for a partial action plus a sensible limitation."
    ),
    "offline_or_error": (
        "Use prompts that trigger unavailable web search, a missing app, unsupported hardware, "
        "or a failed local action. Success is a clear error, limitation, or fallback answer."
    ),
    "general_knowledge": (
        "Use short factual, math, reasoning, or explanatory questions a local assistant "
        "can answer from its own knowledge, such as capital cities, arithmetic, definitions, "
        "or simple cause-and-effect questions."
    ),
}

# A batch that yields no usable cases is normally a model hiccup (truncated
# JSON, empty content, prose instead of JSON). Retrying forever is how the
# generator hung for hours without producing anything, so both a streak of
# empty batches and an absolute batch count stop the loop.
MAX_EMPTY_BATCHES = 4
BATCH_CEILING_FACTOR = 3
# The local reasoning model spends most of its budget thinking before it
# writes the JSON, so a fixed 1500-token cap reliably truncates the reply.
TOKENS_PER_CASE = 260
MIN_NUM_PREDICT = 1200
EXCERPT_CHARS = 200


def _log(message: str) -> None:
    """Print progress immediately so a redirect or pipe shows it right away."""
    print(message, flush=True)


def build_generation_prompt(count: int, seed: int = 0) -> str:
    category = CATEGORIES[seed % len(CATEGORIES)]
    guidance = CATEGORY_GUIDANCE[category]
    return (
        "Generate exactly "
        f"{count} short Voxa voice-assistant test cases.\n"
        "Return valid JSON only: a list of objects.\n"
        "Each object must have these keys:\n"
        "id, category, prompt, expected_contains, expected_keywords, rubric.\n"
        f"Every object's category field must be exactly {category}.\n"
        f"Category guidance: {guidance}\n"
        "Each prompt must be one sentence and no longer than 18 words.\n"
        "expected_contains should be exact phrases from the expected spoken reply when useful; use an empty list if no exact phrase is required.\n"
        "expected_keywords should be 1-4 words that should appear in the reply.\n"
        "rubric should be a short grading hint that states what success means for this Voxa behavior.\n"
    )


def parse_model_json(text: str) -> list[Any]:
    cleaned = text.strip()
    if not cleaned:
        return []

    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = _find_json_array(cleaned)
        if payload is None:
            return []
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []

    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("test_cases", []))
    if not isinstance(payload, list):
        return []

    return [item for item in payload if isinstance(item, dict)]


def generate_cases(
    client: Any,
    model: str,
    count: int,
    *,
    batch_size: int = 10,
    seed: int = 0,
    cancel_event: threading.Event | None = None,
    max_batches: int | None = None,
    log: Callable[[str], None] = _log,
) -> list[TestCase]:
    """Ask the model for test cases in batches, stopping on lack of progress.

    Progress is printed to stdout for every batch attempt so a stalled or
    unproductive run can be diagnosed from the log instead of hanging silently.
    """
    cancel_event = cancel_event or threading.Event()
    cases: list[TestCase] = []
    seen: set[str] = set()
    generated = 0
    batch_number = 0
    empty_batches = 0

    if max_batches is None:
        max_batches = max(1, count) * BATCH_CEILING_FACTOR

    while generated < count and not cancel_event.is_set():
        if batch_number >= max_batches:
            log(
                f"stopped early: {batch_number} batch attempts reached the ceiling "
                f"with only {generated}/{count} cases."
            )
            break

        remaining = count - generated
        batch = min(max(1, batch_size), remaining)
        category = CATEGORIES[(seed + batch_number) % len(CATEGORIES)]
        prompt = build_generation_prompt(batch, seed + batch_number)
        log(
            f"batch {batch_number + 1}/{max_batches}: category={category} "
            f"seed={seed + batch_number} requested={batch} ..."
        )

        started = time.monotonic()
        raw = client.generate_stream(
            model=model,
            prompt=prompt,
            cancel_event=cancel_event,
            on_chunk=lambda _chunk: None,
            num_predict=max(MIN_NUM_PREDICT, batch * TOKENS_PER_CASE),
        )
        elapsed = time.monotonic() - started

        items = parse_model_json(raw)
        added = 0
        duplicates = 0
        invalid = 0

        for item in items:
            if cancel_event.is_set():
                break
            try:
                case = TestCase.from_dict(item)
            except ValueError:
                invalid += 1
                continue

            key = _case_key(case)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            cases.append(case)
            generated += 1
            added += 1

        log(
            f"batch {batch_number + 1}: {elapsed:.1f}s, {len(items)} items parsed, "
            f"{added} new, {duplicates} duplicate, {invalid} invalid "
            f"-> {generated}/{count} total"
        )

        if added == 0:
            empty_batches += 1
            excerpt = " ".join((raw or "").split())[:EXCERPT_CHARS]
            if excerpt:
                log(f"  batch {batch_number + 1} produced no usable cases; model said: {excerpt}")
            else:
                log(
                    f"  batch {batch_number + 1} produced an empty response "
                    "(the model ran out of tokens before writing the JSON)"
                )
            if empty_batches >= MAX_EMPTY_BATCHES:
                log(
                    f"stopped early: {empty_batches} batches in a row produced no usable "
                    f"cases, returning the {generated} collected so far."
                )
                break
        else:
            empty_batches = 0

        batch_number += 1

    return cases


def _find_json_array(text: str) -> str | None:
    for match in re.findall(r"\[.*\]", text, re.DOTALL):
        return match
    return None


def _case_key(case: TestCase) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", case.prompt.casefold()).strip()
    return normalized or case.id
