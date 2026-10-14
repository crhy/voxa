from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from .cases import TestCase

CATEGORIES = (
    "desktop_apps",
    "web_and_maps",
    "media_controls",
    "system_status",
    "files_and_productivity",
    "software_help_and_knowledge",
)

CATEGORY_GUIDANCE = {
    "desktop_apps": (
        "Test one supported desktop operation: launch a common app, show the desktop, "
        "or report open windows. Use generic app names that Voxa's desktop router recognizes."
    ),
    "web_and_maps": (
        "Test one supported browser operation: open a known site, search for images, "
        "find a nearby place, or request directions. Do not depend on current web facts."
    ),
    "media_controls": (
        "Test one supported radio or audio operation: list stations, play a named genre, "
        "or pause playback. Never change, mute or set the volume: the prompts are heard through the "
        "speakers, so a quieter volume silences every later case. Do not require a particular station to be online."
    ),
    "system_status": (
        "Test one supported system operation: report time/date, disk or memory usage, "
        "or take a screenshot. Accept concise spoken confirmations and equivalent wording."
    ),
    "files_and_productivity": (
        "Test one supported file or productivity operation: open/list a standard folder, "
        "toggle hidden files, create a note, or compose an email. Do not send or delete anything."
    ),
    "software_help_and_knowledge": (
        "Test one supported software-management, help, arithmetic, or stable general-knowledge "
        "request. Installing software and checking updates are supported flows."
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
        "id, category, prompt, expected_contains, expected_keywords, rubric, expected_tool, expected_args.\n"
        f"Every object's category field must be exactly {category}.\n"
        f"Category guidance: {guidance}\n"
        "Every case is run independently with a fresh, unknown conversation state. It must not depend on another case.\n"
        "Test exactly one operation per case; do not combine commands.\n"
        "Each prompt must be one natural, speakable sentence of 4-18 words with no markdown.\n"
        "Avoid politeness filler, unnecessary urgency, ambiguous pronouns, and facts whose answer changes over time.\n"
        "expected_contains must contain only complete contiguous phrases that are essential and deterministic; otherwise use an empty list.\n"
        "expected_keywords must contain 1-3 stable semantic terms, not every word in an ideal answer.\n"
        "For an action request, expected_tool must be the exact Voxa tool name and expected_args must contain its deterministic arguments.\n"
        "For a pure knowledge question, expected_tool must be an empty string and expected_args an empty object.\n"
        "Do not require a particular network state, installed optional app, date, officeholder, price, score, or current event.\n"
        "Do not ask for the volume to be changed or muted.\n"
        "Do not request phone calls, smart-home control, purchases, destructive file operations, or other unsupported features.\n"
        "The rubric must describe one observable property of the captured spoken reply and allow equivalent wording.\n"
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
