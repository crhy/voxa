"""V1: what Voxa heard against what was said, and which layer failed.

voxatest always collected Voxa's own words for what she heard (``heard``, built from the action log
in ``voxatest/runner.py``) and never scored them. This module scores them, per case and per run:

* ``hearing_word_accuracy`` - ``1 - WER(what the harness said, what Voxa heard)``, on the bench's
  own normalisation, imported from :mod:`voxa.bench.metrics` (lower-case, punctuation dropped,
  numbers spelled out), with the wake word removed from both sides: voxatest prepends it when it
  speaks (``runner._with_wake_word``), so it is not part of what the case asked.
* ``routed_tool_accuracy`` - the share of the cases that name an ``expected_tool`` in which that
  tool really ran. Cases with no expected tool cannot be scored for routing and are left out.
* ``layer_counts`` - every case is assigned the FIRST layer that failed, in pipeline order:
  ``not_heard`` (no action-log record of what she heard) -> ``misheard`` (hearing accuracy < 1) ->
  ``misrouted`` (heard right, wrong tool) -> ``misanswered`` (routed right, the grader fails) ->
  ``ok``.
* ``confusions`` - the top said-word -> heard-word pairs of the whole run: the words Voxa's ear
  swaps most often, read off the same alignment that produces the WER.

A case the harness itself broke (TTS failed, microphone failed, setup failed) has no action-log
record, so it is counted as ``not_heard``; its ``error`` is kept in the per-case block so a reader
can separate a broken rig from a deaf assistant. Nothing here is a threshold: the numbers a task
gates on live in the task file, and ``voxatest check`` applies them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from voxa.bench.metrics import align, normalise, wer, word_list

LAYERS = ("not_heard", "misheard", "misrouted", "misanswered", "ok")
TOP_CONFUSIONS = 30

# ``name>=0.9``, ``layer_counts.misheard<=2``, ``by_category.timers.hearing_word_accuracy>=0.95``
CONDITION = re.compile(r"^\s*([a-z][a-z0-9_.]*)\s*(>=|<=|==|>|<)\s*(-?[0-9.]+)\s*$")
_OPS = {">=": lambda value, limit: value >= limit,
        ">": lambda value, limit: value > limit,
        "<=": lambda value, limit: value <= limit,
        "<": lambda value, limit: value < limit,
        "==": lambda value, limit: value == limit}


def default_wake_word() -> str:
    """The wake word voxatest speaks with, taken from its own settings default."""
    from .config import HarnessSettings

    return str(HarnessSettings().wake_word or "")


# --------------------------------------------------------------------------- text


def strip_wake(words: list[str], wake: list[str]) -> list[str]:
    """Drop every occurrence of the wake word from a normalised word list."""
    if not wake:
        return list(words)
    kept: list[str] = []
    index = 0
    while index < len(words):
        if words[index : index + len(wake)] == wake:
            index += len(wake)
            continue
        kept.append(words[index])
        index += 1
    return kept


def hearing_accuracy(prompt: str, heard: str, wake_word: str | None = None) -> float | None:
    """1 - WER of what Voxa heard against what the harness said; None with nothing to compare."""
    wake = word_list(default_wake_word() if wake_word is None else wake_word)
    reference = strip_wake(word_list(prompt), wake)
    if not reference:
        return None
    measured = wer(reference, strip_wake(word_list(heard), wake))
    if measured is None:
        return None
    return max(0.0, 1.0 - measured)


def best_record_accuracy(prompt: str, records: list[str], wake_word: str | None = None) -> float | None:
    """The best accuracy of the case's separate action-log records, as a diagnostic next to the total.

    The headline number scores the whole heard text, which is what the harness collected; a case
    whose window also logged an unrelated utterance (a late action of the case before, a radio she
    started) is penalised for those extra words. This is the number for "did she hear the prompt in
    at least one of her own records", reported beside it, never instead of it.
    """
    values = [value for value in (hearing_accuracy(prompt, record, wake_word) for record in records) if value is not None]
    return max(values) if values else None


def confusion_pairs(prompt: str, heard: str, wake_word: str | None = None) -> list[tuple[str, str]]:
    """(said word, heard word) for every reference word the bench alignment calls a substitution."""
    wake = word_list(default_wake_word() if wake_word is None else wake_word)
    reference = strip_wake(word_list(prompt), wake)
    heard_words = strip_wake(word_list(heard), wake)
    if not reference:
        return []
    statuses, _insertions = align(reference, heard_words)
    pairs: list[tuple[str, str]] = []
    cursor = 0
    for word, status in zip(reference, statuses, strict=True):
        if status == "del":
            continue
        other = heard_words[cursor] if cursor < len(heard_words) else ""
        cursor += 1
        if status == "sub" and other:
            pairs.append((word, other))
    return pairs


# --------------------------------------------------------------------------- layers


def tools_ran(result: dict[str, Any]) -> list[str]:
    """Every tool Voxa's action log says ran for this case, in the order it was logged."""
    tools: list[str] = []
    for name in result.get("observed_tools") or []:
        text = str(name)
        if text and text not in tools:
            tools.append(text)
    record = result.get("action_record")
    if isinstance(record, dict):
        text = str(record.get("tool", ""))
        if text and text not in tools:
            tools.append(text)
    return tools


def classify_layer(result: dict[str, Any], accuracy: float | None, wake_word: str | None = None) -> str:
    """The first layer that failed on this case, in the order the signal travels."""
    expected = str(result.get("expected_tool", "") or "")
    if not str(result.get("heard", "")).strip() or accuracy is None:
        return "not_heard"
    if accuracy < 1.0:
        return "misheard"
    if expected and expected not in tools_ran(result):
        return "misrouted"
    if not result.get("passed"):
        return "misanswered"
    return "ok"


def score_case(result: dict[str, Any], wake_word: str | None = None) -> dict[str, Any]:
    """One case's hearing score, routing verdict, failing layer and word confusions."""
    wake = default_wake_word() if wake_word is None else wake_word
    prompt = str(result.get("prompt", "") or "")
    heard = str(result.get("heard", "") or "")
    accuracy = hearing_accuracy(prompt, heard, wake)
    records = [str(item) for item in (result.get("heard_records") or []) if str(item).strip()]
    best = best_record_accuracy(prompt, records or ([heard] if heard.strip() else []), wake)
    expected = str(result.get("expected_tool", "") or "")
    ran = tools_ran(result)
    return {
        "case_id": str(result.get("case_id", "")),
        "category": str(result.get("category", "") or "general"),
        "prompt": prompt,
        "heard": heard,
        "hearing_word_accuracy": None if accuracy is None else round(accuracy, 4),
        "hearing_word_accuracy_best_record": None if best is None else round(best, 4),
        "layer": classify_layer(result, accuracy, wake),
        "expected_tool": expected,
        "tools_ran": ran,
        "routed": None if not expected else expected in ran,
        "passed": bool(result.get("passed")),
        "error": str(result.get("error", "") or ""),
        "confusions": [list(pair) for pair in confusion_pairs(prompt, heard, wake)],
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _group_accuracy(cases: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(case["hearing_word_accuracy"]) for case in cases if case["hearing_word_accuracy"] is not None]
    routed = [case for case in cases if case["expected_tool"]]
    counts = {layer: 0 for layer in LAYERS}
    for case in cases:
        counts[str(case["layer"])] = counts.get(str(case["layer"]), 0) + 1
    return {
        "cases": len(cases),
        "hearing_word_accuracy": _mean(accuracies),
        "routed_tool_accuracy": (
            None if not routed else _mean([1.0 if case["routed"] else 0.0 for case in routed])
        ),
        "layer_counts": counts,
    }


def score_run(results: list[dict[str, Any]], wake_word: str | None = None, top: int = TOP_CONFUSIONS) -> dict[str, Any]:
    """The Comprehension block of a whole run: the four JSON keys the brief fixes, plus the detail."""
    wake = default_wake_word() if wake_word is None else wake_word
    cases = [score_case(result, wake) for result in results]
    accuracies = [float(case["hearing_word_accuracy"]) for case in cases if case["hearing_word_accuracy"] is not None]
    routed = [case for case in cases if case["expected_tool"]]

    counter: Counter[tuple[str, str]] = Counter()
    for case in cases:
        for said, heard_word in case["confusions"]:
            counter[(str(said), str(heard_word))] += 1
    confusions = [
        {"said": said, "heard": heard_word, "count": count}
        for (said, heard_word), count in sorted(counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ][: max(0, int(top))]

    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(str(case["category"]), []).append(case)
    return {
        "hearing_word_accuracy": _mean(accuracies),
        "routed_tool_accuracy": None if not routed else _mean([1.0 if case["routed"] else 0.0 for case in routed]),
        "layer_counts": {layer: sum(1 for case in cases if case["layer"] == layer) for layer in LAYERS},
        "confusions": confusions,
        "cases_scored": len(cases),
        "routed_cases": len(routed),
        "wake_word": wake,
        "by_category": {name: _group_accuracy(group) for name, group in sorted(by_category.items())},
        "cases": cases,
    }


# --------------------------------------------------------------------------- report text


def _number(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def format_comprehension(comprehension: dict[str, Any], top: int = TOP_CONFUSIONS) -> list[str]:
    """The Comprehension section of the text report: rates, layers, categories, confusions."""
    counts = comprehension.get("layer_counts", {})
    lines = [
        "Comprehension",
        f"  hearing_word_accuracy  {_number(comprehension.get('hearing_word_accuracy'))}"
        f"  ({comprehension.get('cases_scored', 0)} cases)",
        f"  routed_tool_accuracy   {_number(comprehension.get('routed_tool_accuracy'))}"
        f"  ({comprehension.get('routed_cases', 0)} cases with an expected tool)",
        "  layers: " + ", ".join(f"{layer} {counts.get(layer, 0)}" for layer in LAYERS),
        "  per category:",
    ]
    for name, group in (comprehension.get("by_category") or {}).items():
        lines.append(
            f"    {name}: cases {group.get('cases', 0)}"
            f" hearing {_number(group.get('hearing_word_accuracy'))}"
            f" routed {_number(group.get('routed_tool_accuracy'))}"
        )
    lines.append(f"  CONFUSIONS (top {top}):")
    rows = comprehension.get("confusions") or []
    if not rows:
        lines.append("    none: every word heard as it was said")
    for row in rows:
        lines.append(f"    {row['count']}x  {row['said']} -> {row['heard']}")
    return lines


def format_comprehension_markdown(comprehension: dict[str, Any], top: int = TOP_CONFUSIONS) -> list[str]:
    """The same section as Markdown, for the failure snapshot."""
    counts = comprehension.get("layer_counts", {})
    lines = [
        "## Comprehension",
        "",
        f"- Hearing word accuracy: {_number(comprehension.get('hearing_word_accuracy'))}"
        f" ({comprehension.get('cases_scored', 0)} cases)",
        f"- Routed tool accuracy: {_number(comprehension.get('routed_tool_accuracy'))}",
        "- Layers: " + ", ".join(f"`{layer}` {counts.get(layer, 0)}" for layer in LAYERS),
        "",
        "| Category | Cases | Hearing accuracy | Routed tool accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, group in (comprehension.get("by_category") or {}).items():
        lines.append(
            f"| {name} | {group.get('cases', 0)} | {_number(group.get('hearing_word_accuracy'))}"
            f" | {_number(group.get('routed_tool_accuracy'))} |"
        )
    lines.extend(["", f"### Confusions (top {top})", ""])
    rows = comprehension.get("confusions") or []
    if not rows:
        lines.append("None: every word was heard as it was said.")
        return lines
    lines.extend(["| Said | Heard | Count |", "| --- | --- | ---: |"])
    for row in rows:
        lines.append(f"| {row['said']} | {row['heard']} | {row['count']} |")
    return lines


def case_line(result: dict[str, Any], wake_word: str | None = None) -> str:
    """One case's comprehension as a single report line: layer, hearing accuracy, routing."""
    case = score_case(result, wake_word)
    routed = "n/a" if case["routed"] is None else ("yes" if case["routed"] else "no")
    return (
        f"comprehension: layer={case['layer']} hearing={_number(case['hearing_word_accuracy'])}"
        f" routed={routed}"
    )


# --------------------------------------------------------------------------- commands


def _resolve(report: dict[str, Any], name: str) -> tuple[Any, str]:
    """Look a dotted condition name up in the Comprehension block: (value, why it is missing)."""
    comprehension = report.get("comprehension")
    if not isinstance(comprehension, dict):
        return None, "the report has no comprehension block (re-score it with `voxatest comprehension`)"
    node: Any = comprehension
    for part in name.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
            continue
        return None, f"{name} is not in the comprehension block"
    return node, ""


def check(argv: list[str] | None = None) -> int:
    """``python -m voxatest check REPORT "<name><op><number>" ...``; non-zero unless every one holds."""
    parser = argparse.ArgumentParser(
        prog="python -m voxatest check",
        description="assert comprehension conditions on a voxatest run report (exits 1 on any failure, gap or null)",
    )
    parser.add_argument("report", type=Path, help="the run report JSON written by `voxatest run`")
    parser.add_argument("conditions", nargs="+",
                        help='conditions like hearing_word_accuracy>=0.9, layer_counts.misheard<=2')
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"check: cannot read {args.report}: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for condition in args.conditions:
        match = CONDITION.match(condition)
        if match is None:
            print(f"{condition}: unparsable condition")
            failures += 1
            continue
        name, operator, limit = match.group(1), match.group(2), float(match.group(3))
        value, problem = _resolve(report, name)
        if problem:
            print(f"{condition}: MISSING {problem}")
            failures += 1
            continue
        if value is None:
            print(f"{condition}: NULL {name} = null (nothing measurable was scored)")
            failures += 1
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            print(f"{condition}: {name} is not a number ({value!r})")
            failures += 1
            continue
        holds = bool(_OPS[operator](float(value), limit))
        shown = f"{value}" if isinstance(value, int) else f"{float(value):.4f}"
        print(f"{condition}: {name} = {shown} {'OK' if holds else 'FAIL'}")
        if not holds:
            failures += 1
    if failures:
        print(f"check FAILED: {failures} of {len(args.conditions)} conditions do not hold", file=sys.stderr)
        return 1
    return 0


def rescore(report: dict[str, Any], wake_word: str | None = None, top: int = TOP_CONFUSIONS) -> dict[str, Any]:
    """Re-score an existing run report: the same report with a fresh comprehension block."""
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("the report has no results array to re-score")
    payload = dict(report)
    payload["comprehension"] = score_run([item for item in results if isinstance(item, dict)], wake_word, top)
    return payload


def comprehension_main(argv: list[str] | None = None) -> int:
    """``python -m voxatest comprehension --report RUN.json`` re-scores a written run report."""
    parser = argparse.ArgumentParser(
        prog="python -m voxatest comprehension",
        description="score what Voxa heard against what was said, from a written run report",
    )
    parser.add_argument("--report", required=True, type=Path, help="the run report JSON written by `voxatest run`")
    parser.add_argument("--out", type=Path, help="write the re-scored report here (the report is left untouched otherwise)")
    parser.add_argument("--wake-word", default=None, help="wake word to strip from both sides (default: Voxa's setting)")
    parser.add_argument("--top", type=int, default=TOP_CONFUSIONS, help="how many confusion rows to keep")
    parser.add_argument("--json", action="store_true", help="print the comprehension block as JSON")
    args = parser.parse_args(argv)

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"comprehension: cannot read {args.report}: {exc}", file=sys.stderr)
        return 2
    try:
        rescored = rescore(report, args.wake_word, args.top)
    except ValueError as exc:
        print(f"comprehension: {exc}", file=sys.stderr)
        return 2

    comprehension = rescored["comprehension"]
    if args.json:
        print(json.dumps(comprehension, indent=2, ensure_ascii=False))
    else:
        print("\n".join(format_comprehension(comprehension, args.top)))
    if args.out:
        from .report import _atomic_write_text

        _atomic_write_text(args.out, json.dumps(rescored, indent=2, ensure_ascii=False) + "\n")
        print(f"Re-scored report: {args.out}", flush=True)
    return 0


__all__ = [
    "LAYERS",
    "TOP_CONFUSIONS",
    "align",
    "best_record_accuracy",
    "case_line",
    "check",
    "classify_layer",
    "comprehension_main",
    "confusion_pairs",
    "default_wake_word",
    "format_comprehension",
    "format_comprehension_markdown",
    "hearing_accuracy",
    "normalise",
    "rescore",
    "score_case",
    "score_run",
    "strip_wake",
    "tools_ran",
    "wer",
    "word_list",
]
