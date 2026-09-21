from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def summarize(results: list[Any]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.grade.passed)
    failed = total - passed
    errors = sum(1 for result in results if result.error)

    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        category = result.category or "general"
        stats = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        stats["total"] += 1
        if result.grade.passed:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    average_score = sum(result.grade.score for result in results) / total if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "average_score": round(average_score, 2),
        "by_category": by_category,
    }


def format_results(results: list[Any]) -> str:
    if not results:
        return "No test cases were run."

    summary = summarize(results)
    lines = [
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Errors: {summary['errors']}",
        f"Average score: {summary['average_score']}",
        "",
    ]

    for result in results:
        status = "PASS" if result.grade.passed else "FAIL"
        transcript = result.transcript or "no transcript"
        lines.append(f"{status} [{result.category}] {result.case_id}: score {result.grade.score} - {transcript}")
        if result.grade.notes:
            lines.append(f"  notes: {result.grade.notes}")
        if result.error:
            lines.append(f"  error: {result.error}")

    return "\n".join(lines)


def write_reports(
    results: list[Any],
    reports_dir: Path,
    *,
    json_path: Path | None = None,
    text_path: Path | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if json_path is None:
        json_path = reports_dir / f"run_{timestamp}.json"
    if text_path is None:
        text_path = reports_dir / f"run_{timestamp}.txt"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "summary": summarize(results),
        "results": [result.to_dict() for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path.write_text(format_results(results) + "\n", encoding="utf-8")

    return {
        "json_path": json_path,
        "text_path": text_path,
        "summary": summarize(results),
    }
