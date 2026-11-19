"""Gate evaluation for the hearing bench.

``python -m voxa.bench run`` evaluates ``voxa/bench/gates.toml`` against the report it just built
and exits non-zero when one condition fails; ``--no-gates`` skips that. Roadmap B3 creates the file
with every metric the brief names and NO threshold yet, so a run reports every metric and fails
nothing; roadmap B5 records the thresholds from the baseline run. The numbers a task gates on live
in the task file, never here.

Two shapes are read, and both mean the same thing:

* ``[[metric]] name = "user_recall_nst" direction = ">=" threshold = 0.95`` - one metric, one limit;
* ``[[gate]] condition = "user_recall_nst>=0.95"`` - the compact form the roadmap writes.

A metric reported with ``value: null`` (a stage that cannot measure it yet) never fails a gate: it
is listed as unmeasured instead. A metric a condition names that is not in the report at all is a
failure, because a gap in the report is not a pass.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

GATES_PATH = Path(__file__).with_name("gates.toml")
CONDITION = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(>=|<=|==|>|<)\s*(-?[0-9.]+)\s*$")
_OPS = {"<=": lambda value, limit: value <= limit,
        ">=": lambda value, limit: value >= limit,
        "<": lambda value, limit: value < limit,
        ">": lambda value, limit: value > limit,
        "==": lambda value, limit: value == limit}


def parse_condition(text: str) -> tuple[str | None, str, float]:
    """``("user_recall_nst", ">=", 0.95)``, or ``(None, "", 0.0)`` when it does not parse."""
    match = CONDITION.match(text)
    if match is None:
        return None, "", 0.0
    return match.group(1), match.group(2), float(match.group(3))


def compare(value: float, operator: str, limit: float) -> bool:
    """Apply one condition's operator to one measured value."""
    return bool(_OPS[operator](float(value), float(limit)))


def _load(path: Path | None = None) -> dict:
    path = path if path is not None else GATES_PATH
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def thresholds(path: Path | None = None) -> dict[str, tuple[str, float]]:
    """``metric name -> (operator, limit)`` for every ``[[metric]]`` entry that sets a threshold."""
    out: dict[str, tuple[str, float]] = {}
    for entry in _load(path).get("metric", []):
        name = str(entry.get("name", ""))
        if not name or entry.get("threshold") is None:
            continue
        direction = str(entry.get("direction", ">="))
        out[name] = (direction, float(entry["threshold"]))
    return out


def evaluate(path: Path | None = None, report: dict | None = None) -> list[str]:
    """One line per failed condition; an empty list when every gate holds or the file is absent."""
    data = _load(path)
    metrics = (report or {}).get("metrics", {})
    failures: list[str] = []
    limits = thresholds(path)
    for name, (operator, limit) in limits.items():
        measured = metrics.get(name)
        if measured is None:
            failures.append(f"{name}{operator}{limit}: metric {name} is not in the report")
            continue
        value = measured.get("value")
        if value is None:
            continue  # not measurable at this stage: reported, never failed
        if not compare(float(value), operator, limit):
            failures.append(f"{name}{operator}{limit}: {name}={value}")
    for gate in data.get("gate", []):
        condition = str(gate.get("condition", ""))
        name, operator, limit = parse_condition(condition)
        if name is None:
            failures.append(f"unparsable gate {condition!r} in {path}")
            continue
        measured = metrics.get(name)
        if measured is None:
            failures.append(f"{condition}: metric {name} is not in the report")
            continue
        value = measured.get("value")
        if value is None:
            continue
        if not compare(float(value), operator, limit):
            failures.append(f"{condition}: {name}={value}")
    return failures
