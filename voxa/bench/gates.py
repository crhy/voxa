"""Gate evaluation for the hearing bench.

``python -m voxa.bench run`` evaluates every condition in ``voxa/bench/gates.toml`` against the
report it just built and exits non-zero when one fails; ``--no-gates`` skips the check. The
gates file itself, with the baseline numbers, is roadmap B5 — this module only reads it, so B2
can run with gates enabled and simply find nothing to check.

Conditions are written the way the roadmap writes them: ``<metric><op><number>``, for example
``user_recall_nst>=0.95``. A metric reported with ``value: null`` (a stage that cannot measure it
yet) never fails a gate; it is listed as unmeasured instead.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

GATES_PATH = Path(__file__).with_name("gates.toml")
_CONDITION = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(>=|<=|==|>|<)\s*(-?[0-9.]+)\s*$")
_OPS = {"<=": lambda value, limit: value <= limit, ">=": lambda value, limit: value >= limit,
        "<": lambda value, limit: value < limit, ">": lambda value, limit: value > limit,
        "==": lambda value, limit: value == limit}


def evaluate(path: Path = GATES_PATH, report: dict | None = None) -> list[str]:
    """One line per failed condition; an empty list when every gate holds or the file is absent."""
    if not path.exists():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    metrics = (report or {}).get("metrics", {})
    failures: list[str] = []
    for gate in data.get("gate", []):
        condition = str(gate.get("condition", ""))
        match = _CONDITION.match(condition)
        if match is None:
            failures.append(f"unparsable gate {condition!r} in {path}")
            continue
        name, op, limit = match.group(1), match.group(2), float(match.group(3))
        measured = metrics.get(name)
        if measured is None:
            failures.append(f"{condition}: metric {name} is not in the report")
            continue
        value = measured.get("value")
        if value is None:
            continue  # not measurable at this stage: reported, never failed
        if not _OPS[op](float(value), limit):
            failures.append(f"{condition}: {name}={value}")
    return failures
