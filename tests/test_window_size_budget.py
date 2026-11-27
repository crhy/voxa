"""window.py must stop growing (#138 part 1).

The extraction left voxa/window.py at its new, smaller line count; this test
pins it (count + 50 lines of slack) so the window can only shrink from here,
and it forbids the back door: no module under voxa/ except the window itself
(and its application.py shell) may import voxa.window again.
"""

from __future__ import annotations

import re
from pathlib import Path

import voxa

# voxa/window.py's line count when the pipeline moved out, plus the allowed slack.
WINDOW_LINE_BUDGET = 4800

_EXEMPT = {"window.py", "application.py"}

_IMPORT_WINDOW = re.compile(r"^\s*(?:from|import)\b.*\bwindow\b", re.IGNORECASE)


def _window_source() -> Path:
    return Path(voxa.__file__).parent / "window.py"


def test_window_py_stays_within_its_line_budget() -> None:
    lines = len(_window_source().read_text(encoding="utf-8").splitlines())
    assert lines <= WINDOW_LINE_BUDGET, (
        f"voxa/window.py grew to {lines} lines; the pipeline belongs in the "
        f"controller (budget {WINDOW_LINE_BUDGET})."
    )


def test_no_voxa_module_imports_the_window_again() -> None:
    offenders: list[str] = []
    for path in sorted(Path(voxa.__file__).parent.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # ``import windows``/``ui.window`` are not the window module; the
            # exact module path is: "voxa.window", ".window", "..window".
            if not _IMPORT_WINDOW.match(line):
                continue
            if re.search(r"\b(?:voxa\.window|from\s+\.{1,2}\s+.*\bwindow\b|\.window\b|\.\.window\b)", line):
                offenders.append(f"{path.relative_to(Path(voxa.__file__).parent)}: {line.strip()}")
    assert offenders == []
