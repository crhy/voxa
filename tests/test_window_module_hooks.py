"""conftest's _WINDOW_HOOKS list cannot miss a hook (#138 review, g138a's leak).

MainWindow wires lambdas onto a handful of tool MODULES (module globals!).
Without the conftest restore fixture a later test would find the previous
window's dead lambda and answer about a window that no longer exists. The
fixture's list is hand-written, so this test fails the moment window.py
starts wiring a module hook that conftest would not restore: a new hook can
no longer leak between tests unnoticed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import voxa.window


def _conftest_hooks_list() -> dict[str, set[str]]:
    tree = ast.parse((Path(__file__).parent / "conftest.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        else:
            continue
        if getattr(target, "id", "") == "_WINDOW_HOOKS":
            return {name: set(attrs) for name, attrs in ast.literal_eval(node.value)}
    raise AssertionError("tests/conftest.py defines no _WINDOW_HOOKS list")


def test_every_module_hook_window_py_assigns_is_listed_in_conftest() -> None:
    window_globals = vars(voxa.window)
    module_names = {
        name
        for name, value in window_globals.items()
        if inspect.ismodule(value) and str(getattr(value, "__name__", "")).startswith("voxa.")
    }

    assigned: dict[str, set[str]] = {}
    tree = ast.parse(Path(inspect.getsourcefile(voxa.window)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in module_names
            ):
                assigned.setdefault(target.value.id, set()).add(target.attr)

    listed = _conftest_hooks_list()
    missing = {
        f"{window_globals[name].__name__}.{attr}"
        for name, attrs in assigned.items()
        for attr in attrs - listed.get(window_globals[name].__name__, set())
    }
    assert not missing, (
        "voxa/window.py wires these module hooks but tests/conftest.py "
        f"_WINDOW_HOOKS would not restore them: {sorted(missing)}"
    )
