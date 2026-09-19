"""Every module in the package must import, and the model/service layer must stay GTK-free.

Two architecture rules from issue #7 are checked here:

* nothing in the package is dead or broken at import time (each module imports cleanly,
  apart from the ones that genuinely need PyGObject, which are skipped when gi is absent);
* the GTK-free modules (state, controller, config, conversation, dictation, ollama,
  llama.cpp, hardware, catalog) can be imported in a fresh process without gi ever
  appearing in sys.modules - so tests and services never need a display.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import pytest

import voxa

GTK_FREE_MODULES = (
    "voxa.ui.state",
    "voxa.controller",
    "voxa.config",
    "voxa.conversation",
    "voxa.dictation",
    "voxa.ollama",
    "voxa.llamacpp",
    "voxa.hardware",
    "voxa.catalog",
)


def _module_names() -> list[str]:
    names = ["voxa"]
    for info in pkgutil.walk_packages(voxa.__path__, prefix="voxa."):
        names.append(info.name)
    return names


def test_every_module_in_the_package_imports() -> None:
    try:
        import gi  # noqa: F401
    except ImportError:
        gi_available = False
    else:
        gi_available = True

    for name in _module_names():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if not gi_available and "gi" in str(exc):
                continue  # PyGObject is genuinely missing on this machine
            raise AssertionError(f"module {name} failed to import: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any other failure is a real bug
            raise AssertionError(f"module {name} failed to import: {exc}") from exc

    if not gi_available:
        pytest.skip("PyGObject (gi) is not installed; GTK modules were skipped")


def test_the_model_and_service_layer_needs_no_display() -> None:
    code = (
        "import sys, " + ", ".join(GTK_FREE_MODULES) + "; "
        "assert 'gi' not in sys.modules, sorted(m for m in sys.modules if 'gi' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
