"""vf02 #72 (2b): the model picker always shows the model actually in use, marks one that
a reachable backend does not list, and names an unreachable backend instead of "(None)".

These are pure functions so a fake backend is just a list — no GTK or server needed.
"""

from __future__ import annotations

from voxa.model_choices import (
    NOT_INSTALLED_SUFFIX,
    backend_offline_message,
    model_choices,
)


def test_empty_list_then_filled_selects_configured_model() -> None:
    # A backend that starts with no models and fills in later: once the configured
    # model is listed the picker selects it.
    choices, selected = model_choices("qwen3.5-0.8b-q4km:latest", [])
    assert selected == f"qwen3.5-0.8b-q4km:latest{NOT_INSTALLED_SUFFIX}"
    assert selected in choices

    choices, selected = model_choices(
        "qwen3.5-0.8b-q4km:latest",
        ["qwen2.5:0.5b", "qwen3.5-0.8b-q4km:latest"],
    )
    assert selected == "qwen3.5-0.8b-q4km:latest"
    assert selected in choices


def test_configured_model_missing_from_list_shown_not_installed() -> None:
    choices, selected = model_choices(
        "qwen3.5-0.8b-q4km:latest",
        ["qwen2.5:0.5b", "mistral:7b"],
    )
    assert selected == f"qwen3.5-0.8b-q4km:latest{NOT_INSTALLED_SUFFIX}"
    # The real models stay selectable; the configured one is pinned first, marked.
    assert choices[0] == selected
    assert "qwen2.5:0.5b" in choices and "mistral:7b" in choices


def test_no_none_placeholder_when_configured_exists() -> None:
    _, selected = model_choices("some-model", [])
    assert selected and selected != "None" and selected != "(None)"


def test_unreachable_backend_names_the_backend_not_none() -> None:
    assert backend_offline_message("Ollama") == "Ollama is not running"
    msg = backend_offline_message("Ollama")
    assert msg and msg not in ("", "(None)", "None")
