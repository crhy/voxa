"""Capability 82 (perform computer tasks by voice): the window-in-front paraphrase and the clipboard cases."""

import json
import subprocess
from pathlib import Path

from voxa.agent.hearing import normalize
from voxa.agent.intents import route
from voxatest.setup import run_setup


def test_heard_window_in_front_routes_to_active_window():
    assert route("which window was in front").tool == "active_window_name"
    assert route("which window is in front").tool == "active_window_name"
    assert route("what window was this").tool == "active_window_name"
    assert route("which window was active").tool == "active_window_name"


def test_window_in_front_prompt_still_routes():
    call = route(normalize("Which window is in front"))
    assert call is not None and call.tool == "active_window_name"


def test_unrelated_was_phrases_are_not_window_lookups():
    for prompt in ("what was on my clipboard", "what was open", "what was this window doing"):
        call = route(prompt)
        assert call is None or call.tool != "active_window_name"


def test_clipboard_prompts_still_route():
    for prompt in ("what's on my clipboard", "can you read my clipboard", "hey what is on my clipboard"):
        assert route(prompt).tool == "read_clipboard_aloud"


def test_run_setup_clipboard_step_writes_text():
    written: list[str] = []
    error = run_setup(
        [{"clipboard": "purple marker"}],
        data_dir=Path("."),
        host=lambda cmd, *, stdin=None: subprocess.CompletedProcess(cmd, 0, b"", b""),
        clipboard_write=lambda text: written.append(text) or True,
    )
    assert error == ""
    assert written == ["purple marker"]


def test_run_setup_clipboard_failure_returns_error():
    error = run_setup(
        [{"clipboard": "purple marker"}],
        data_dir=Path("."),
        host=lambda cmd, *, stdin=None: subprocess.CompletedProcess(cmd, 0, b"", b""),
        clipboard_write=lambda text: False,
    )
    assert "clipboard" in error


def test_shipped_clipboard_cases_seed_the_clipboard():
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    cases = {c["id"]: c for c in data["cases"] if c["id"].startswith("cap82-e4")}
    assert set(cases) == {"cap82-e4", "cap82-e4-p1", "cap82-e4-p2"}
    for case in cases.values():
        assert case["expected_tool"] == "read_clipboard_aloud"
        assert {"clipboard": "purple marker"} in case["setup"]
        assert case["vm_only"] is True
        assert "marker" in case["expected_keywords"]
