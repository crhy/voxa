import json
from pathlib import Path

from voxa.agent.hearing import normalize
from voxa.agent.intents import route


def test_e1_cases_route() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    cases = [case for case in data["cases"] if "-e" in case["id"] or "-x" in case["id"] or "-y" in case["id"]]
    assert cases
    for case in cases:
        call = route(normalize(case["prompt"]))
        assert call is not None
        assert call.tool == case["expected_tool"]
