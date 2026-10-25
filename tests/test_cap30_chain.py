import json
from datetime import date
from pathlib import Path

from voxa.agent.hearing import normalize
from voxa.agent.intents import route
from voxa.agent.memory_facts import add, facts, forget, search, spoken_all


def _cap30_cases() -> list[dict]:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    return [case for case in data["cases"] if case.get("capability") == 30]


def _heard(prompt: str) -> str:
    # Whisper writes the spoken word "wifi" as "Wi-Fi"; that is what reaches route() and the memory file.
    return prompt.replace("wifi", "Wi-Fi")


def _fact(prompt: str) -> str:
    return _heard(prompt).split(" that ", 1)[1]


def _topic(prompt: str) -> str:
    return _heard(prompt).split(" about ", 1)[1]


def test_cap30_cases_are_a_chained_run() -> None:
    cases = _cap30_cases()
    assert 3 <= len(cases) <= 15
    content = ""
    for case in cases:
        heard = _heard(case["prompt"])
        call = route(normalize(heard))
        assert call is not None, heard
        assert call.tool == case["expected_tool"], (case["id"], heard, call.tool)
        if case["expected_tool"] == "remember_fact":
            content = add(content, date.today(), _fact(case["prompt"]))
        elif case["expected_tool"] == "forget_fact":
            content, removed = forget(content, _topic(case["prompt"]))
            assert removed is not None, f"{case['id']} forgets a fact that is no longer there"
        elif case["expected_tool"] == "recall_fact":
            found = search(facts(content), _topic(case["prompt"]))
            assert found, case["id"]
            for word in case["expected_keywords"]:
                assert word.casefold() in " ".join(found).casefold(), case["id"]
        elif case["expected_tool"] == "list_facts":
            assert facts(content), case["id"]
            for word in case["expected_keywords"]:
                assert word.casefold() in spoken_all(facts(content)).casefold(), case["id"]
        verify = case.get("verify") or {}
        wanted = str(verify.get("file_contains", "")).casefold()
        unwanted = str(verify.get("file_lacks", "")).casefold()
        if wanted:
            assert wanted in content.casefold(), (case["id"], wanted)
        if unwanted:
            assert unwanted not in content.casefold(), (case["id"], unwanted)


def test_every_forget_is_preceded_by_a_remember() -> None:
    # A "forget X" paraphrase may never run after X was already forgotten: each forget needs the fact put back first.
    present: dict[str, bool] = {}
    for case in _cap30_cases():
        if case["expected_tool"] == "remember_fact":
            present[_fact(case["prompt"]).casefold()] = True
        elif case["expected_tool"] == "forget_fact":
            topic = _topic(case["prompt"]).casefold()
            held = [fact for fact, on in present.items() if on and topic in fact]
            assert held, f"{case['id']} forgets {topic!r}, which is not in the memory file"
            for fact in held:
                present[fact] = False


def test_cap30_checks_the_spelling_whisper_actually_writes() -> None:
    # The memory file holds "Wi-Fi", so a bare "wifi" check can never pass.
    for case in _cap30_cases():
        verify = case.get("verify") or {}
        for key in ("file_contains", "file_lacks"):
            wanted = str(verify.get(key, ""))
            assert wanted != "wifi", case["id"]
