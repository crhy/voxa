"""Saving by voice: the name question, cleaning the spoken name, replacing an existing file, cancelling."""

import types

from voxa.agent import saveflow
from voxa.agent.saveflow import SaveFlow, clean_name


def test_spoken_names_are_cleaned():
    assert clean_name("Shopping list.") == "Shopping list"
    assert clean_name("call it meeting notes") == "meeting notes"
    assert clean_name("save it as Budget 2026!") == "Budget 2026"
    assert clean_name("notes dot txt") == "notes.txt"
    assert clean_name('"my/file: draft?"') == "my file draft"
    assert clean_name("...") == ""


def test_name_then_saved():
    flow = SaveFlow()
    assert flow.feed("shopping list") == ("", "type")
    assert flow.name == "shopping list"
    assert flow.saved() == "Saved as shopping list."
    assert not flow.active


def test_empty_answer_asks_again():
    flow = SaveFlow()
    speech, action = flow.feed("...")
    assert action is None and saveflow.QUESTION in speech and flow.active


def test_existing_file_yes_replaces():
    flow = SaveFlow()
    flow.feed("notes")
    assert "already exists" in flow.exists()
    assert flow.feed("yes, replace it") == ("", "replace")


def test_existing_file_no_asks_for_another_name():
    flow = SaveFlow("notes", replacing=True)
    speech, action = flow.feed("no")
    assert action == "rename" and "instead" in speech
    assert flow.feed("notes two") == ("", "type")


def test_cancel_closes_the_dialog():
    flow = SaveFlow()
    speech, action = flow.feed("never mind")
    assert action == "cancel" and not flow.active and "not save" in speech


def test_unclear_answer_to_replace_asks_again():
    flow = SaveFlow("notes", replacing=True)
    speech, action = flow.feed("um what")
    assert action is None and "yes or no" in speech


def _fake_desktop(monkeypatch, script):
    """script: list of active-window ids returned in turn (the last one repeats)."""
    from voxa.agent.tools import files

    keys = []
    state = {"i": 0}

    def active():
        value = script[min(state["i"], len(script) - 1)]
        return value, "Save As" if value == "dlg" else ""

    def press(combo):
        keys.append(combo)
        state["i"] += 1

    monkeypatch.setattr(files, "_active_window", active)
    monkeypatch.setattr(files, "press_keys", press)
    monkeypatch.setattr(files, "type_text", lambda text: keys.append("type:" + text))
    monkeypatch.setattr(files, "xdotool_available", lambda: True)
    monkeypatch.setattr(files.time, "sleep", lambda s: None)
    monkeypatch.setattr(files, "_wait_for", lambda condition, seconds=2.0: condition())
    return files, keys


def test_plain_save_of_a_named_file_just_saves(monkeypatch):
    files, keys = _fake_desktop(monkeypatch, ["main", "main"])
    result = files.file_dialog({"action": "save"})
    assert result.speech == "Saved." and keys == ["ctrl+s"]


def test_plain_save_of_a_new_file_asks_for_the_name(monkeypatch):
    files, keys = _fake_desktop(monkeypatch, ["main", "dlg"])
    result = files.file_dialog({"action": "save"})
    assert result.detail == saveflow.ASK_NAME and result.speech == saveflow.QUESTION
    assert files.CURRENT_SAVE is not None


def test_typing_the_name_reports_saved_or_exists(monkeypatch):
    files, keys = _fake_desktop(monkeypatch, ["main", "dlg", "dlg", "main"])  # ctrl+shift+s, ctrl+a, Return -> back to the editor
    result = files.file_dialog({"action": "save", "name": "notes."})
    assert result.speech == "Saved as notes." and "type:notes" in keys
    files, keys = _fake_desktop(monkeypatch, ["main", "dlg", "dlg", "question"])
    result = files.file_dialog({"action": "save", "name": "notes"})
    assert result.detail == saveflow.ASK_REPLACE + "notes"
    assert isinstance(result, types.SimpleNamespace) or "Replace it?" in result.speech
