from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
import voxa.agent.tools.notes as notes_mod
from voxa.agent.intents import route
from voxa.agent.notes import note_line, parse_notes, spoken_notes, without_last

APPEND_SCRIPT = 'printf "%s\\n" "$1" >> "$2"'
REWRITE_SCRIPT = 'printf "%s" "$1" > "$2"'


def test_note_line():
    assert note_line("call the vet on Friday", "2026-10-07 10:45") == "- 2026-10-07 10:45 — Call the vet on Friday."
    assert note_line("call the vet!", "2026-10-07 10:45") == "- 2026-10-07 10:45 — Call the vet!"
    assert note_line("are we done?", "2026-10-07 10:45") == "- 2026-10-07 10:45 — Are we done?"
    assert note_line("buy milk.", "2026-10-07 10:45") == "- 2026-10-07 10:45 — Buy milk."


def test_parse_notes():
    content = "# My Notes\n\n- 2026-10-07 10:45 — Call the vet on Friday.\n- 2026-10-08 09:00 — Buy milk.\n"
    assert parse_notes(content) == ["Call the vet on Friday.", "Buy milk."]
    assert parse_notes("") == []
    assert parse_notes("just prose\n") == []


def test_without_last():
    content = "# My Notes\n\n- 2026-10-07 10:45 — Call the vet on Friday.\n- 2026-10-08 09:00 — Buy milk.\n"
    new_content, text = without_last(content)
    assert text == "Buy milk."
    assert new_content == "# My Notes\n\n- 2026-10-07 10:45 — Call the vet on Friday.\n"
    assert without_last("# nothing\n") == ("# nothing\n", None)
    assert without_last("") == ("", None)


def test_spoken_notes():
    assert spoken_notes([]) == "You have no notes."
    assert spoken_notes(["Call the vet on Friday."]) == "You have one note: Call the vet on Friday."
    seven = [f"N{i}." for i in range(1, 8)]
    assert spoken_notes(seven) == "You have 7 notes. The latest are: N7. N6. N5. N4. N3."
    assert spoken_notes(["A.", "B.", "C."]) == "You have 3 notes. The latest are: C. B. A."


class FakeHost:
    def __init__(self) -> None:
        self.file = ""
        self.calls: list[list[str]] = []

    def run(self, command, stdin=None, check=True):
        argv = list(command)
        self.calls.append(argv)
        if argv[0] == "cat":
            return subprocess.CompletedProcess(argv, 0, self.file, "")
        if argv[0] == "date":
            return subprocess.CompletedProcess(argv, 0, "2026-10-07 10:45", "")
        if argv[:3] == ["sh", "-c", APPEND_SCRIPT]:
            self.file += argv[4] + "\n"
        elif argv[:3] == ["sh", "-c", REWRITE_SCRIPT]:
            self.file = argv[4]
        return subprocess.CompletedProcess(argv, 0, "", "")


def _fake_host(monkeypatch):
    fake = FakeHost()
    monkeypatch.setattr(notes_mod, "_run", fake.run)
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: "/home/u/Documents")
    return fake


def test_take_read_delete(monkeypatch):
    fake = _fake_host(monkeypatch)
    for text in ("call the vet on Friday", "buy milk", "water the plants"):
        result = notes_mod._take_note_handler({"text": text})
        assert result.ok and result.speech == "Noted."
    read = notes_mod._read_notes_handler({})
    assert read.speech == "You have 3 notes. The latest are: Water the plants. Buy milk. Call the vet on Friday."
    deleted = notes_mod._delete_last_note_handler({})
    assert deleted.speech == "Deleted the note: Water the plants."
    again = notes_mod._read_notes_handler({})
    assert again.speech == "You have 2 notes. The latest are: Buy milk. Call the vet on Friday."
    assert fake.file == (
        "- 2026-10-07 10:45 — Call the vet on Friday.\n"
        "- 2026-10-07 10:45 — Buy milk.\n"
    )


def test_empty_text(monkeypatch):
    _fake_host(monkeypatch)
    result = notes_mod._take_note_handler({"text": "   "})
    assert not result.ok and result.speech == "What should the note say?"


def test_dangerous_text_is_one_argv_element(monkeypatch):
    fake = _fake_host(monkeypatch)
    tricky = 'say "hi" $HOME `whoami`'
    result = notes_mod._take_note_handler({"text": tricky})
    assert result.ok and result.speech == "Noted."
    append_calls = [argv for argv in fake.calls if argv[:3] == ["sh", "-c", APPEND_SCRIPT]]
    assert len(append_calls) == 1
    argv = append_calls[0]
    assert argv[4] == note_line(tricky, "2026-10-07 10:45")
    for element in argv[:2] + argv[3:]:
        assert ">>" not in element


def test_open_notes(monkeypatch):
    fake = _fake_host(monkeypatch)
    empty = notes_mod._open_notes_handler({})
    assert not empty.ok and empty.speech == "You have no notes yet."
    notes_mod._take_note_handler({"text": "call the vet"})
    opened = notes_mod._open_notes_handler({})
    assert opened.ok and opened.speech == "Opening your notes."
    assert any(argv[:2] == ["gio", "open"] for argv in fake.calls)


def test_router():
    call = route("take a note: call the vet on Friday")
    assert call is not None and call.tool == "take_note" and call.args["text"] == "call the vet on Friday"
    for phrase in ("note that the rent is due", "write this down: buy milk", "jot that down buy eggs"):
        call = route(phrase)
        assert call is not None and call.tool == "take_note" and call.args["text"]
    for phrase in ("read my notes", "what's in my notes?", "do i have any notes", "show me my notes"):
        assert route(phrase).tool == "read_notes"
    for phrase in ("delete my last note", "remove the last note"):
        assert route(phrase).tool == "delete_last_note"
    assert route("open my notes").tool == "open_notes"
    assert route("open notes").tool == "open_notes"
    assert route("remind me to call the vet at 3pm").tool == "set_reminder"
    screenshot = route("take a screenshot")
    assert screenshot is None or screenshot.tool != "take_note"
    assert route("note") is None
