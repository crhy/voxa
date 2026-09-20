import zipfile
from pathlib import Path

from voxa import documents


def test_parse_document_command_extracts_kind_title_and_topic() -> None:
    cases = [
        ("write a report about quarterly results", ("report", "", "quarterly results")),
        ("draft a memo titled policy update about remote work", ("memo", "policy update", "remote work")),
        ("create a letter named Dear Bob", ("letter", "Dear Bob", "")),
        ("please make the note about taxes", ("note", "", "taxes")),
        ("write a document", ("document", "", "")),
    ]
    for prompt, expected in cases:
        request = documents.parse_document_command(prompt)
        assert request is not None, prompt
        assert (request.kind, request.title, request.topic) == expected, prompt


def test_parse_document_command_rejects_plain_questions_and_other_commands() -> None:
    for prompt in (
        "open firefox",
        "what is a document",
        "tell me about the report",
        "write a document about the budget then open firefox",
        "",
    ):
        assert documents.parse_document_command(prompt) is None, prompt


def test_build_prompt_repeats_the_request() -> None:
    assert documents.build_prompt(documents.DocumentRequest("report", "Budget", "quarterly results")) == "Write a report titled “Budget” about quarterly results."
    assert documents.build_prompt(documents.DocumentRequest("memo", "", "remote work")) == "Write a memo about remote work."
    assert documents.build_prompt(documents.DocumentRequest("letter", "Dear Bob", "")) == "Write a letter titled “Dear Bob”."


def test_parse_draft_uses_first_line_as_title() -> None:
    title, body = documents.parse_draft("Meeting notes\nFirst line\nSecond")
    assert title == "Meeting notes"
    assert body == "First line\nSecond"


def test_parse_draft_honors_explicit_title() -> None:
    title, body = documents.parse_draft("Anything the model wrote", requested_title="Policy")
    assert title == "Policy"
    assert body == "Anything the model wrote"


def test_unique_path_never_overwrites(tmp_path: Path) -> None:
    path = documents.unique_path("Meeting notes", tmp_path)
    assert path == tmp_path / "Voxa Drafts" / "meeting-notes.odt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("first")

    second = documents.unique_path("Meeting notes", tmp_path)
    second.write_text("second")
    third = documents.unique_path("Meeting notes", tmp_path)
    assert second == tmp_path / "Voxa Drafts" / "meeting-notes 2.odt"
    assert third == tmp_path / "Voxa Drafts" / "meeting-notes 3.odt"


def test_write_odt_creates_a_readable_minimal_document(tmp_path: Path) -> None:
    path = tmp_path / "doc.odt"
    documents.write_odt(path, "Hello & World", "Line 1\nLine 2")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert "mimetype" in names
        assert "META-INF/manifest.xml" in names
        assert "content.xml" in names
        assert archive.read("mimetype") == documents.MIMETYPE.encode()
        content = archive.read("content.xml").decode()

    assert "Hello &amp; World" in content
    assert "Line 2" in content


def test_open_in_libreoffice_launches_soffice(monkeypatch) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(documents.subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr(documents, "IN_FLATPAK", False)
    documents.open_in_libreoffice(Path("/tmp/doc.odt"))
    assert launched[0] == ["soffice", "/tmp/doc.odt"]

    launched.clear()
    monkeypatch.setattr(documents, "IN_FLATPAK", True)
    documents.open_in_libreoffice(Path("/tmp/doc.odt"))
    assert launched[0] == ["flatpak-spawn", "--host", "soffice", "/tmp/doc.odt"]
