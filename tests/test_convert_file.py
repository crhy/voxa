"""Voice commands for converting files route and work end to end."""

from __future__ import annotations

import subprocess

from voxa.agent.convert import command, kind, output_path, spoken_done, target_extension
from voxa.agent.intents import ToolCall, route
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.tools import default_registry
from voxa.agent.tools.convert import _convert_handler, convert_tools


def test_target_extension_variants() -> None:
    assert target_extension("pdf") == "pdf"
    assert target_extension("a PDF") == "pdf"
    assert target_extension("the word document") == "docx"
    assert target_extension("word") == "docx"
    assert target_extension("docx") == "docx"
    assert target_extension("open document") == "odt"
    assert target_extension("odt") == "odt"
    assert target_extension("plain text") == "txt"
    assert target_extension("text") == "txt"
    assert target_extension("txt") == "txt"
    assert target_extension("web page") == "html"
    assert target_extension("html") == "html"
    assert target_extension("spreadsheet") == "xlsx"
    assert target_extension("excel") == "xlsx"
    assert target_extension("xlsx") == "xlsx"
    assert target_extension("csv") == "csv"
    assert target_extension("powerpoint") == "pptx"
    assert target_extension("pptx") == "pptx"
    assert target_extension("png") == "png"
    assert target_extension("jpeg") == "jpg"
    assert target_extension("jpg") == "jpg"
    assert target_extension("webp") == "webp"
    assert target_extension("kilometers") is None
    assert target_extension("markdown") is None


def test_kind_and_paths() -> None:
    assert kind("png") == "image"
    assert kind("jpg") == "image"
    assert kind("webp") == "image"
    assert kind("pdf") == "document"
    assert kind("csv") == "document"
    assert output_path("/home/Documents/Report.docx", "pdf") == "/home/Documents/Report.pdf"
    assert output_path("Report.docx", "txt") == "Report.txt"
    assert output_path("/home/Pictures/photo.png", "jpg") == "/home/Pictures/photo.jpg"


def test_commands() -> None:
    assert command("/home/Documents/Report.docx", "pdf") == [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        "/home/Documents",
        "/home/Documents/Report.docx",
    ]
    assert command("/home/Pictures/photo.png", "jpg") == [
        "convert",
        "/home/Pictures/photo.png",
        "/home/Pictures/photo.jpg",
    ]


def test_spoken_done() -> None:
    assert spoken_done("/home/Documents/Report.docx", "pdf") == (
        "I converted Report to PDF. It is next to the original."
    )
    assert spoken_done("/home/Documents/Report.docx", "docx") == (
        "I converted Report to Word. It is next to the original."
    )
    assert spoken_done("/home/Pictures/photo.png", "jpg") == (
        "I converted photo to JPG. It is next to the original."
    )


def test_convert_phrases_route() -> None:
    assert route("convert report to pdf") == ToolCall(
        "convert_file", {"name": "report", "format": "pdf", "folder": ""}
    )
    assert route("convert the report into a pdf in documents") == ToolCall(
        "convert_file", {"name": "report", "format": "pdf", "folder": "documents"}
    )
    assert route("convert my report to excel in the downloads folder") == ToolCall(
        "convert_file", {"name": "report", "format": "excel", "folder": "downloads"}
    )
    assert route("save report as pdf") == ToolCall(
        "convert_file", {"name": "report", "format": "pdf", "folder": ""}
    )
    assert route("export the report as a word document") == ToolCall(
        "convert_file", {"name": "report", "format": "word document", "folder": ""}
    )
    assert route("make pdf of report") == ToolCall(
        "convert_file", {"name": "report", "format": "pdf", "folder": ""}
    )
    assert route("make an excel of the report") == ToolCall(
        "convert_file", {"name": "report", "format": "excel", "folder": ""}
    )


def test_convert_guard_phrases() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("convert 5 miles to kilometers").tool == "calculate"


def test_convert_tool_metadata() -> None:
    tools = convert_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool == Tool(
        name="convert_file",
        description="Convert a document or image to another file format, keeping the result next to the original.",
        parameters={"name": "the file to convert", "format": "the format to convert to", "folder": "which folder it is in"},
        risk=RiskLevel.REVERSIBLE,
        handler=_convert_handler,
        required=("name", "format"),
    )


def test_registry_has_convert_tool() -> None:
    assert "convert_file" in default_registry().names()


def test_convert_handler_unknown_format() -> None:
    result = _convert_handler({"name": "report", "format": "markdown"})
    assert not result.ok
    assert result.speech == "I can't convert to that format."


def test_convert_handler_success(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.convert.simulation.actions_simulated", lambda: True)
    monkeypatch.setattr(
        "voxa.agent.tools.convert._run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._folder",
        lambda spoken: {
            "desktop": "/home/test/Desktop",
            "documents": "/home/test/Documents",
            "downloads": "/home/test/Downloads",
        }.get(spoken),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._names",
        lambda folder: ["Report.docx"] if folder == "/home/test/Documents" else [],
    )

    result = _convert_handler({"name": "report", "format": "pdf", "folder": "documents"})
    assert result.ok
    assert result.speech == "I converted Report to PDF. It is next to the original."


def test_convert_handler_file_not_found(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.convert.simulation.actions_simulated", lambda: True)
    monkeypatch.setattr(
        "voxa.agent.tools.convert._run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._folder",
        lambda spoken: {
            "desktop": "/home/test/Desktop",
            "documents": "/home/test/Documents",
            "downloads": "/home/test/Downloads",
        }.get(spoken),
    )
    monkeypatch.setattr("voxa.agent.tools.convert.filemanage._names", lambda folder: [])

    result = _convert_handler({"name": "missing", "format": "pdf", "folder": "documents"})
    assert not result.ok
    assert result.speech == "I can't find a file called missing."


def test_convert_handler_conversion_failed(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.convert.simulation.actions_simulated", lambda: False)

    def fake_run(cmd, **kwargs):
        if cmd[:1] == ["test"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("voxa.agent.tools.convert._run", fake_run)
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._folder",
        lambda spoken: {
            "desktop": "/home/test/Desktop",
            "documents": "/home/test/Documents",
            "downloads": "/home/test/Downloads",
        }.get(spoken),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._names",
        lambda folder: ["Report.docx"] if folder == "/home/test/Documents" else [],
    )

    result = _convert_handler({"name": "report", "format": "pdf", "folder": "documents"})
    assert not result.ok
    assert result.speech == "The conversion did not work."


def test_convert_handler_exception(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.convert.simulation.actions_simulated", lambda: False)
    monkeypatch.setattr(
        "voxa.agent.tools.convert._run",
        lambda cmd, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("convert", 60)),
    )
    monkeypatch.setattr("voxa.agent.tools.convert.filemanage._error_text", lambda error: "timed out")
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._folder",
        lambda spoken: {
            "desktop": "/home/test/Desktop",
            "documents": "/home/test/Documents",
            "downloads": "/home/test/Downloads",
        }.get(spoken),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.convert.filemanage._names",
        lambda folder: ["Report.docx"] if folder == "/home/test/Documents" else [],
    )

    result = _convert_handler({"name": "report", "format": "pdf", "folder": "documents"})
    assert not result.ok
    assert result.speech == "That did not work: timed out"
