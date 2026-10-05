import ast
import io
import tokenize
from pathlib import Path
from unittest.mock import patch

from voxa.agent import host


def test_host_command_pins_directory_when_in_flatpak() -> None:
    with patch.object(host, "IN_FLATPAK", True):
        assert host.host_command(["soffice", "/tmp/doc.odt"]) == [
            "flatpak-spawn",
            "--host",
            "--directory=/",
            "soffice",
            "/tmp/doc.odt",
        ]


def test_host_command_runs_plain_outside_flatpak() -> None:
    with patch.object(host, "IN_FLATPAK", False):
        command = ["soffice", "/tmp/doc.odt"]
        assert host.host_command(command) == command


def test_no_host_call_omits_directory_pin() -> None:
    for path in sorted(Path("voxa").rglob("*.py")):
        with path.open(encoding="utf-8") as handle:
            source = handle.read()
        values = [
            ast.literal_eval(token.string)
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.STRING
        ]
        for index, value in enumerate(values):
            if value == "flatpak-spawn" and values[index + 1 : index + 2] == ["--host"]:
                assert values[index + 2 : index + 3] == ["--directory=/"], (
                    f"{path}: --host without --directory=/ on the next token"
                )
