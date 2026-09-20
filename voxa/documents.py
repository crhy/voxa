"""Write model-drafted documents as new LibreOffice files and open them.

Creating a file is a Level-2 external action, so this stays deliberately
reversible: every document is a brand-new file (never an overwrite) in a
dedicated "Voxa Drafts" folder, opened in LibreOffice Writer for the user to
edit, keep or delete. The ODF is built with the standard library only.
"""

from __future__ import annotations

import os
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

IN_FLATPAK = os.path.exists("/.flatpak-info")

_DOCUMENT = re.compile(
    r"^\s*(?:please\s+)?(?:draft|write|compose|create|make)\s+(?:me\s+)?(?:a|the)?\s*"
    r"(?P<kind>document|doc|letter|report|memo|note|essay)"
    r"(?:\s+(?:called|titled|named)\s+(?P<title>.+?))?"
    r"(?:\s+(?:about|regarding|on)\s+(?P<topic>.+?))?"
    r"\s*[.!?]*\s*$",
    re.I,
)

_EXTRA_COMMAND = re.compile(
    r"\b(?:then|after|and|also)\s+(?:open|launch|close|send|create|write|make|draft|compose|email|document|firefox|browser|app|program|file)\b",
    re.I,
)

DRAFT_SYSTEM_PROMPT = """You write documents for the user to open and edit in LibreOffice Writer. Given a short
request, write the complete document: a clear title on the first line, then the full body
in well-organized paragraphs. Do not include commentary about the task, placeholders, or
notes to the user — write only the finished document text."""

MIMETYPE = "application/vnd.oasis.opendocument.text"
MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
    'manifest:version="1.2">\n'
    ' <manifest:file-entry manifest:full-path="/" manifest:version="1.2" '
    'manifest:media-type="application/vnd.oasis.opendocument.text"/>\n'
    ' <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>\n'
    '</manifest:manifest>\n'
)
CONTENT_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<office:document-content '
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'office:version="1.2">\n'
    '<office:automatic-styles>\n'
    '  <style:style style:name="VoxaTitle" style:family="paragraph">\n'
    '    <style:paragraph-properties fo:margin-top="0pt" fo:margin-bottom="12pt"/>\n'
    '    <style:text-properties fo:font-size="18pt" fo:font-weight="bold"/>\n'
    '  </style:style>\n'
    '</office:automatic-styles>\n'
    '<office:body><office:text>\n'
)
CONTENT_TAIL = "</office:text></office:body>\n</office:document-content>\n"


@dataclass(frozen=True, slots=True)
class DocumentRequest:
    kind: str
    title: str
    topic: str


def parse_document_command(prompt: str) -> DocumentRequest | None:
    """The kind, title and topic from "write me a report about X", or None."""
    if _EXTRA_COMMAND.search(prompt):
        return None
    match = _DOCUMENT.match(prompt)
    if match is None:
        return None
    return DocumentRequest(
        match.group("kind").casefold(),
        (match.group("title") or "").strip(),
        (match.group("topic") or "").strip(),
    )


def build_prompt(request: DocumentRequest) -> str:
    parts = [f"Write a {request.kind}"]
    if request.title:
        parts.append(f"titled “{request.title}”")
    if request.topic:
        parts.append(f"about {request.topic}")
    return " ".join(parts) + "."


def parse_draft(text: str, requested_title: str = "") -> tuple[str, str]:
    """Split the model's reply into (title, body), honouring an explicit title."""
    text = text.strip()
    if requested_title:
        return requested_title, text
    first, _, rest = text.partition("\n")
    return first.strip(), rest.strip()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return slug or "document"


def unique_path(title: str, directory: Path | None = None) -> Path:
    """A never-colliding .odt path inside the user's Voxa Drafts folder."""
    base = Path(directory) if directory is not None else Path(
        os.environ.get("XDG_DOCUMENTS_DIR") or (Path.home() / "Documents")
    )
    folder = base / "Voxa Drafts"
    folder.mkdir(parents=True, exist_ok=True)
    stem = _slug(title)
    path = folder / f"{stem}.odt"
    number = 2
    while path.exists():
        path = folder / f"{stem} {number}.odt"
        number += 1
    return path


def write_odt(path: Path, title: str, body: str) -> None:
    """Write a minimal but valid ODF text document with only the standard library."""
    paragraphs = []
    if title:
        paragraphs.append(f'<text:p text:style-name="VoxaTitle">{escape(title)}</text:p>')
    for line in body.splitlines():
        paragraphs.append(f"<text:p>{escape(line)}</text:p>")
    content = CONTENT_HEAD + "".join(paragraphs) + CONTENT_TAIL
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), MIMETYPE, compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/manifest.xml", MANIFEST)
        archive.writestr("content.xml", content)


def _host(command: list[str]) -> list[str]:
    return ["flatpak-spawn", "--host", *command] if IN_FLATPAK else command


def open_in_libreoffice(path: Path) -> None:
    subprocess.Popen(  # noqa: S603 - fixed argv, the path is one Voxa just created
        _host(["soffice", str(path)]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
