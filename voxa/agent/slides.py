"""Turn model-drafted slide text into a minimal OpenDocument presentation.

A presentation is a brand-new file (never an overwrite) in the same "Voxa
Drafts" folder used for documents, opened in LibreOffice Impress so the user
can edit, keep or delete it. The .odp is built with the standard library only.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

MIMETYPE = "application/vnd.oasis.opendocument.presentation"

_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s*(.+)$")
_BULLET = re.compile(r"^\s*[-*•]\s*(.+)$")


def parse_slides(text: str) -> list[tuple[str, list[str]]]:
    """Group the model's reply into (title, bullets) pairs, one per slide.

    Clean replies number each slide title and dash its bullets. If no slide
    is numbered, every non-empty line becomes its own title (a defensive
    fallback), and empty input yields no slides.
    """
    lines = [raw.strip() for raw in text.splitlines() if raw.strip()]
    if not lines:
        return []
    if not any(_NUMBERED.match(line) for line in lines):
        return [(line, []) for line in lines]
    slides: list[tuple[str, list[str]]] = []
    title: str | None = None
    bullets: list[str] = []
    for line in lines:
        numbered = _NUMBERED.match(line)
        if numbered:
            if title is not None:
                slides.append((title, bullets))
            title = numbered.group(2).strip()
            bullets = []
        else:
            if title is None:
                title = ""
            bullet = _BULLET.match(line)
            bullets.append((bullet.group(1) if bullet else line).strip())
    if title is not None:
        slides.append((title, bullets))
    return slides


def _page(index: int, title: str, bullets: list[str]) -> str:
    title_frame = (
        f'<draw:frame draw:style-name="fr1" draw:name="title{index}" '
        'text:anchor-type="anywhere" svg:x="1cm" svg:y="1cm" svg:width="24cm" svg:height="4cm" '
        'draw:z-order="auto">\n'
        '<draw:text-box><text:p>'
        f'<text:span text:style-name="T{index}">{escape(title)}</text:span>'
        "</text:p></draw:text-box>\n"
        "</draw:frame>\n"
    )
    bullet_paras = "".join(f"<text:p>{escape(item)}</text:p>" for item in bullets)
    body_frame = (
        f'<draw:frame draw:style-name="fr2" draw:name="body{index}" '
        'text:anchor-type="anywhere" svg:x="1cm" svg:y="6cm" svg:width="24cm" svg:height="12cm" '
        'draw:z-order="auto">\n'
        f"<draw:text-box>{bullet_paras}</draw:text-box>\n"
        "</draw:frame>\n"
    )
    return (
        f'<draw:page draw:name="page{index}" draw:style-name="dp1" '
        f'draw:master-page-name="Default">\n{title_frame}{body_frame}</draw:page>\n'
    )


CONTENT_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<office:document-content "
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
    'office:version="1.2">\n'
    "<office:body><office:presentation>\n"
)
CONTENT_TAIL = "</office:presentation></office:body>\n</office:document-content>\n"


def _manifest() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
        'manifest:version="1.2">\n'
        ' <manifest:file-entry manifest:full-path="/" manifest:version="1.2" '
        'manifest:media-type="application/vnd.oasis.opendocument.presentation"/>\n'
        ' <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>\n'
        "</manifest:manifest>\n"
    )


def write_odp(path: Path, title: str, slides: list[tuple[str, list[str]]]) -> None:
    """Write a minimal but valid ODF presentation with only the standard library."""
    pages = "".join(_page(i + 1, slide_title, bullets) for i, (slide_title, bullets) in enumerate(slides))
    content = CONTENT_HEAD + pages + CONTENT_TAIL
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            zipfile.ZipInfo("mimetype"), MIMETYPE, compress_type=zipfile.ZIP_STORED
        )
        archive.writestr("META-INF/manifest.xml", _manifest())
        archive.writestr("content.xml", content)
