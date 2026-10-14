from __future__ import annotations

import os

FORMATS = {
    "pdf": "pdf",
    "word": "docx",
    "docx": "docx",
    "word document": "docx",
    "odt": "odt",
    "open document": "odt",
    "text": "txt",
    "plain text": "txt",
    "txt": "txt",
    "html": "html",
    "web page": "html",
    "excel": "xlsx",
    "xlsx": "xlsx",
    "spreadsheet": "xlsx",
    "csv": "csv",
    "powerpoint": "pptx",
    "pptx": "pptx",
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
}

IMAGE_KINDS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}

UPPER_LABELS = {"pdf", "csv", "png", "jpg", "webp", "html"}
NAME_LABELS = {"docx": "Word", "xlsx": "Excel", "pptx": "PowerPoint"}


def target_extension(spoken: str) -> str | None:
    norm = " ".join(spoken.lower().split())
    for prefix in ("a ", "an ", "the "):
        if norm.startswith(prefix):
            norm = norm[len(prefix) :]
    return FORMATS.get(norm)


def kind(extension: str) -> str:
    return "image" if extension in IMAGE_KINDS else "document"


def output_path(source: str, extension: str) -> str:
    folder = os.path.dirname(source)
    stem = os.path.basename(source)
    dot = stem.rfind(".")
    if dot > 0:
        stem = stem[:dot]
    name = stem + "." + extension
    return os.path.join(folder, name) if folder else name


def command(source: str, extension: str) -> list[str]:
    if kind(extension) == "image":
        return ["convert", source, output_path(source, extension)]
    return ["soffice", "--headless", "--convert-to", extension, "--outdir", os.path.dirname(source), source]


def _label(extension: str) -> str:
    if extension in UPPER_LABELS:
        return extension.upper()
    if extension in NAME_LABELS:
        return NAME_LABELS[extension]
    return "text"


def spoken_done(source: str, extension: str) -> str:
    stem = os.path.basename(source)
    dot = stem.rfind(".")
    if dot > 0:
        stem = stem[:dot]
    return f"I converted {stem} to {_label(extension)}. It is next to the original."
