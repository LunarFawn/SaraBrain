"""PDF reader. Soft-imports pypdf; raises a clear install hint if missing.

Page-level provenance: each paragraph carries `<filename>#page=N`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from . import TextSegment


def _import_pypdf():
    try:
        import pypdf  # noqa: F401
        return pypdf
    except ImportError as e:
        raise RuntimeError(
            "PDF ingest requires pypdf. Install with: "
            ".venv/bin/pip install pypdf"
        ) from e


def read(source: str) -> Iterator[TextSegment]:
    pypdf = _import_pypdf()
    path = Path(source)
    name = path.name
    reader = pypdf.PdfReader(str(path))
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        for para in _split_paragraphs(page_text):
            if not para.strip():
                continue
            yield TextSegment(
                text=para.strip(),
                provenance=f"{name}#page={page_num}",
            )


def _split_paragraphs(page_text: str) -> list[str]:
    """Split page text on blank lines or runs of newlines."""
    out: list[str] = []
    current: list[str] = []
    for line in page_text.splitlines():
        if line.strip() == "":
            if current:
                out.append(" ".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        out.append(" ".join(current))
    return out
