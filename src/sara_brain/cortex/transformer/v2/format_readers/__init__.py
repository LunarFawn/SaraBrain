"""Format readers for the book/paper ingest pipeline.

Each reader is a callable that takes a source (file path or URL string)
and yields TextSegment objects in document order. Segments are
paragraph- or section-sized chunks with a provenance tag.

Format detection is by extension first. Override with `format=`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class TextSegment:
    """A chunk of text with provenance for audit/refute."""
    text: str
    provenance: str  # e.g. "paper2_full.txt#chapter=3,part=5" or "url=https://...#h1=Title"


def detect_format(source: str) -> str:
    """Return one of: txt, md, pdf, epub, html, url."""
    s = source.strip()
    if s.startswith(("http://", "https://")):
        return "url"
    suffix = Path(s).suffix.lower()
    return {
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
        ".pdf": "pdf",
        ".epub": "epub",
        ".html": "html",
        ".htm": "html",
    }.get(suffix, "txt")


def read(source: str, format: str | None = None) -> Iterator[TextSegment]:
    """Dispatch to the right reader."""
    fmt = format or detect_format(source)
    if fmt == "txt":
        from . import txt
        yield from txt.read(source)
    elif fmt == "md":
        from . import md
        yield from md.read(source)
    elif fmt == "pdf":
        from . import pdf
        yield from pdf.read(source)
    elif fmt == "epub":
        from . import epub
        yield from epub.read(source)
    elif fmt in {"html", "url"}:
        from . import html
        yield from html.read(source)
    else:
        raise ValueError(f"unknown format: {fmt}")


__all__ = ["TextSegment", "detect_format", "read"]
