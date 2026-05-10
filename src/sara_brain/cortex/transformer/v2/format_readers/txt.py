"""Plain text reader. One TextSegment per blank-line-separated paragraph.

Strips light Markdown inline syntax (**bold**, *italic*, ~~strike~~,
leading # / * / - markers, [text](link), backtick code) when it appears,
because authors regularly save Markdown content with a .txt extension
and the trailing punctuation breaks downstream sentence parsing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from . import TextSegment


_BOLD_ITAL_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_|~~)")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_markdown(text: str) -> str:
    text = _LINK_RE.sub(r"\1", text)
    text = _CODE_INLINE_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _BOLD_ITAL_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def read(source: str) -> Iterator[TextSegment]:
    path = Path(source)
    body = path.read_text(encoding="utf-8")
    name = path.name
    line_no = 1
    for chunk in _split_paragraphs(body):
        text, lines = chunk
        if not text.strip():
            line_no += lines
            continue
        cleaned = _strip_markdown(text).strip()
        if cleaned:
            provenance = f"{name}#L{line_no}"
            yield TextSegment(text=cleaned, provenance=provenance)
        line_no += lines


def _split_paragraphs(body: str) -> list[tuple[str, int]]:
    """Split on blank lines. Returns (paragraph, line_count) pairs so
    the caller can track line numbers across paragraphs."""
    paragraphs: list[tuple[str, int]] = []
    current: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.strip() == "":
            if current:
                paragraphs.append(("".join(current), len(current)))
                current = []
            paragraphs.append(("", 1))
        else:
            current.append(line)
    if current:
        paragraphs.append(("".join(current), len(current)))
    return paragraphs
