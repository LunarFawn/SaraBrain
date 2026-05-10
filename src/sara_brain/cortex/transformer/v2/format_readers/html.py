"""HTML / URL reader. Stdlib only (html.parser + urllib).

Strips tags, tracks the most-recent h1/h2/h3 as section breadcrumbs,
yields one TextSegment per paragraph-like block (<p>, <li>, <blockquote>).
"""
from __future__ import annotations

import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from . import TextSegment


_BLOCK_TAGS = {"p", "li", "blockquote", "div", "section", "article"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "footer", "aside"}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = [""] * 6
        self.skip_depth = 0
        self.in_heading: int | None = None
        self.current_block: list[str] = []
        self.in_block = False
        self.results: list[tuple[str, str]] = []  # (text, breadcrumb)

    def _crumb(self) -> str:
        return " > ".join(h for h in self.headings if h)

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in _HEADING_TAGS:
            self._flush_block()
            self.in_heading = int(tag[1])
            self.current_block = []
            return
        if tag in _BLOCK_TAGS:
            self._flush_block()
            self.in_block = True
            self.current_block = []
            return
        if tag == "br":
            self.current_block.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in _HEADING_TAGS and self.in_heading is not None:
            text = " ".join("".join(self.current_block).split()).strip()
            level = self.in_heading
            self.headings[level - 1] = text
            for i in range(level, 6):
                self.headings[i] = ""
            self.in_heading = None
            self.current_block = []
            return
        if tag in _BLOCK_TAGS and self.in_block:
            self._flush_block()

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_heading is not None or self.in_block:
            self.current_block.append(data)

    def _flush_block(self) -> None:
        if not self.in_block and self.in_heading is None and not self.current_block:
            return
        text = " ".join("".join(self.current_block).split()).strip()
        if text and self.in_block:
            self.results.append((text, self._crumb()))
        self.in_block = False
        self.current_block = []


def _read_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    return Path(source).read_text(encoding="utf-8")


def read(source: str) -> Iterator[TextSegment]:
    raw = _read_source(source)
    parser = _Extractor()
    parser.feed(raw)
    parser._flush_block()
    label = source if source.startswith(("http://", "https://")) else Path(source).name
    for text, crumb in parser.results:
        prov = f"{label}"
        if crumb:
            prov += f"|{crumb}"
        yield TextSegment(text=text, provenance=prov)
