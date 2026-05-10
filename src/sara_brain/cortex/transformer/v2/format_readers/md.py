"""Markdown reader. Tracks current heading path as provenance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from . import TextSegment

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```")


def read(source: str) -> Iterator[TextSegment]:
    path = Path(source)
    name = path.name
    body = path.read_text(encoding="utf-8")
    headings: list[str] = [""] * 6  # one slot per H1..H6
    in_fence = False
    para: list[str] = []
    line_no = 1
    para_start = 1

    def flush() -> Iterator[TextSegment]:
        nonlocal para, para_start
        if not para:
            return
        text = "".join(para).strip()
        if text:
            crumb = " > ".join(h for h in headings if h)
            prov = f"{name}#L{para_start}"
            if crumb:
                prov += f"|{crumb}"
            yield TextSegment(text=text, provenance=prov)
        para = []

    for line in body.splitlines(keepends=True):
        if _FENCE_RE.match(line.strip()):
            yield from flush()
            in_fence = not in_fence
            line_no += 1
            para_start = line_no
            continue
        if in_fence:
            line_no += 1
            continue
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m:
            yield from flush()
            level = len(m.group(1))
            headings[level - 1] = m.group(2).strip()
            for i in range(level, 6):
                headings[i] = ""
            line_no += 1
            para_start = line_no + 1
            continue
        if line.strip() == "":
            yield from flush()
            line_no += 1
            para_start = line_no + 1
            continue
        if not para:
            para_start = line_no
        para.append(line)
        line_no += 1
    yield from flush()
