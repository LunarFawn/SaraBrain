#!/usr/bin/env python3
"""Download and chunk Wikipedia chemistry pages for Sara Brain ingest.

Pulls each page, strips HTML preserving paragraph structure,
and saves small chunk files that a 3B model can process thoroughly.

Usage:
    python benchmarks/prep_chemistry_sources.py

Output:
    benchmarks/chemistry_chunks/organic_reaction_001.txt
    benchmarks/chemistry_chunks/organic_reaction_002.txt
    ...
"""

from __future__ import annotations

import re
import os
import urllib.request

SOURCES = [
    ("organic_reaction", "https://en.wikipedia.org/wiki/Organic_reaction"),
    ("organic_chemistry", "https://en.wikipedia.org/wiki/Organic_chemistry"),
    ("nucleophilic_substitution", "https://en.wikipedia.org/wiki/Nucleophilic_substitution"),
    ("elimination_reaction", "https://en.wikipedia.org/wiki/Elimination_reaction"),
    ("acid_base", "https://en.wikipedia.org/wiki/Acid%E2%80%93base_reaction"),
    ("stereochemistry", "https://en.wikipedia.org/wiki/Stereochemistry"),
    ("nmr_spectroscopy", "https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy"),
    ("aldol_reaction", "https://en.wikipedia.org/wiki/Aldol_reaction"),
    ("diels_alder", "https://en.wikipedia.org/wiki/Diels%E2%80%93Alder_reaction"),
    ("grignard_reaction", "https://en.wikipedia.org/wiki/Grignard_reaction"),
    ("aromaticity", "https://en.wikipedia.org/wiki/Aromaticity"),
    ("redox", "https://en.wikipedia.org/wiki/Redox"),
    ("functional_group", "https://en.wikipedia.org/wiki/Functional_group"),
]

# Small chunks so the 3B model can focus on each one
MAX_CHUNK_CHARS = 800


def fetch_url(url: str) -> str:
    """Fetch and strip HTML, preserving paragraph structure."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "SaraBrain/0.1 (benchmark prep)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # Strip scripts and styles
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    # Strip navigation, footer, sidebar, references
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*class="[^"]*sidebar[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*class="[^"]*reflist[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)
    # Strip citation brackets [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Convert block tags to newlines
    text = re.sub(
        r"</?(p|div|br|h[1-6]|li|tr|td|th|blockquote|section|article)[^>]*>",
        "\n", text, flags=re.IGNORECASE
    )
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Collapse whitespace within lines, preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[\n\s]*", "\n\n", text)
    return text.strip()


def extract_body(text: str) -> str:
    """Try to extract just the article body, skipping nav/footer junk."""
    lines = text.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        # Skip very short lines (nav items, single words)
        if len(stripped) < 20:
            # But keep blank lines for paragraph breaks
            if not stripped and in_body:
                body_lines.append("")
            continue
        # Skip lines that look like navigation
        if any(kw in stripped.lower() for kw in [
            "jump to", "main menu", "move to sidebar", "toggle",
            "retrieved from", "categories:", "hidden categories",
            "privacy policy", "terms of use", "cookie statement",
            "creative commons", "wikipedia®", "powered by",
        ]):
            continue
        in_body = True
        body_lines.append(stripped)
    return "\n".join(body_lines)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into small chunks at sentence boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds the limit, split on sentences
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) > max_chars and sent_buf:
                    chunks.append(" ".join(sent_buf))
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(sent)
                sent_len += len(sent) + 1
            if sent_buf:
                chunks.append(" ".join(sent_buf))
            continue

        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    # Filter out chunks that are too short to contain useful facts
    return [c for c in chunks if len(c.strip()) > 50]


def main():
    out_dir = "benchmarks/chemistry_chunks"
    os.makedirs(out_dir, exist_ok=True)

    total_chunks = 0
    for name, url in SOURCES:
        print(f"  Fetching {name}... ", end="", flush=True)
        try:
            raw = fetch_url(url)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        body = extract_body(raw)
        chunks = chunk_text(body)
        print(f"{len(body)} chars → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            filename = f"{name}_{i+1:03d}.txt"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "w") as f:
                f.write(chunk)

        total_chunks += len(chunks)

    print(f"\n  Total: {total_chunks} chunk files in {out_dir}/")
    print(f"  Ingest with:")
    print(f"    for f in benchmarks/chemistry_chunks/*.txt; do")
    print(f'      sara-cortex --db GPQA_Diamond_chemistry_r1.db --no-llm -c "/ingest $f"')
    print(f"    done")


if __name__ == "__main__":
    main()
