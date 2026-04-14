#!/usr/bin/env python3
"""Download and chunk Wikipedia biology pages for MMLU high school bio."""

from __future__ import annotations

import html
import os
import re
import urllib.request

SOURCES = [
    ("gene", "https://en.wikipedia.org/wiki/Gene"),
    ("evolution", "https://en.wikipedia.org/wiki/Evolution"),
    ("cell_biology", "https://en.wikipedia.org/wiki/Cell_(biology)"),
    ("natural_selection", "https://en.wikipedia.org/wiki/Natural_selection"),
    ("population_genetics", "https://en.wikipedia.org/wiki/Population_genetics"),
    ("dna", "https://en.wikipedia.org/wiki/DNA"),
    ("species", "https://en.wikipedia.org/wiki/Species"),
    ("protein", "https://en.wikipedia.org/wiki/Protein"),
    ("rna", "https://en.wikipedia.org/wiki/RNA"),
    ("cell_membrane", "https://en.wikipedia.org/wiki/Cell_membrane"),
    ("meiosis", "https://en.wikipedia.org/wiki/Meiosis"),
    ("mitosis", "https://en.wikipedia.org/wiki/Mitosis"),
    ("mitochondrion", "https://en.wikipedia.org/wiki/Mitochondrion"),
    ("atp", "https://en.wikipedia.org/wiki/Adenosine_triphosphate"),
    ("mutation", "https://en.wikipedia.org/wiki/Mutation"),
    ("virus", "https://en.wikipedia.org/wiki/Virus"),
    ("transcription", "https://en.wikipedia.org/wiki/Transcription_(biology)"),
    ("translation", "https://en.wikipedia.org/wiki/Translation_(biology)"),
    ("photosynthesis", "https://en.wikipedia.org/wiki/Photosynthesis"),
    ("cellular_respiration", "https://en.wikipedia.org/wiki/Cellular_respiration"),
    ("enzyme", "https://en.wikipedia.org/wiki/Enzyme"),
    ("chromosome", "https://en.wikipedia.org/wiki/Chromosome"),
    ("ecosystem", "https://en.wikipedia.org/wiki/Ecosystem"),
    ("genetic_drift", "https://en.wikipedia.org/wiki/Genetic_drift"),
]

MAX_CHUNK_CHARS = 800


def fetch_url(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "SaraBrain/0.1 (benchmark prep)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*class="[^"]*sidebar[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*class="[^"]*reflist[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(
        r"</?(p|div|br|h[1-6]|li|tr|td|th|blockquote|section|article)[^>]*>",
        "\n", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[\n\s]*", "\n\n", text)
    return text.strip()


def extract_body(text: str) -> str:
    lines = text.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 20:
            if not stripped and in_body:
                body_lines.append("")
            continue
        if any(kw in stripped.lower() for kw in [
            "jump to", "main menu", "move to sidebar", "toggle",
            "retrieved from", "categories:", "hidden categories",
            "privacy policy", "terms of use", "cookie statement",
            "creative commons", "wikipedia®", "powered by",
            "from wikipedia, the free",
        ]):
            continue
        in_body = True
        body_lines.append(stripped)
    return "\n".join(body_lines)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
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
    return [c for c in chunks if len(c.strip()) > 50]


def main():
    out_dir = "benchmarks/biology_chunks"
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
            with open(os.path.join(out_dir, filename), "w") as f:
                f.write(chunk)
        total_chunks += len(chunks)
    print(f"\n  Total: {total_chunks} chunks in {out_dir}/")


if __name__ == "__main__":
    main()
