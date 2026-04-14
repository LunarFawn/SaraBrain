#!/usr/bin/env python3
"""Curiosity-driven ingest — the student-reading pattern.

Instead of dumb chunking, Sara reads like a student studying:

  Pass 1 — Skim:          LLM reads the source, extracts general facts
  Pass 2 — Self-assess:   Find concepts mentioned that Sara has thin paths on
  Pass 3 — Targeted dive: For each gap, re-read sections about that concept,
                          focusing the LLM on what Sara needs
  Pass 4 — Verify:        Sample what Sara learned; LLM checks it

Repeats until gaps close or max iterations reached.

Usage:
    python benchmarks/curious_ingest.py --db trivia_brain.db \\
        --source benchmarks/biology_source/gene.txt

    # Or with a URL that gets fetched first:
    python benchmarks/curious_ingest.py --db trivia_brain.db \\
        --url https://en.wikipedia.org/wiki/Gene
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.request


def fetch_url(url: str) -> str:
    """Fetch a URL and return cleaned text."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "SaraBrain/0.1 (curious ingest)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL)
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


def call_ollama(prompt: str, system: str, model: str,
                base_url: str, max_tokens: int = 2000) -> str | None:
    """Call Ollama and return response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return None


def parse_facts(raw: str) -> list[str]:
    """Parse LLM output into clean fact statements."""
    facts = []
    for line in raw.splitlines():
        s = line.strip().lstrip("-*•·0123456789.)").strip()
        if not s or s.upper() == "NONE":
            continue
        if len(s) > 200:
            continue
        if any(kw in s.lower() for kw in ("http", "www", "doi:")):
            continue
        facts.append(s)
    seen = set()
    result = []
    for f in facts:
        low = f.lower()
        if low not in seen:
            seen.add(low)
            result.append(f)
    return result


# Pass 1: Cold read — general extraction
SKIM_SYSTEM = """You are a careful fact extractor for a knowledge graph.

Read the document and extract factually correct, verifiable statements.

Rules:
- Each statement must be factually correct as written. If unsure, omit it.
- One fact per line. Simple format: <subject> is/has/requires <property>.
- Use simple subject-predicate-object sentences.
- Include specific numbers, dates, named concepts, mechanisms.
- Do NOT extract: author names, DOIs, citation markers, URLs.
- Do NOT extract: vague statements without specifics.
- Simple words only. No explanations, no commentary. Just facts."""


# Pass 3: Targeted dive — focused extraction on a gap concept
TARGETED_SYSTEM_TEMPLATE = """You are extracting facts about a specific topic.

The brain you are teaching has only surface-level knowledge of: {concept}

Read the text below and extract EVERY factual statement about {concept}
that you can find. Focus on:
- What {concept} is (definition)
- What {concept} does (function/mechanism)
- How {concept} works (process)
- What {concept} is made of (structure/composition)
- Numbers, conditions, properties specific to {concept}
- How {concept} relates to other concepts

Rules:
- Each statement must be factually correct.
- One fact per line. Format: <subject> is/has/requires <property>.
- Stay focused on {concept} and closely related ideas.
- Simple words. No explanations. Just facts.
- If the text has nothing about {concept}, output NONE."""


def pass1_skim(brain, text: str, model: str, base_url: str) -> int:
    """Skim the whole source for general facts."""
    # Break very long texts into large chunks (2500 chars)
    # to fit in context but keep them big enough to have context
    chunks = []
    current = []
    current_len = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para or len(para) < 30:
            continue
        if current_len + len(para) > 2500 and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))

    total_facts = 0
    for i, chunk in enumerate(chunks):
        print(f"    chunk {i+1}/{len(chunks)}... ", end="", flush=True)
        raw = call_ollama(chunk, SKIM_SYSTEM, model, base_url)
        if raw is None:
            print("(failed)")
            continue
        facts = parse_facts(raw)
        added = 0
        for fact in facts:
            if brain.teach(fact) is not None:
                added += 1
        brain.conn.commit()
        total_facts += added
        print(f"+{added} facts")
    return total_facts


def pass2_assess(brain, text: str) -> list[tuple[str, int]]:
    """Find concepts in the text that Sara has thin paths on."""
    # Get concepts Sara knows that appear in the text
    mentioned = brain.concepts_mentioned(text)
    # Filter to those below the curiosity threshold
    gaps = []
    for concept in mentioned:
        d = brain.depth(concept)
        if d < brain.CURIOSITY_THRESHOLD:
            gaps.append((concept, d))
    # Biggest gaps first (lowest depth)
    gaps.sort(key=lambda x: x[1])
    return gaps


def pass3_directed(brain, text: str, gap: tuple[str, int],
                   model: str, base_url: str, max_sections: int = 5) -> int:
    """Directed re-read focused on one gap concept."""
    concept, depth = gap
    concept_lower = concept.lower()

    # Find paragraphs that mention the concept
    paragraphs = text.split("\n\n")
    relevant = [p for p in paragraphs if concept_lower in p.lower()
                and len(p.strip()) > 30]

    if not relevant:
        return 0

    # Take the most relevant sections (longest ones are usually substantive)
    relevant = sorted(relevant, key=len, reverse=True)[:max_sections]
    combined = "\n\n".join(relevant)

    system = TARGETED_SYSTEM_TEMPLATE.format(concept=concept)
    raw = call_ollama(combined, system, model, base_url)
    if raw is None:
        return 0

    facts = parse_facts(raw)
    added = 0
    for fact in facts:
        if brain.teach(fact) is not None:
            added += 1
    brain.conn.commit()
    return added


def articulate_understanding(brain, text: str) -> str:
    """Sara articulates what she still doesn't fully understand from this doc."""
    mentioned = brain.concepts_mentioned(text)
    thin = [(c, brain.depth(c)) for c in mentioned
            if brain.depth(c) < brain.CURIOSITY_THRESHOLD]
    thin.sort(key=lambda x: x[1])

    if not thin:
        return "  Sara has satisfied depth on all concepts from this document."

    lines = [f"  Sara still has thin knowledge on {len(thin)} concept(s):"]
    for concept, depth in thin[:15]:
        lines.append(f"    {depth:4d} paths — {concept}")
    if len(thin) > 15:
        lines.append(f"    ... and {len(thin) - 15} more")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--source", help="Text file path")
    parser.add_argument("--url", help="URL to fetch")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--max-gaps", type=int, default=10,
                        help="Max gap concepts to target in pass 3")
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Max rounds of gap-closing")
    args = parser.parse_args()

    if not args.source and not args.url:
        print("Error: --source or --url required")
        return 1

    # Load source text
    if args.url:
        print(f"\n  Fetching {args.url}...")
        text = fetch_url(args.url)
        source_label = args.url
    else:
        with open(args.source) as f:
            text = f.read()
        source_label = args.source

    print(f"  Source: {len(text)} chars\n")

    from sara_brain.core.brain import Brain

    brain = Brain(args.db)
    stats = brain.stats()
    print(f"  Brain: {args.db}")
    print(f"  Before: {stats['neurons']} neurons, {stats['paths']} paths\n")

    # Configure LLM
    if not brain.settings_repo.get("llm_model"):
        brain.settings_repo.set("llm_provider", "ollama")
        brain.settings_repo.set("llm_model", args.model)
        brain.settings_repo.set("llm_api_url", args.base_url)
        brain.conn.commit()

    start = time.time()

    # ── Pass 1: Skim ──
    print("  ═══ PASS 1: Skim ═══")
    p1_facts = pass1_skim(brain, text, args.model, args.base_url)
    print(f"  Pass 1 learned: {p1_facts} facts\n")

    # ── Iterative Pass 2 + 3 ──
    for iteration in range(args.max_iterations):
        print(f"  ═══ PASS 2 (iter {iteration+1}): Self-assess ═══")
        gaps = pass2_assess(brain, text)
        if not gaps:
            print("  No gaps found in this document's concepts.\n")
            break
        print(f"  Found {len(gaps)} gap concepts (below {brain.CURIOSITY_THRESHOLD} paths)")
        targets = gaps[:args.max_gaps]
        for c, d in targets[:5]:
            print(f"    {d:4d} paths — {c}")
        if len(targets) > 5:
            print(f"    ... and {len(targets) - 5} more")
        print()

        print(f"  ═══ PASS 3 (iter {iteration+1}): Directed re-read ═══")
        p3_total = 0
        for concept, depth in targets:
            print(f"    ── {concept} ({depth} paths) ── ", end="", flush=True)
            added = pass3_directed(brain, text, (concept, depth),
                                    args.model, args.base_url)
            p3_total += added
            new_depth = brain.depth(concept)
            print(f"+{added} facts → now {new_depth} paths")
        print(f"  Pass 3 learned: {p3_total} facts\n")

        if p3_total == 0:
            print("  No new facts from directed reads. Stopping.\n")
            break

    # ── Report ──
    print("  ═══ SELF-REPORT ═══")
    print(articulate_understanding(brain, text))
    print()

    elapsed = time.time() - start
    stats = brain.stats()
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Final: {stats['neurons']} neurons, {stats['paths']} paths")
    brain.close()


if __name__ == "__main__":
    main()
