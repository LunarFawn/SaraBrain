#!/usr/bin/env python3
"""Compartmentalized Study Tool — each section gets its own brain region.

Eliminates cross-activation noise by isolating concepts. "Cell" in
mitosis is a different neuron than "cell" in cancer. They don't cross.

Study flow:
1. Split source text by section headers
2. Create a brain region per section
3. Study each section into its own region (memorize → recite → consolidate)
4. Query: detect which region(s) a question targets, only activate those

Usage:
    python benchmarks/study_compartmentalized.py --db compartment.db \\
        --source training_material/ch10_cell_reproduction.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.parsing.statement_parser import StatementParser
from sara_brain.parsing.taxonomy import Taxonomy
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo
from sara_brain.storage.path_repo import PathRepo
from sara_brain.core.learner import Learner


# ── LLM ──

def call_ollama(prompt: str, system: str, model: str,
                base_url: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 500},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


# ── Text ──

def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) by section headers.

    Detects patterns like "10.1 Cell Division" or "CHAPTER 10".
    Returns list of (region_name, text) tuples.
    """
    # Find section headers: "10.1 Title", "10.2 Title", etc.
    pattern = re.compile(r'^(\d+\.\d+)\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("general", text)]

    sections = []
    for i, match in enumerate(matches):
        num = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        # Create a clean region name from the title
        region_name = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        if not region_name:
            region_name = f"section_{num.replace('.', '_')}"

        sections.append((region_name, section_text))

    return sections


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]


def split_sentences(paragraph: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sents if len(s.strip()) > 10]


# ── Fact extraction ──

EXTRACT_SYSTEM = """You are extracting facts from a sentence for a knowledge graph.

Rules:
- Convert into simple statements. Formats accepted:
  "X is Y" (attributes)
  "X causes Y" (causation)
  "X precedes Y" or "X follows Y" (sequence)
  "X prevents Y" (prevention)
  "X contains Y" or "X includes Y" (composition)
  "X requires Y" (dependency)
  "X [verb] during Y" (temporal)
- One fact per line
- Simple words only
- No citations, author names, or reference numbers
- If no teachable fact exists, output NONE
- Maximum 3 facts per sentence"""


def extract_facts(sentence: str, model: str, base_url: str) -> list[str]:
    raw = call_ollama(sentence, EXTRACT_SYSTEM, model, base_url)
    if not raw or "NONE" in raw.upper():
        return []
    facts = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*•·0123456789.)")
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 8 and len(cleaned) < 200:
            if "none" not in cleaned.lower():
                facts.append(cleaned)
    return facts[:3]


# ── Study one section into its own region ──

def study_section(brain: Brain, region_name: str, section_text: str,
                  model: str, base_url: str) -> dict:
    """Study a section into its own isolated region.

    Creates the region tables if they don't exist, then extracts
    facts and teaches them into the region-specific tables.
    """
    # Create region tables
    brain.db.create_region(region_name)

    # Region-specific repos
    nr = NeuronRepo(brain.conn, prefix=region_name)
    sr = SegmentRepo(brain.conn, prefix=region_name)
    pr = PathRepo(brain.conn, prefix=region_name)

    # Region-specific learner
    parser = StatementParser(Taxonomy())
    learner = Learner(parser, nr, sr, pr)

    paragraphs = split_paragraphs(section_text)
    total_taught = 0
    total_rejected = 0
    total_extracted = 0

    for pi, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        for sentence in sentences:
            facts = extract_facts(sentence, model, base_url)
            total_extracted += len(facts)

            for fact in facts:
                result = learner.learn(fact)
                if result:
                    total_taught += 1
                else:
                    total_rejected += 1

    brain.conn.commit()

    stats = {
        "region": region_name,
        "paragraphs": len(paragraphs),
        "extracted": total_extracted,
        "taught": total_taught,
        "rejected": total_rejected,
        "neurons": nr.count(),
        "paths": pr.count(),
    }
    return stats


# ── Query with region selection ──

def detect_regions(question: str, brain: Brain,
                   regions: list[str]) -> list[str]:
    """Detect which region(s) a question is about.

    Checks which regions contain neurons matching the question's
    key concepts. Only matching regions will be queried.
    """
    words = re.findall(r"[a-z][a-z'-]+", question.lower())
    content_words = [w for w in words if len(w) >= 4]

    matched = []
    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        hits = 0
        for word in content_words:
            n = nr.get_by_label(word)
            if n is not None:
                hits += 1
        if hits > 0:
            matched.append((region, hits))

    # Sort by hit count, return region names
    matched.sort(key=lambda x: -x[1])
    return [r for r, _ in matched] if matched else regions[:1]


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--limit-sections", type=int, default=0)
    args = ap.parse_args()

    with open(args.source) as f:
        text = f.read()

    sections = split_sections(text)
    if args.limit_sections > 0:
        sections = sections[:args.limit_sections]

    brain = Brain(args.db)

    print(f"\n  ╔{'═'*55}╗")
    print(f"  ║  Compartmentalized Study — one region per section   ║")
    print(f"  ╚{'═'*55}╝")
    print(f"  Brain: {args.db}")
    print(f"  Source: {args.source}")
    print(f"  Sections: {len(sections)}\n")

    for name, _ in sections:
        print(f"    • {name}")
    print()

    start = time.time()
    all_stats = []
    region_names = []

    for si, (region_name, section_text) in enumerate(sections):
        print(f"  ═══ Section {si+1}/{len(sections)}: {region_name} ═══",
              flush=True)

        stats = study_section(
            brain, region_name, section_text,
            args.model, args.base_url,
        )
        all_stats.append(stats)
        region_names.append(region_name)

        print(f"    extracted: {stats['extracted']}, "
              f"taught: {stats['taught']}, "
              f"rejected: {stats['rejected']}, "
              f"neurons: {stats['neurons']}, "
              f"paths: {stats['paths']}\n", flush=True)

    elapsed = time.time() - start

    # Summary
    total_taught = sum(s["taught"] for s in all_stats)
    total_rejected = sum(s["rejected"] for s in all_stats)
    total_neurons = sum(s["neurons"] for s in all_stats)
    total_paths = sum(s["paths"] for s in all_stats)

    print(f"  {'='*55}")
    print(f"  COMPARTMENTALIZED STUDY REPORT")
    print(f"  {'='*55}")
    print(f"  Sections:  {len(sections)}")
    print(f"  Regions:   {', '.join(region_names)}")
    print(f"  Total taught:   {total_taught}")
    print(f"  Total rejected: {total_rejected}")
    print(f"  Total neurons:  {total_neurons} (across all regions)")
    print(f"  Total paths:    {total_paths} (across all regions)")
    print(f"  Time:           {elapsed/60:.1f} min")
    print(f"  {'='*55}")
    print()

    # Show per-region breakdown
    print(f"  Per-region breakdown:")
    for s in all_stats:
        print(f"    {s['region']:30s}  {s['paths']:4d} paths  "
              f"{s['neurons']:4d} neurons")
    print()

    # Save region list for the query tool
    meta = {
        "regions": region_names,
        "source": args.source,
        "stats": all_stats,
    }
    meta_path = args.db + ".regions.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Region metadata saved to {meta_path}")

    brain.close()


if __name__ == "__main__":
    main()
