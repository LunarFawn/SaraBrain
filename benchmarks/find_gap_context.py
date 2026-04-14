#!/usr/bin/env python3
"""Find sections of source text relevant to Sara's knowledge gaps.

This is the directed re-read phase. Given a brain and a source file,
output the text sections that mention concepts Sara has thin paths on.

Usage:
    python benchmarks/find_gap_context.py --db trivia_brain.db \\
        --source benchmarks/biology_chunks/gene_030.txt
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--source", help="Single source file")
    parser.add_argument("--sources-dir",
                        help="Directory of source files to scan")
    parser.add_argument("--max-gaps", type=int, default=10,
                        help="Focus on top N thinnest concepts")
    parser.add_argument("--min-paths", type=int, default=1,
                        help="Only gaps with at least N paths "
                             "(excludes topics Sara has never heard of)")
    parser.add_argument("--max-sections", type=int, default=3,
                        help="Max sections to show per gap concept")
    args = parser.parse_args()

    from sara_brain.core.brain import Brain

    brain = Brain(args.db)
    gaps = brain.knowledge_gaps()
    # Filter to gaps Sara has SOME knowledge on (she's heard of it)
    partial = [(t, d) for t, d in gaps if d >= args.min_paths][:args.max_gaps]

    if not partial:
        print("No thin-knowledge gaps found.")
        return

    print(f"\n  Sara's thinnest knowledge areas (top {len(partial)}):\n")
    for t, d in partial:
        print(f"    {d} paths — {t}")
    print()

    # Collect source texts to scan
    texts = []
    if args.source:
        with open(args.source) as f:
            texts.append((args.source, f.read()))
    elif args.sources_dir:
        for fn in sorted(os.listdir(args.sources_dir)):
            if fn.endswith(".txt"):
                path = os.path.join(args.sources_dir, fn)
                with open(path) as f:
                    texts.append((path, f.read()))

    if not texts:
        print("No source text to scan. Use --source or --sources-dir.")
        return

    # For each gap concept, find sentences mentioning it
    print(f"\n  ═══ DIRECTED READ: finding sections about Sara's gaps ═══\n")

    for concept, depth in partial:
        concept_lower = concept.lower()
        matches = []
        for source_path, text in texts:
            # Split into sentences
            sentences = text.replace("\n", " ").split(".")
            for i, sent in enumerate(sentences):
                if concept_lower in sent.lower() and len(sent.strip()) > 30:
                    matches.append((source_path, sent.strip()))

        if not matches:
            continue

        print(f"  ── {concept} ({depth} paths currently) ──")
        shown = 0
        seen_texts = set()
        for src, sent in matches:
            if sent[:80] in seen_texts:
                continue
            seen_texts.add(sent[:80])
            print(f"    [{os.path.basename(src)}]")
            print(f"    {sent.strip()[:300]}")
            print()
            shown += 1
            if shown >= args.max_sections:
                break
        print()


if __name__ == "__main__":
    main()
