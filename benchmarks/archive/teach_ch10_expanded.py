#!/usr/bin/env python3
"""Teach ch10 into a fresh brain.db using grammar-expansion decomposition.

Walks every sentence of ch10_facts.txt and calls Brain.teach_expanded,
which uses spaCy dep-parse to emit every SVO sub-fact it can extract
(primary + adjective + prep-phrase + adverb + relative clause
modifiers), then teaches each as a separate path.

Usage:
    .venv/bin/python benchmarks/teach_ch10_expanded.py --db brain.db
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sara_brain.core.brain import Brain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--source", type=Path,
                    default=Path("benchmarks/biology2e_facts/ch10_facts.txt"))
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(f"Refusing to overwrite {args.db}")
    brain = Brain(str(args.db))

    sentences = [
        l.strip() for l in args.source.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    print(f"{len(sentences)} source sentences")

    total_sub_facts = 0
    t0 = time.time()
    for i, s in enumerate(sentences, 1):
        n = brain.teach_expanded(s)
        total_sub_facts += n
        if i % 25 == 0 or i == len(sentences):
            print(f"  [{i}/{len(sentences)}] total sub-facts so far: {total_sub_facts}")
    elapsed = time.time() - t0

    print()
    print(f"elapsed: {elapsed:.1f}s")
    print(f"source sentences: {len(sentences)}")
    print(f"sub-facts taught: {total_sub_facts}")
    print(f"neurons:  {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"paths:    {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")
    print(f"segments: {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    verb_count = brain.conn.execute(
        "SELECT COUNT(*) FROM segments s "
        "JOIN neurons n2 ON n2.id=s.target_id "
        "WHERE s.relation='is_a' AND n2.label='verb'"
    ).fetchone()[0]
    print(f"learned verbs: {verb_count}")


if __name__ == "__main__":
    main()
