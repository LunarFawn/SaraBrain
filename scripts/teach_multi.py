#!/usr/bin/env python3
"""Teach multiple fact files into a single brain.db.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from sara_brain.core.brain import Brain

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--sources", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    if args.db.exists():
        print(f"Adding to existing {args.db}")
    brain = Brain(str(args.db))

    for source in args.sources:
        print(f"Teaching from {source}...")
        sentences = [
            l.strip() for l in source.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        print(f"{len(sentences)} source sentences")

        total_sub_facts = 0
        t0 = time.time()
        for i, s in enumerate(sentences, 1):
            n = brain.teach_expanded(s)
            total_sub_facts += n
        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.1f}s, sub-facts: {total_sub_facts}")

    print()
    print(f"Final Stats for {args.db}:")
    print(f"neurons:  {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"paths:    {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")
    print(f"segments: {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")

if __name__ == "__main__":
    main()
