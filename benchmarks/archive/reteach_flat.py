#!/usr/bin/env python3
"""Re-teach all extracted biology facts into a flat brain.db.

Reads benchmarks/biology2e_facts/ch*_facts.txt and teaches each fact
via the regular Brain.teach() path. Learner now detects compound
subjects/objects and emits IS-A edges on ingest, so the resulting
brain.db has the hierarchy baked in from a single pass.

Usage:
    .venv/bin/python benchmarks/reteach_flat.py --db brain.db \\
        --facts-dir benchmarks/biology2e_facts
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sara_brain.core.brain import Brain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--facts-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.db} — move it aside first."
        )

    brain = Brain(str(args.db))
    fact_files = sorted(args.facts_dir.glob("ch*_facts.txt"))
    print(f"Found {len(fact_files)} chapter files")

    total_taught = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.time()

    for path in fact_files:
        region_label = path.stem
        taught = skipped = failed = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                result = brain.learner.learn(
                    line, source_label=region_label, apply_filter=False,
                )
            except Exception:
                failed += 1
                continue
            if result is None:
                skipped += 1
            else:
                taught += 1
        brain.conn.commit()
        total_taught += taught
        total_skipped += skipped
        total_failed += failed
        print(f"  {path.name}: taught={taught} skipped={skipped} failed={failed}")

    elapsed = time.time() - t0
    print()
    print(f"  taught:  {total_taught}")
    print(f"  skipped: {total_skipped}")
    print(f"  failed:  {total_failed}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
