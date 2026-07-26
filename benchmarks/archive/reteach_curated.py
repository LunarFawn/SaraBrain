#!/usr/bin/env python3
"""Replay curated teach statements into a fresh brain.db.

Input is a plain-text file — one teach statement per line, lines
starting with `#` are comments. Every non-comment line is a
per-fact judgment already made by the teacher; this script just
replays them in order.

Usage:
    .venv/bin/python benchmarks/reteach_curated.py \\
        --db brain.db \\
        --facts benchmarks/ch10_hand_curated.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sara_brain.core.brain import Brain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--facts", required=True, type=Path)
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(f"Refusing to overwrite {args.db}")
    brain = Brain(str(args.db))

    taught = skipped = 0
    for line in args.facts.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = brain.teach(line)
        if r is None:
            skipped += 1
            print(f"  ✗  {line}")
        else:
            taught += 1
    brain.conn.commit()
    print(f"\ntaught={taught} skipped={skipped}")
    print(f"neurons={brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"segments={brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    print(f"is_a segments={brain.conn.execute(chr(34)+'SELECT COUNT(*) FROM segments WHERE relation='+chr(39)+'is_a'+chr(39)+chr(34)).fetchone()[0] if False else brain.conn.execute('SELECT COUNT(*) FROM segments WHERE relation=?', ('is_a',)).fetchone()[0]}")
    print(f"part_of segments={brain.conn.execute('SELECT COUNT(*) FROM segments WHERE relation=?', ('part_of',)).fetchone()[0]}")
    print(f"paths={brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")


if __name__ == "__main__":
    main()
