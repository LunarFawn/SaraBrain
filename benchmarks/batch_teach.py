#!/usr/bin/env python3
"""Batch teach facts into Sara Brain from a text file.

One fact per line. Blank lines and lines starting with # are ignored.

Usage:
    python benchmarks/batch_teach.py --db claude_phd.db --file benchmarks/claude_extracted_facts.txt
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--track", action="store_true", default=True,
                        help="Track which lines have been taught (resume)")
    args = parser.parse_args()

    from sara_brain.core.brain import Brain

    brain = Brain(args.db)
    stats = brain.stats()
    print(f"\n  Brain: {args.db}")
    print(f"  Before: {stats['neurons']} neurons, {stats['paths']} paths\n")

    # Track which lines have been taught (by content hash)
    progress_file = args.file + ".taught"
    already_taught = set()
    if args.track and os.path.exists(progress_file):
        with open(progress_file) as pf:
            already_taught = set(line.rstrip("\n") for line in pf)

    taught = 0
    skipped = 0
    failed = 0
    unchanged = 0

    with open(args.file) as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line or line.startswith("#"):
                continue

            if line in already_taught:
                unchanged += 1
                continue

            try:
                result = brain.teach(line)
                if result is not None:
                    taught += 1
                    if args.track:
                        with open(progress_file, "a") as pf:
                            pf.write(line + "\n")
                else:
                    skipped += 1
            except Exception as e:
                print(f"  FAIL: {line[:60]}... — {e}")
                failed += 1

    brain.conn.commit()
    stats = brain.stats()

    print(f"  Taught:    {taught}")
    print(f"  Skipped:   {skipped} (parser couldn't extract)")
    print(f"  Unchanged: {unchanged} (already taught)")
    print(f"  Failed:    {failed}")
    print(f"  After:     {stats['neurons']} neurons, {stats['paths']} paths\n")
    brain.close()


if __name__ == "__main__":
    main()
