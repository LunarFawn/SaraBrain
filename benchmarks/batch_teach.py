#!/usr/bin/env python3
"""Batch teach facts into Sara Brain from a text file.

One fact per line. Blank lines and lines starting with # are ignored.

When --region is given, facts are taught into the named compartmentalized
region (e.g., cell_division, meiosis). The region is created if it does
not already exist. Without --region, facts are taught into the main
(unprefixed) tables as before.

Usage:
    # Flat teaching (original behavior)
    .venv/bin/python benchmarks/batch_teach.py \\
        --db claude_phd.db --file facts.txt

    # Regional teaching (compartmentalized brain)
    .venv/bin/python benchmarks/batch_teach.py \\
        --db claude_taught.db --file ch10_1_cell_division_facts.txt \\
        --region cell_division
"""

from __future__ import annotations

import argparse
import os

from sara_brain.core.brain import Brain


def _build_regional_learner(brain: Brain, region: str):
    """Construct a Learner wired to the given region's prefixed repos.

    The region is created if it does not already exist (idempotent).
    """
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo
    from sara_brain.storage.path_repo import PathRepo
    from sara_brain.core.learner import Learner
    from sara_brain.parsing.statement_parser import StatementParser
    from sara_brain.parsing.taxonomy import Taxonomy

    brain.db.create_region(region)

    nr = NeuronRepo(brain.conn, prefix=region)
    sr = SegmentRepo(brain.conn, prefix=region)
    pr = PathRepo(brain.conn, prefix=region)
    # Parser + taxonomy operate on labels, not tables — shared across regions.
    parser_obj = StatementParser(taxonomy=Taxonomy())
    return Learner(parser_obj, nr, sr, pr)


def _region_stats(brain: Brain, region: str | None) -> tuple[int, int]:
    cur = brain.conn.cursor()
    try:
        if region:
            n = cur.execute(f"SELECT COUNT(*) FROM {region}_neurons").fetchone()[0]
            p = cur.execute(f"SELECT COUNT(*) FROM {region}_paths").fetchone()[0]
        else:
            n = cur.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
            p = cur.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
    except Exception:
        # Region tables may not exist yet (fresh region) — treat as empty.
        return 0, 0
    return n, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--region", default=None,
                        help="Region name (e.g. cell_division). If unset, "
                             "teach into the main unprefixed tables.")
    parser.add_argument("--track", action="store_true", default=True,
                        help="Track which lines have been taught (resume)")
    args = parser.parse_args()

    brain = Brain(args.db)

    scope = f"region={args.region}" if args.region else "flat (no region)"
    before_n, before_p = _region_stats(brain, args.region)
    print(f"\n  Brain: {args.db}  ({scope})")
    print(f"  Before: {before_n} neurons, {before_p} paths\n")

    # Use a regional Learner when --region is set; otherwise Brain.teach().
    if args.region:
        learner = _build_regional_learner(brain, args.region)

        def teach(line: str):
            return learner.learn(line, apply_filter=False)
    else:
        def teach(line: str):
            return brain.teach(line)

    # Track which lines have been taught (by content hash)
    progress_file = args.file + ".taught"
    if args.region:
        progress_file = args.file + f".{args.region}.taught"
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
                result = teach(line)
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
    after_n, after_p = _region_stats(brain, args.region)

    print(f"  Taught:    {taught}")
    print(f"  Skipped:   {skipped} (parser couldn't extract)")
    print(f"  Unchanged: {unchanged} (already taught)")
    print(f"  Failed:    {failed}")
    print(f"  After:     {after_n} neurons (+{after_n-before_n}), "
          f"{after_p} paths (+{after_p-before_p})\n")
    brain.close()


if __name__ == "__main__":
    main()
