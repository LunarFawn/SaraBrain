#!/usr/bin/env python3
"""Retrain the biology brain using the verified training pipeline.

Runs curious_ingest with tentative writes + source provenance on the
34 Wikipedia biology topics. Facts from a single source stay invisible
until a second source confirms them — cross-confirmed knowledge becomes
visible, single-source assertions remain hidden.

Compared to bio_full.db (built with the old unverified pipeline), this
produces a smaller visible graph but higher-quality knowledge: the
things you can query for are things at least two independent sources
agreed on.

Usage:
    python benchmarks/retrain_bio.py --db bio_v2.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


TOPICS = [
    # High priority — core MMLU biology coverage
    ("Gene", "https://en.wikipedia.org/wiki/Gene"),
    ("Evolution", "https://en.wikipedia.org/wiki/Evolution"),
    ("Cell_(biology)", "https://en.wikipedia.org/wiki/Cell_(biology)"),
    ("Natural_selection", "https://en.wikipedia.org/wiki/Natural_selection"),
    ("Population_genetics", "https://en.wikipedia.org/wiki/Population_genetics"),
    ("DNA", "https://en.wikipedia.org/wiki/DNA"),
    ("Species", "https://en.wikipedia.org/wiki/Species"),
    ("Protein", "https://en.wikipedia.org/wiki/Protein"),
    ("RNA", "https://en.wikipedia.org/wiki/RNA"),
    ("Cell_membrane", "https://en.wikipedia.org/wiki/Cell_membrane"),
    ("Meiosis", "https://en.wikipedia.org/wiki/Meiosis"),
    ("Mitosis", "https://en.wikipedia.org/wiki/Mitosis"),
    ("Mitochondrion", "https://en.wikipedia.org/wiki/Mitochondrion"),
    ("ATP", "https://en.wikipedia.org/wiki/Adenosine_triphosphate"),
    ("Mutation", "https://en.wikipedia.org/wiki/Mutation"),
    ("Virus", "https://en.wikipedia.org/wiki/Virus"),
    ("Transcription", "https://en.wikipedia.org/wiki/Transcription_(biology)"),
    ("Translation", "https://en.wikipedia.org/wiki/Translation_(biology)"),
    ("Photosynthesis", "https://en.wikipedia.org/wiki/Photosynthesis"),
    ("Cellular_respiration", "https://en.wikipedia.org/wiki/Cellular_respiration"),
    ("Enzyme", "https://en.wikipedia.org/wiki/Enzyme"),
    ("Chromosome", "https://en.wikipedia.org/wiki/Chromosome"),
    ("Ecosystem", "https://en.wikipedia.org/wiki/Ecosystem"),
    ("Genetic_drift", "https://en.wikipedia.org/wiki/Genetic_drift"),
    # MMLU-gap topics
    ("Immune_system", "https://en.wikipedia.org/wiki/Immune_system"),
    ("Osmosis", "https://en.wikipedia.org/wiki/Osmosis"),
    ("Diffusion", "https://en.wikipedia.org/wiki/Diffusion"),
    ("Meristem", "https://en.wikipedia.org/wiki/Meristem"),
    ("Convergent_evolution", "https://en.wikipedia.org/wiki/Convergent_evolution"),
    ("Biogeochemical_cycle", "https://en.wikipedia.org/wiki/Biogeochemical_cycle"),
    ("Lipid", "https://en.wikipedia.org/wiki/Lipid"),
    ("Antibody", "https://en.wikipedia.org/wiki/Antibody"),
    ("Sexual_selection", "https://en.wikipedia.org/wiki/Sexual_selection"),
    ("Adaptation", "https://en.wikipedia.org/wiki/Adaptation"),
    ("Gene_expression", "https://en.wikipedia.org/wiki/Gene_expression"),
]


def report_stats(brain) -> None:
    """Print tentative vs confirmed breakdown."""
    c = brain.conn.cursor()
    total_segs = c.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    tentative = c.execute(
        "SELECT COUNT(*) FROM segments WHERE strength < 0.5"
    ).fetchone()[0]
    confirmed = c.execute(
        "SELECT COUNT(*) FROM segments WHERE strength >= 0.5"
    ).fetchone()[0]
    distinct_sources = c.execute(
        "SELECT COUNT(DISTINCT source_label) FROM segment_sources"
    ).fetchone()[0]
    total_source_records = c.execute(
        "SELECT COUNT(*) FROM segment_sources"
    ).fetchone()[0]

    stats = brain.stats()
    print()
    print(f"  {'='*60}")
    print(f"  Brain: {stats['neurons']} neurons, "
          f"{stats['paths']} paths, {total_segs} segments")
    print(f"  Tentative segments (strength < 0.5): {tentative}")
    print(f"  Confirmed segments (strength >= 0.5): {confirmed}")
    print(f"  Distinct sources: {distinct_sources}")
    print(f"  Total source records: {total_source_records}")
    if total_segs > 0:
        pct = confirmed / total_segs * 100
        print(f"  Confirmation rate: {pct:.1f}%")
    print(f"  {'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--max-gaps", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--seek-wikis", type=int, default=3)
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing DB and progress file")
    args = parser.parse_args()

    progress_file = args.db + ".crawl_progress"

    if args.fresh:
        for f in [args.db, args.db + "-shm", args.db + "-wal", progress_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  removed {f}")

    # Initialize a fresh brain (creates schema including segment_sources)
    from sara_brain.core.brain import Brain
    brain = Brain(args.db)
    brain.close()

    done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            done = set(line.strip() for line in f if line.strip())
        print(f"  Resuming — {len(done)} topics already crawled\n")

    total = len(TOPICS)
    remaining = [(n, u) for n, u in TOPICS if n not in done]
    print(f"  {len(remaining)}/{total} topics to crawl\n")

    start = time.time()
    script = os.path.join(os.path.dirname(__file__), "curious_ingest.py")

    for i, (name, url) in enumerate(remaining):
        idx = TOPICS.index((name, url)) + 1
        elapsed = time.time() - start
        if i > 0:
            rate = i / elapsed * 60
            remaining_time = (len(remaining) - i) / rate if rate > 0 else 0
            eta = f", ~{remaining_time:.0f}m remaining"
        else:
            eta = ""

        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {name}{eta}")
        print(f"  URL: {url}")
        print(f"{'='*60}\n", flush=True)

        cmd = [
            sys.executable, script,
            "--db", args.db,
            "--url", url,
            "--model", args.model,
            "--max-gaps", str(args.max_gaps),
            "--max-iterations", str(args.max_iterations),
            "--seek-wikis", str(args.seek_wikis),
        ]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            with open(progress_file, "a") as f:
                f.write(name + "\n")
        else:
            print(f"  WARNING: crawl failed for {name}, skipping\n")

    print(f"\n  Total crawl time: {(time.time() - start)/60:.1f} minutes")

    # Final stats
    brain = Brain(args.db)
    report_stats(brain)
    brain.close()


if __name__ == "__main__":
    main()
