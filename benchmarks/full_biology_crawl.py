#!/usr/bin/env python3
"""Full MMLU biology knowledge crawl.

Runs the curiosity-driven ingest on every core biology topic.
Resume-safe: skips topics already done.

Usage:
    python benchmarks/full_biology_crawl.py --db bio_full.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


TOPICS = [
    # High priority — covers most MMLU biology questions
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
    # MMLU-gap-filling topics that the first run showed Sara was missing
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--max-gaps", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--seek-wikis", type=int, default=3)
    args = parser.parse_args()

    progress_file = args.db + ".crawl_progress"
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


if __name__ == "__main__":
    main()
