#!/usr/bin/env python3
"""Audit how often Sara already knows the answer but routes wrong.

For each failing MMLU question, compare:
  - the regions the scorer actually selected (top_k)
  - the regions that contain paths strongly matching the correct
    answer's content lemmas

If the correct answer's content appears in a region NOT selected,
that's a routing failure — Sara has the knowledge but couldn't see
it. If the correct answer's content doesn't appear in ANY region,
that's a teaching gap — she genuinely doesn't know.

Usage:
    .venv/bin/python benchmarks/routing_audit.py \\
        --db biology2e.db \\
        --results benchmarks/mmlu_biology_round3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import spacy

from sara_brain.core.brain import Brain

sys.path.insert(0, "benchmarks")
from run_spacy_ch10 import (  # noqa: E402
    content_lemmas, load_region_paths,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--min-hits", type=int, default=2,
                    help="Min content-lemma overlap for a path to "
                         "count as knowing the answer.")
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_sm")
    brain = Brain(args.db)
    regions = [r[0] for r in brain.conn.execute(
        "SELECT name FROM regions ORDER BY name"
    ).fetchall()]

    print(f"Loading all {len(regions)} regions' paths …")
    all_paths: dict[str, list[dict]] = {}
    for region in regions:
        all_paths[region] = load_region_paths(brain, region, nlp)
    total_paths = sum(len(v) for v in all_paths.values())
    print(f"Loaded {total_paths} paths across {len(regions)} regions.\n")

    data = json.loads(Path(args.results).read_text())
    failing = [q for q in data["results"] if q["outcome"] != "correct"]

    knowledge_exists = 0         # correct-answer content exists in some region
    routing_miss = 0             # ...and that region was NOT in selected
    routing_hit_but_scored_wrong = 0  # it WAS in selected — scorer limit
    true_teaching_gap = 0        # content doesn't exist anywhere

    per_q_detail: list[dict] = []

    for q in failing:
        # Get correct answer's content lemmas
        correct_letter = q["correct"]
        correct_choice = next(
            (c for c in q["choices"] if c["letter"] == correct_letter),
            None,
        )
        if correct_choice is None:
            continue
        ans_doc = nlp(correct_choice["text"])
        ans_lemmas = set(content_lemmas(ans_doc))
        if not ans_lemmas:
            continue

        selected = set(q.get("regions") or [])

        # Find regions that contain paths with strong overlap to the
        # correct answer's content lemmas.
        knowing_regions: set[str] = set()
        sample_path: dict | None = None
        for region, paths in all_paths.items():
            for p in paths:
                hits = ans_lemmas & p["words"]
                if len(hits) >= args.min_hits:
                    knowing_regions.add(region)
                    if sample_path is None and not (selected & {region}):
                        sample_path = {
                            "region": region,
                            "source": p["source_text"][:70],
                            "hits": sorted(hits),
                        }
                    break

        if not knowing_regions:
            true_teaching_gap += 1
            per_q_detail.append({
                "id": q["id"], "status": "no_knowledge",
                "correct": correct_choice["text"][:60],
            })
        else:
            knowledge_exists += 1
            if knowing_regions & selected:
                routing_hit_but_scored_wrong += 1
                per_q_detail.append({
                    "id": q["id"],
                    "status": "routed_right_but_scored_wrong",
                    "correct": correct_choice["text"][:60],
                    "knowing_regions": sorted(knowing_regions)[:5],
                    "selected": sorted(selected),
                })
            else:
                routing_miss += 1
                per_q_detail.append({
                    "id": q["id"], "status": "routing_miss",
                    "correct": correct_choice["text"][:60],
                    "knowing_regions": sorted(knowing_regions)[:5],
                    "selected": sorted(selected),
                    "sample_missed_path": sample_path,
                })

    print("=" * 60)
    print(f"Failing questions inspected:           {len(failing)}")
    print(f"Knowledge exists in graph:             {knowledge_exists}")
    print(f"  ROUTED RIGHT, scorer picked wrong:   "
          f"{routing_hit_but_scored_wrong}")
    print(f"  ROUTING MISSED (wrong region):       {routing_miss}")
    print(f"Teaching gap (no matching path):       {true_teaching_gap}")
    print("=" * 60)

    # Print sample routing misses
    misses = [d for d in per_q_detail if d["status"] == "routing_miss"][:10]
    if misses:
        print("\nSample routing misses:")
        for m in misses:
            print(f"  Q{m['id']}: correct='{m['correct']}'")
            print(f"    selected: {m['selected']}")
            print(f"    knowing:  {m['knowing_regions']}")
            if m.get("sample_missed_path"):
                sp = m["sample_missed_path"]
                print(f"    path in {sp['region']}: "
                      f"'{sp['source']}' hits={sp['hits']}")

    brain.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
